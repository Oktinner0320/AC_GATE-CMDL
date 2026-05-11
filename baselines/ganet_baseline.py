"""Lightweight GA-Net (Gated Attention Network) baseline.

Adapts the classical Gated Attention Network of Xue et al. 2020
("Not All Attention Is Needed: Gated Attention Network for Sequence Data")
to the AC-GATE panel forecasting interface used by the LSTM and TFT
baselines. The model preserves the GA-Net core idea — an auxiliary
input-dependent gate network selects which positions are passed through
the main self-attention block — while operating on the same lag-window
view that the LSTM / TFT baselines consume.

Design notes (fair-comparison, parity with the other baselines):
* Identical forward signature to ``PlainLSTMBaseline`` / ``TFTBaseline``.
  The runner only needs to swap the model class.
* Re-uses the entity embedding + static-feature projection contract so
  the panel side-information is injected the same way.
* Same backbone budget: ``cfg.d_model``, ``cfg.lstm_layers``,
  ``cfg.dropout``, ``cfg.max_lag``. No extra task-specific
  hyper-parameters that the LSTM/TFT baselines do not also receive.
* The aux gate operates over the K lag positions per time step, mirroring
  the original GA-Net's "auxiliary network produces a soft mask over
  sequence positions" pattern. During training a Gumbel-sigmoid
  relaxation gives differentiable selection; at evaluation we use the
  deterministic sigmoid, as the original paper recommends.
* No shortcut on ``X_t``: the input contract matches CMDL's lag-window
  visibility (the same constraint applied to PlainLSTM / TFT baselines).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from config.cmdl_config import CMDLConfig
from model.backbone import GateAddNorm
from model.lag_gate import GatedResidualNetwork


@dataclass(slots=True)
class GANetBaselineOutput:
    """Structured outputs returned by the GA-Net baseline."""

    y_pred: torch.Tensor
    sequence: torch.Tensor
    final_state: torch.Tensor
    aux_gate_weights: torch.Tensor  # [B, T-K, K] soft per-lag gate values


def _gumbel_sigmoid(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Gumbel-sigmoid relaxation as used in the GA-Net paper.

    Implements ``sigmoid((logits + g1 - g2) / tau)`` with two independent
    Gumbel(0, 1) samples, which is the standard binary-Concrete relaxation
    of a Bernoulli draw. Returns soft gate values in (0, 1).
    """

    uniform_a = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
    uniform_b = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
    gumbel_a = -torch.log(-torch.log(uniform_a))
    gumbel_b = -torch.log(-torch.log(uniform_b))
    return torch.sigmoid((logits + gumbel_a - gumbel_b) / max(tau, 1e-6))


class GANetBaseline(nn.Module):
    """GA-Net adapted to the AC-GATE panel interface.

    Architecture:
    1. Project lagged sequential inputs into ``d_model`` and slice into per-
       time-step lag windows ``[B, T-K, K, d_model]``.
    2. Build a static context vector from the entity embedding plus a GRN-style
       projection of ``s_i``.
    3. Auxiliary gate network: a lightweight bidirectional GRU consumes the
       per-lag tokens conditioned on the static context, then a linear head
       produces a per-lag scalar gate logit. A Gumbel-sigmoid relaxation
       (training) or deterministic sigmoid (eval) yields soft gates in (0, 1).
    4. Gated multi-head self-attention over the K lag tokens. Gates act both
       as multiplicative weights on the value features and as an additive
       bias on the attention logits, so positions the aux network suppresses
       cannot dominate either path. This matches the spirit of the original
       GA-Net "select a subset of positions for attention" mechanism while
       remaining fully differentiable.
    5. Aggregate the gated, attended lag features into one per-time-step
       feature, feed through an LSTM encoder with static-derived initial
       state, apply the standard GateAddNorm, and project to ``y_pred``.
    """

    def __init__(
        self,
        cfg: CMDLConfig,
        num_attention_heads: int = 4,
        gumbel_tau: float = 1.0,
    ) -> None:
        super().__init__()
        if cfg.d_model % num_attention_heads != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by num_attention_heads "
                f"({num_attention_heads})"
            )
        if gumbel_tau <= 0.0:
            raise ValueError("gumbel_tau must be positive")

        self.cfg = cfg
        self.num_attention_heads = num_attention_heads
        self.gumbel_tau = gumbel_tau

        # Static / entity context.
        self.entity_embedding = nn.Embedding(cfg.n_entities, cfg.d_model)
        self.static_grn = GatedResidualNetwork(
            input_dim=cfg.static_dim,
            hidden_dim=cfg.d_model,
            output_dim=cfg.d_model,
            dropout=cfg.dropout,
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )

        # Sequential input projection.
        self.seq_input_projection = nn.Linear(cfg.seq_features, cfg.d_model)

        # Auxiliary gate network: bi-GRU over K lag tokens, conditioned on the
        # static context. Hidden size is d_model // 2 per direction so the
        # concatenated bidirectional output matches d_model exactly.
        gru_hidden = max(1, cfg.d_model // 2)
        self.aux_gru = nn.GRU(
            input_size=cfg.d_model,
            hidden_size=gru_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        # Project static context to GRU initial state (broadcast over both
        # directions).
        self.aux_h0_projection = nn.Linear(cfg.d_model, 2 * gru_hidden)
        # Per-lag gate head: takes [aux_gru_out || static_context] -> 1 logit.
        self.gate_head = nn.Sequential(
            nn.Linear(2 * gru_hidden + cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

        # Main multi-head self-attention over the K lag tokens.
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=num_attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.post_attention_gating = GateAddNorm(input_dim=cfg.d_model, dropout=cfg.dropout)

        # Per-lag feature transform (shared GRN, applied after attention).
        self.lag_feature_grn = GatedResidualNetwork(
            input_dim=cfg.d_model,
            hidden_dim=cfg.d_model,
            output_dim=cfg.d_model,
            dropout=cfg.dropout,
        )

        # LSTM encoder over time, with a static-derived initial state (kept
        # consistent with the TFT baseline so the comparison is fair).
        lstm_dropout = cfg.dropout if cfg.lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.hidden_initializer = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.Tanh(),
        )
        self.cell_initializer = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.Tanh(),
        )
        self.post_lstm_gating = GateAddNorm(input_dim=cfg.d_model, dropout=cfg.dropout)

        self.regression_head = nn.Linear(cfg.d_model, 1)

    def _build_lag_windows(self, projected_sequence: torch.Tensor) -> torch.Tensor:
        """Return tensor [B, T-K, K, d_model] of lagged feature tokens."""
        total_steps = projected_sequence.size(1)
        windows = [
            projected_sequence[:, self.cfg.max_lag - lag : total_steps - lag, :]
            for lag in range(1, self.cfg.max_lag + 1)
        ]
        return torch.stack(windows, dim=2)

    def _compute_gates(
        self,
        lag_windows: torch.Tensor,
        static_context: torch.Tensor,
    ) -> torch.Tensor:
        """Aux network gate computation. Returns soft gates [B, T-K, K]."""

        batch_size, t_eff, k, d_model = lag_windows.shape
        flat = lag_windows.reshape(batch_size * t_eff, k, d_model)

        # Static-conditioned bi-GRU initial state.
        gru_h0 = self.aux_h0_projection(static_context)  # [B, 2 * hidden]
        gru_h0 = gru_h0.unsqueeze(1).expand(-1, t_eff, -1).reshape(batch_size * t_eff, -1)
        gru_h0 = gru_h0.view(batch_size * t_eff, 2, -1).permute(1, 0, 2).contiguous()
        # gru_h0 -> [num_directions, B*T-K, hidden]

        gru_out, _ = self.aux_gru(flat, gru_h0)  # [B*T-K, K, 2 * hidden]

        # Concatenate static context for the gate head; broadcast across K.
        static_ctx = static_context.unsqueeze(1).expand(-1, t_eff, -1)
        static_ctx = static_ctx.reshape(batch_size * t_eff, 1, -1).expand(-1, k, -1)
        gate_input = torch.cat([gru_out, static_ctx], dim=-1)
        gate_logits = self.gate_head(gate_input).squeeze(-1)  # [B*T-K, K]

        if self.training:
            gates = _gumbel_sigmoid(gate_logits, tau=self.gumbel_tau)
        else:
            gates = torch.sigmoid(gate_logits)
        return gates.reshape(batch_size, t_eff, k)

    def forward(
        self,
        entity_ids: torch.Tensor,
        X_it: torch.Tensor,
        s_i: torch.Tensor,
        macro_controls: Optional[torch.Tensor] = None,
    ) -> GANetBaselineOutput:
        if entity_ids.dim() != 1:
            raise ValueError(
                f"Expected entity_ids with shape [B], got {tuple(entity_ids.shape)}"
            )
        if X_it.dim() != 3 or X_it.size(-1) != self.cfg.seq_features:
            raise ValueError(
                f"Expected X_it with shape [B, T, {self.cfg.seq_features}], "
                f"got {tuple(X_it.shape)}"
            )
        if s_i.dim() != 2 or s_i.size(-1) != self.cfg.static_dim:
            raise ValueError(
                f"Expected s_i with shape [B, {self.cfg.static_dim}], "
                f"got {tuple(s_i.shape)}"
            )
        batch_size = entity_ids.size(0)
        if X_it.size(0) != batch_size or s_i.size(0) != batch_size:
            raise ValueError("entity_ids, X_it, and s_i must agree on batch size")

        # Static context.
        entity_context = self.entity_embedding(entity_ids)
        static_context = self.static_grn(s_i)
        fused_context = self.context_fusion(
            torch.cat([entity_context, static_context], dim=-1)
        )

        # Lag windows: [B, T-K, K, d_model].
        projected = self.seq_input_projection(X_it)
        lag_windows = self._build_lag_windows(projected)
        b, t_eff, k, d = lag_windows.shape

        # Aux gate net produces soft gates over K lag positions.
        gates = self._compute_gates(lag_windows, fused_context)  # [B, T-K, K]

        # Gated multi-head self-attention over K lag tokens (per time step).
        flat_tokens = lag_windows.reshape(b * t_eff, k, d)
        gate_flat = gates.reshape(b * t_eff, k)

        # Multiplicative gating on the values entering attention.
        gated_tokens = flat_tokens * gate_flat.unsqueeze(-1)

        # Additive log-bias on attention logits using the gates so suppressed
        # positions also receive low attention probability. log(eps + g) is
        # the natural choice for a soft mask analog of "do not attend here".
        gate_log_bias = torch.log(gate_flat.clamp_min(1e-6))  # [B*T-K, K]
        # Broadcast bias across the query dimension so every query position
        # gets the same key-side suppression.
        attn_bias = gate_log_bias.unsqueeze(1).expand(-1, k, -1)  # [B*T-K, K, K]
        # nn.MultiheadAttention with num_heads must receive an attn_mask shaped
        # either [L, S] or [B*num_heads, L, S].
        attn_bias = (
            attn_bias.unsqueeze(1)
            .expand(-1, self.num_attention_heads, -1, -1)
            .reshape(b * t_eff * self.num_attention_heads, k, k)
        )
        attended, _ = self.attention(
            gated_tokens,
            gated_tokens,
            gated_tokens,
            attn_mask=attn_bias,
            need_weights=False,
        )
        attended = self.post_attention_gating(attended, residual=gated_tokens)
        # Per-lag GRN refinement.
        refined = self.lag_feature_grn(attended)  # [B*T-K, K, d_model]

        # Aggregate K lag tokens into one per-time-step feature using the gates
        # as soft selection weights normalized to a positive convex combination
        # (falling back to a uniform mean if all gates collapse to zero).
        weight_sum = gate_flat.sum(dim=-1, keepdim=True)
        weights = torch.where(
            weight_sum > 1e-6,
            gate_flat / weight_sum.clamp_min(1e-6),
            torch.full_like(gate_flat, 1.0 / k),
        )
        aggregated = (refined * weights.unsqueeze(-1)).sum(dim=1)  # [B*T-K, d_model]
        aggregated = aggregated.reshape(b, t_eff, d)

        # LSTM encoder over time with a static-derived initial state.
        h0 = (
            self.hidden_initializer(fused_context)
            .unsqueeze(0)
            .expand(self.cfg.lstm_layers, -1, -1)
            .contiguous()
        )
        c0 = (
            self.cell_initializer(fused_context)
            .unsqueeze(0)
            .expand(self.cfg.lstm_layers, -1, -1)
            .contiguous()
        )
        lstm_output, _ = self.lstm(aggregated, (h0, c0))
        sequence = self.post_lstm_gating(lstm_output, residual=aggregated)
        y_pred = self.regression_head(sequence).squeeze(-1)

        return GANetBaselineOutput(
            y_pred=y_pred,
            sequence=sequence,
            final_state=sequence[:, -1, :],
            aux_gate_weights=gates,
        )


__all__ = ["GANetBaseline", "GANetBaselineOutput"]
