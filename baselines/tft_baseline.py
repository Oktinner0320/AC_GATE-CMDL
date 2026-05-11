"""Lightweight TFT-style baseline used for AC-GATE comparison.

This baseline reproduces the structurally interpretable elements of the Temporal
Fusion Transformer (TFT) on the same panel interface as :class:`PlainLSTMBaseline`:

- Per-lag Gated Residual Network (GRN) feature extractor
- Static-context-conditioned Variable Selection over lags (softmax weights)
- Two-layer LSTM encoder
- Causal multi-head self-attention with gated residual
- Position-wise GRN + final regression head

It intentionally avoids the full pytorch-forecasting TFT (with quantile heads,
covariate splits, etc.) so that we can keep the same training loop, optimiser,
warm-up handling, and post-hoc lag occlusion tooling already used by the LSTM
baseline. The forward signature matches :class:`PlainLSTMBaseline` so the
existing per-domain runners only need to swap the model class.
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
class TFTBaselineOutput:
    """Structured outputs returned by the lightweight TFT baseline."""

    y_pred: torch.Tensor
    sequence: torch.Tensor
    final_state: torch.Tensor
    variable_selection_weights: torch.Tensor


class _StaticGRN(nn.Module):
    """Single-tensor GRN wrapper for static context (no time dimension)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.grn = GatedResidualNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GRN expects [B, T, D]; promote and squeeze.
        return self.grn(x.unsqueeze(1)).squeeze(1)


class TFTBaseline(nn.Module):
    """Lightweight TFT-style panel forecaster.

    Mirrors the AC-GATE input contract: lagged sequential inputs are the only
    temporal signal (no shortcut on contemporaneous X_t). Static features and a
    learned entity embedding form the static context vector; this static
    context conditions a per-lag variable selection that aggregates lagged
    features, after which an LSTM encoder + causal self-attention produces the
    one-step-ahead prediction sequence aligned with warm-up trimming.
    """

    def __init__(self, cfg: CMDLConfig, num_attention_heads: int = 4) -> None:
        super().__init__()
        if cfg.d_model % num_attention_heads != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        self.cfg = cfg
        self.num_attention_heads = num_attention_heads

        # Static / entity context.
        self.entity_embedding = nn.Embedding(cfg.n_entities, cfg.d_model)
        self.static_grn = _StaticGRN(
            input_dim=cfg.static_dim,
            hidden_dim=cfg.d_model,
            output_dim=cfg.d_model,
            dropout=cfg.dropout,
        )
        self.context_fusion = _StaticGRN(
            input_dim=2 * cfg.d_model,
            hidden_dim=cfg.d_model,
            output_dim=cfg.d_model,
            dropout=cfg.dropout,
        )

        # Per-lag feature transformation (shared GRN applied to each lag slice).
        self.seq_input_projection = nn.Linear(cfg.seq_features, cfg.d_model)
        self.lag_feature_grn = GatedResidualNetwork(
            input_dim=cfg.d_model,
            hidden_dim=cfg.d_model,
            output_dim=cfg.d_model,
            dropout=cfg.dropout,
        )

        # Variable Selection over lags: produces softmax weights over K lags
        # conditioned on static context.
        self.vsn_score_grn = GatedResidualNetwork(
            input_dim=cfg.max_lag * cfg.d_model,
            hidden_dim=cfg.d_model,
            output_dim=cfg.max_lag,
            dropout=cfg.dropout,
            context_dim=cfg.d_model,
        )

        # LSTM encoder over the selected per-step feature.
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

        # Causal multi-head self-attention.
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=num_attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.post_attention_gating = GateAddNorm(input_dim=cfg.d_model, dropout=cfg.dropout)

        # Position-wise GRN and output gating.
        self.position_wise_grn = GatedResidualNetwork(
            input_dim=cfg.d_model,
            hidden_dim=cfg.d_model,
            output_dim=cfg.d_model,
            dropout=cfg.dropout,
        )
        self.output_gating = GateAddNorm(input_dim=cfg.d_model, dropout=cfg.dropout)
        self.regression_head = nn.Linear(cfg.d_model, 1)

    def _build_lag_windows(self, projected_sequence: torch.Tensor) -> torch.Tensor:
        """Return tensor [B, T-K, K, d_model] of lagged features."""
        total_steps = projected_sequence.size(1)
        windows = [
            projected_sequence[:, self.cfg.max_lag - lag : total_steps - lag, :]
            for lag in range(1, self.cfg.max_lag + 1)
        ]
        return torch.stack(windows, dim=2)

    def forward(
        self,
        entity_ids: torch.Tensor,
        X_it: torch.Tensor,
        s_i: torch.Tensor,
        macro_controls: Optional[torch.Tensor] = None,
    ) -> TFTBaselineOutput:
        if entity_ids.dim() != 1:
            raise ValueError(f"Expected entity_ids with shape [B], got {tuple(entity_ids.shape)}")
        if X_it.dim() != 3 or X_it.size(-1) != self.cfg.seq_features:
            raise ValueError(
                f"Expected X_it with shape [B, T, {self.cfg.seq_features}], got {tuple(X_it.shape)}"
            )
        if s_i.dim() != 2 or s_i.size(-1) != self.cfg.static_dim:
            raise ValueError(
                f"Expected s_i with shape [B, {self.cfg.static_dim}], got {tuple(s_i.shape)}"
            )
        batch_size = entity_ids.size(0)
        if X_it.size(0) != batch_size or s_i.size(0) != batch_size:
            raise ValueError("entity_ids, X_it, and s_i must agree on batch size")

        # Static context.
        entity_context = self.entity_embedding(entity_ids)
        static_context = self.static_grn(s_i)
        fused_context = self.context_fusion(torch.cat([entity_context, static_context], dim=-1))

        # Lag windows -> per-lag GRN features.
        projected = self.seq_input_projection(X_it)
        lag_windows = self._build_lag_windows(projected)  # [B, T-K, K, d_model]
        b, t_eff, k, d = lag_windows.shape
        lag_features = self.lag_feature_grn(lag_windows.reshape(b * t_eff, k, d)).reshape(b, t_eff, k, d)

        # Variable Selection over lags conditioned on static context.
        vsn_input = lag_windows.reshape(b, t_eff, k * d)
        repeated_context = fused_context.unsqueeze(1).expand(-1, t_eff, -1)
        vsn_logits = self.vsn_score_grn(vsn_input, context=repeated_context)
        vsn_weights = torch.softmax(vsn_logits, dim=-1)  # [B, T-K, K]
        selected = (vsn_weights.unsqueeze(-1) * lag_features).sum(dim=2)  # [B, T-K, d_model]

        # LSTM encoder with static-derived initial state.
        h0 = self.hidden_initializer(fused_context).unsqueeze(0).expand(self.cfg.lstm_layers, -1, -1).contiguous()
        c0 = self.cell_initializer(fused_context).unsqueeze(0).expand(self.cfg.lstm_layers, -1, -1).contiguous()
        lstm_output, _ = self.lstm(selected, (h0, c0))
        gated = self.post_lstm_gating(lstm_output, residual=selected)

        # Causal multi-head self-attention.
        causal_mask = torch.triu(
            torch.ones(t_eff, t_eff, device=gated.device, dtype=torch.bool),
            diagonal=1,
        )
        attended, _ = self.attention(gated, gated, gated, attn_mask=causal_mask, need_weights=False)
        attended = self.post_attention_gating(attended, residual=gated)

        # Position-wise GRN + output gate.
        pw = self.position_wise_grn(attended)
        sequence = self.output_gating(pw, residual=attended)
        y_pred = self.regression_head(sequence).squeeze(-1)

        return TFTBaselineOutput(
            y_pred=y_pred,
            sequence=sequence,
            final_state=sequence[:, -1, :],
            variable_selection_weights=vsn_weights,
        )


__all__ = ["TFTBaseline", "TFTBaselineOutput"]
