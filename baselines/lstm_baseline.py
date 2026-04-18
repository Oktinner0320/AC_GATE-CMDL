"""Plain LSTM baseline used for synthetic Step 4.5 comparisons.

该基线保留实体嵌入和静态特征条件化，但移除 AC encoder、lag gate 和 proxy 重构，
用于隔离 AC-GATE 带来的增益，而不是把全部侧信息一并删掉。
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from config.cmdl_config import CMDLConfig
from model.backbone import GateAddNorm


@dataclass(slots=True)
class PlainLSTMBaselineOutput:
    """Structured outputs returned by the plain LSTM baseline.

    基线只输出预测序列和 backbone 隐状态，不提供原生 z、omega 或 k*。
    """

    y_pred: torch.Tensor
    sequence: torch.Tensor
    final_state: torch.Tensor


class PlainLSTMBaseline(nn.Module):
    """Matched LSTM baseline without AC-gated lag selection.

    该模型保持与 CMDL 尽量一致的训练接口与侧信息注入方式，
    但使用标准 LSTM 直接建模有效时间段上的预测任务。
    """

    def __init__(self, cfg: CMDLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_adapter = nn.Sequential(
            nn.Linear(cfg.seq_features, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )
        self.entity_embedding = nn.Embedding(cfg.n_entities, cfg.d_model)
        self.static_projection = nn.Linear(cfg.static_dim, cfg.d_model)
        self.control_projection = nn.Linear(1, cfg.d_model)
        self.input_fusion = nn.Sequential(
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )
        self.hidden_initializer = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.d_model), nn.Tanh())
        self.cell_initializer = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.d_model), nn.Tanh())
        lstm_dropout = cfg.dropout if cfg.lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cfg.d_model,
            hidden_size=cfg.d_model,
            num_layers=cfg.lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.post_lstm_gating = GateAddNorm(input_dim=cfg.d_model, dropout=cfg.dropout)
        self.regression_head = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        entity_ids: torch.Tensor,
        X_it: torch.Tensor,
        s_i: torch.Tensor,
        macro_controls: Optional[torch.Tensor] = None,
    ) -> PlainLSTMBaselineOutput:
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
        if macro_controls is not None and macro_controls.shape[:2] != X_it.shape[:2]:
            raise ValueError("macro_controls must align with X_it on the batch and time dimensions")

        adapted_sequence = self.input_adapter(X_it)
        current_sequence = adapted_sequence[:, self.cfg.max_lag :, :]
        valid_macro_controls = None if macro_controls is None else macro_controls[:, self.cfg.max_lag :, :]
        num_steps = current_sequence.size(1)

        entity_context = self.entity_embedding(entity_ids)
        repeated_entity = entity_context.unsqueeze(1).expand(-1, num_steps, -1)
        static_context = self.static_projection(s_i).unsqueeze(1).expand(-1, num_steps, -1)
        if valid_macro_controls is None:
            macro_context = current_sequence.new_zeros(batch_size, num_steps, self.cfg.d_model)
        else:
            if valid_macro_controls.dim() != 3:
                raise ValueError(
                    f"Expected macro_controls with shape [B, T, C], got {tuple(valid_macro_controls.shape)}"
                )
            macro_summary = valid_macro_controls.mean(dim=-1, keepdim=True)
            macro_context = self.control_projection(macro_summary)

        fused_inputs = torch.cat(
            [current_sequence, repeated_entity, static_context, macro_context],
            dim=-1,
        )
        fused_inputs = self.input_fusion(fused_inputs)

        initialization_signal = torch.cat([entity_context, static_context[:, 0, :]], dim=-1)
        h0 = self.hidden_initializer(initialization_signal).unsqueeze(0)
        h0 = h0.expand(self.cfg.lstm_layers, -1, -1).contiguous()
        c0 = self.cell_initializer(initialization_signal).unsqueeze(0)
        c0 = c0.expand(self.cfg.lstm_layers, -1, -1).contiguous()

        lstm_output, _ = self.lstm(fused_inputs, (h0, c0))
        sequence = self.post_lstm_gating(lstm_output, residual=fused_inputs)
        y_pred = self.regression_head(sequence).squeeze(-1)
        return PlainLSTMBaselineOutput(
            y_pred=y_pred,
            sequence=sequence,
            final_state=sequence[:, -1, :],
        )


__all__ = ["PlainLSTMBaseline", "PlainLSTMBaselineOutput"]