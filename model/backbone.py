"""Step 3 backbone building blocks.

本文件承载时序主干、共享 GLU 和 post-LSTM 门控残差块。
This file contains the sequential backbone, the shared GLU, and the post-LSTM gated residual block.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class TimeDistributed(nn.Module):
    """
    This module can wrap any given module and stacks the time dimension with the batch dimension of the inputs
    before applying the module.
    Borrowed from this fruitful `discussion thread
    <https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4>`_.

    Parameters
    ----------
    module : nn.Module
        The wrapped module.
    batch_first: bool
        A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    return_reshaped: bool
        A boolean indicating whether to return the output in the corresponding original shape or not.
    """

    def __init__(self, module: nn.Module, batch_first: bool = True, return_reshaped: bool = True):
        super(TimeDistributed, self).__init__()
        self.module: nn.Module = module  # the wrapped module
        self.batch_first: bool = batch_first  # indicates the dimensions order of the sequential data.
        self.return_reshaped: bool = return_reshaped

    def forward(self, x):

        # in case the incoming tensor is a two-dimensional tensor - infer no temporal information is involved,
        # and simply apply the module
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and time-steps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * time-steps, input_size)
        # apply the module on each time-step separately
        y = self.module(x_reshape)

        # reshaping the module output as sequential tensor (if required)
        if self.return_reshaped:
            if self.batch_first:
                y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, time-steps, output_size)
            else:
                y = y.view(-1, x.size(1), y.size(-1))  # (time-steps, samples, output_size)

        return y


class GatedLinearUnit(nn.Module):
    """保持输入输出维度一致的共享 GLU。
    Dimension-preserving GLU shared by the lag gate and backbone blocks.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(input_dim, input_dim)
        self.value_projection = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_projection(x))
        value = self.value_projection(x)
        return gate * value


class GateAddNorm(nn.Module):
    """TFT 风格的 Dropout -> GLU -> Residual -> LayerNorm 组合块。
    Dropout -> GLU -> residual add -> LayerNorm block from TFT.
    """

    def __init__(self, input_dim: int, dropout: Optional[float] = None) -> None:
        super().__init__()
        self.dropout_rate = dropout
        if dropout:
            self.dropout_layer = nn.Dropout(dropout)
        self.gate = TimeDistributed(GatedLinearUnit(input_dim), batch_first=True)
        self.layernorm = TimeDistributed(nn.LayerNorm(input_dim), batch_first=True)

    def forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.dropout_rate:
            x = self.dropout_layer(x)
        x = self.gate(x)
        if residual is not None:
            x = x + residual
        return self.layernorm(x)


@dataclass(slots=True)
class BackboneOutput:
    """Step 3 backbone 的结构化输出。
    Structured outputs returned by the Step 3 backbone.
    """

    sequence: torch.Tensor
    final_state: torch.Tensor


class UniversalPanelBackbone(nn.Module):
    """融合滞后上下文、实体效应和静态信息的时序主干。
    Sequence backbone that fuses lag context, entity effects, and static signals.
    """

    def __init__(self, d_model: int, static_dim: int, lstm_layers: int = 2, dropout: float = 0.05) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if static_dim < 1:
            raise ValueError("static_dim must be positive")
        if lstm_layers < 1:
            raise ValueError("lstm_layers must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.d_model = d_model
        self.static_dim = static_dim
        self.lstm_layers = lstm_layers

        # 先把静态信号、标量 z_i 和可选 macro 控制映射到统一宽度，便于后续拼接。
        # Static signals, scalar z_i, and optional macro controls are first mapped into the same width.
        self.static_projection = nn.Linear(static_dim, d_model)
        self.z_projection = nn.Linear(1, d_model)
        self.control_projection = nn.Linear(1, d_model)
        # 输入融合层把 4 路 d_model 特征压回单一路径，避免直接把大拼接喂给 LSTM。
        # The fusion layer compresses four d_model streams back to one before the LSTM.
        # NOTE: current_sequence is excluded to prevent the backbone from bypassing lag_context.
        self.input_fusion = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # hidden/cell 初始化显式注入实体、静态和 z_i 信息，而不是使用全零初态。
        # Hidden/cell initializers inject entity, static, and z_i information instead of zeros.
        self.hidden_initializer = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.Tanh())
        self.cell_initializer = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.Tanh())
        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.post_lstm_gating = GateAddNorm(input_dim=d_model, dropout=dropout)

    def forward(
        self,
        current_sequence: torch.Tensor,
        lag_context_sequence: torch.Tensor,
        entity_embedding: torch.Tensor,
        static_features: torch.Tensor,
        z_i: torch.Tensor,
        macro_controls: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        if current_sequence.dim() != 3:
            raise ValueError(
                f"Expected current_sequence with shape [B, T, D], got {tuple(current_sequence.shape)}"
            )
        if lag_context_sequence.shape != current_sequence.shape:
            raise ValueError(
                "lag_context_sequence must match current_sequence shape; "
                f"got {tuple(lag_context_sequence.shape)} vs {tuple(current_sequence.shape)}"
            )
        if entity_embedding.dim() != 2 or entity_embedding.size(-1) != self.d_model:
            raise ValueError(
                f"Expected entity_embedding with shape [B, {self.d_model}], got {tuple(entity_embedding.shape)}"
            )
        if static_features.dim() != 2 or static_features.size(-1) != self.static_dim:
            raise ValueError(
                f"Expected static_features with shape [B, {self.static_dim}], got {tuple(static_features.shape)}"
            )
        if z_i.dim() != 2 or z_i.size(-1) != 1:
            raise ValueError(f"Expected z_i with shape [B, 1], got {tuple(z_i.shape)}")

        batch_size, num_steps, _ = current_sequence.shape
        # 把实体级信号扩展到时间维，形成每个有效时间步可消费的上下文张量。
        # Expand entity-level signals across time so every valid step can consume the same context.
        static_context = self.static_projection(static_features).unsqueeze(1).expand(-1, num_steps, -1)
        entity_context = entity_embedding.unsqueeze(1).expand(-1, num_steps, -1)
        if macro_controls is None:
            # 当前 synthetic smoke test 没有宏观控制变量，因此这里退化为零张量占位。
            # The current synthetic smoke test has no macro controls, so use a zero placeholder here.
            macro_context = current_sequence.new_zeros(batch_size, num_steps, self.d_model)
        else:
            if macro_controls.dim() != 3 or macro_controls.shape[:2] != current_sequence.shape[:2]:
                raise ValueError(
                    "macro_controls must have shape [B, T, C] with the same first two dimensions as current_sequence"
                )
            # 暂时先把多维控制变量做均值摘要，再映射到 d_model；后续可替换为更细的 encoder。
            # For now, summarize multi-dimensional controls by their mean before projection to d_model.
            macro_summary = macro_controls.mean(dim=-1, keepdim=True)
            macro_context = self.control_projection(macro_summary)

        # 将滞后上下文和实体侧信息统一压缩为 LSTM 的单路输入（不含 current_sequence 以消除捷径路径）。
        # Compress lag context and entity-side information into a single LSTM input stream (current_sequence excluded to remove shortcut).
        fused_inputs = torch.cat(
            [lag_context_sequence, entity_context, static_context, macro_context],
            dim=-1,
        )
        fused_inputs = self.input_fusion(fused_inputs)

        # 用 entity/static/z_i 的组合初始化循环状态，让 LSTM 从第一个有效步就带有实体条件信息。
        # Initialize recurrent states from entity/static/z_i so the LSTM is conditioned from the first valid step.
        z_context = self.z_projection(z_i)
        initialization_signal = torch.cat(
            [entity_embedding, static_context[:, 0, :], z_context],
            dim=-1,
        )
        h0 = self.hidden_initializer(initialization_signal).unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()
        c0 = self.cell_initializer(initialization_signal).unsqueeze(0).expand(self.lstm_layers, -1, -1).contiguous()

        # LSTM 负责局部时序建模，GateAddNorm 负责稳定输出并保留直接残差路径。
        # The LSTM models local temporal structure, while GateAddNorm stabilizes outputs with a residual path.
        lstm_output, _ = self.lstm(fused_inputs, (h0, c0))
        sequence = self.post_lstm_gating(lstm_output, residual=fused_inputs)
        return BackboneOutput(sequence=sequence, final_state=sequence[:, -1, :])


__all__ = [
    "BackboneOutput",
    "GateAddNorm",
    "GatedLinearUnit",
    "TimeDistributed",
    "UniversalPanelBackbone",
]
