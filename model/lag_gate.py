"""CMDL 的滞后门控模块集合。
Lag-gating modules used by the CMDL core model.

本文件同时保留两类组件：
1. 面向 Step 3 时序主干复用的 time-distributed GRN；
2. 面向 Step 2 标量 z_i 条件化的 ScaleInvariantLagGate。
This file keeps two families of components:
1. a time-distributed GRN for later sequential reuse in Step 3;
2. the scalar-conditioned ScaleInvariantLagGate used directly in Step 2.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

# 复用 backbone 中的共享 GLU，避免 Step 2 / Step 3 的门控实现分叉。
# Reuse the shared GLU from the backbone to keep Step 2 / Step 3 gating behavior consistent.
from model.backbone import GatedLinearUnit, TimeDistributed



@dataclass(slots=True)
class LagGateOutput:

    omega: torch.Tensor                     # 滞后概率分布 / Lag distribution，形状 [B, K] / shape [B, K]，长度为 max_lag / length max_lag
    k_star: torch.Tensor                     # 期望滞后位置 / Expected lag position，形状 [B] / shape [B]，标量值范围在 (0, max_lag] / scalar value in (0, max_lag]
    lag_context: Optional[torch.Tensor]     # 加权后的历史上下文 / Weighted historical context，形状 [B, d_model] / shape [B, d_model]，当提供 lagged_sequence 时非 None / non-None when lagged_sequence is provided
    logits: torch.Tensor                     # 原始 logits / Raw logits，形状 [B, K] / shape [B, K]，未经过 softmax 的滞后打分 / unnormalized lag scores before softmax


class GatedResidualNetwork(nn.Module):
    """面向时序张量的 TFT 风格 GRN。
    Time-distributed GRN kept for later sequential blocks in Step 3.

    这个版本保留 TimeDistributed 包装，适合 [B, T, D] 形式的序列输入。
    It keeps the TimeDistributed wrapping and is intended for sequence inputs shaped like [B, T, D].
    """

    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int,
                dropout:Optional[float] = 0.05,
                context_dim:Optional[int] = None,
                batch_first:Optional[bool] = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # 如果输入维度和输出维度不同，先为残差分支补一个线性投影。
        # When input and output widths differ, project the residual branch first.
        self.project_residual: bool = self.input_dim != self.output_dim
        if self.project_residual:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_dim, self.output_dim), batch_first=batch_first)

        # 主路径先投影到 hidden_dim，再过非线性和第二层投影。
        # The main path first projects to hidden_dim, then goes through nonlinearity and a second projection.
        self.fc1 = TimeDistributed(nn.Linear(self.input_dim, self.hidden_dim), batch_first=batch_first)

        # context 分支仅在显式提供辅助上下文时启用。
        # The context branch is only enabled when an auxiliary context signal is provided.
        if self.context_dim is not None:
            self.context_projection = TimeDistributed(nn.Linear(self.context_dim, self.hidden_dim, bias=False), batch_first=batch_first)
        self.activation = nn.ELU()
        self.fc2 = TimeDistributed(nn.Linear(self.hidden_dim, self.output_dim), batch_first=batch_first)

        # 投影后的输出先做 dropout，再经过 GLU、残差和 LayerNorm。
        # The projected output then goes through dropout, GLU, residual addition, and LayerNorm.
        self.dropout = nn.Dropout(self.dropout_rate)
        self.gate = TimeDistributed(GatedLinearUnit(self.output_dim), batch_first=batch_first)
        self.layernorm = TimeDistributed(nn.LayerNorm(self.output_dim), batch_first=batch_first)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """执行时序版 GRN 前向。
        Run the time-distributed GRN forward pass.

        逻辑顺序是 residual 准备 -> 主分支投影 -> 可选 context 融合 -> 非线性 ->
        第二次投影 -> dropout -> GLU -> residual add -> LayerNorm。
        The order is residual preparation -> main projection -> optional context fusion -> nonlinearity ->
        second projection -> dropout -> GLU -> residual add -> LayerNorm.
        """
        # residual 分支无论如何都要准备好；如果需要投影，先过 skip_layer。
        if self.project_residual:
            residual = self.skip_layer(x)
        else:
            residual = x
        # 主分支先投影到 hidden_dim。
        x = self.fc1(x)
        if context is not None:
            context = self.context_projection(context)
            x = x + context
        # 过非线性和第二层投影。
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x + residual
        x = self.layernorm(x)
        return x


class ScalarGatedResidualNetwork(nn.Module):
    """面向非时序输入的简化 GRN。
    Plain GRN variant for non-sequential conditioning on scalar z_i.

    这个版本去掉了 TimeDistributed，只保留 Step 2 所需的标量条件化骨架。
    This variant removes TimeDistributed and keeps only the scalar-conditioning skeleton needed in Step 2.
    """

    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, dropout:float = 0.05) -> None:
        super().__init__()
        # 这里的 residual 逻辑与时序版一致，只是张量形状退化成 [B, D]。
        # The residual logic matches the sequential GRN, but the tensor shape is reduced to [B, D].
        self.project_residual = input_dim != output_dim
        self.skip_layer = nn.Linear(input_dim, output_dim) if self.project_residual else nn.Identity()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = GatedLinearUnit(output_dim)
        self.layernorm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行标量条件版 GRN 前向。
        Run the scalar-conditioned GRN forward pass.
        """

        residual = self.skip_layer(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x + residual
        return self.layernorm(x)


class ScaleInvariantLagGate(nn.Module):
    """根据 z_i 预测滞后分布并聚合历史上下文。
    Predict a lag distribution conditioned on z_i and aggregate lagged context.

    它的核心职责是把标量 z_i 转成 K 维 lag logits，再通过相对位置偏置和温度 softmax
    得到 omega，最后计算可解释的 k_star，并可选地把 omega 作用到历史窗口上得到 lag_context。
    Its core job is to transform scalar z_i into K-dimensional lag logits, apply relative-position bias
    and temperature-scaled softmax to obtain omega, compute interpretable k_star, and optionally aggregate
    a lagged window into lag_context.
    """

    def __init__(self, max_lag:int, d_model:int,
        temperature:float = 1.0,
        hidden_dim:int = 16,
        dropout:float = 0.05,
        lag_bias_strength:float = 1.0) -> None:
        super().__init__()
        if max_lag < 1:
            raise ValueError("max_lag must be at least 1")
        if d_model < 1:
            raise ValueError("d_model must be at least 1")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if lag_bias_strength < 0.0:
            raise ValueError("lag_bias_strength must be non-negative")

        self.max_lag = max_lag
        self.d_model = d_model
        self.temperature = temperature
        self.lag_bias_strength = lag_bias_strength
        # logit_network 负责从标量 z_i 产生长度为 K 的原始打分。
        # The logit network produces K raw lag scores from scalar z_i.
        self.logit_network = ScalarGatedResidualNetwork(input_dim=1, hidden_dim=hidden_dim, output_dim=max_lag, dropout=dropout)

        # relative_positions 给更远的 lag 提供固定位置成本；lag_indices 用于计算期望滞后 k_star。
        # relative_positions adds a fixed positional cost to farther lags, and lag_indices are used for k_star.
        relative_positions = torch.arange(1, max_lag + 1, dtype=torch.float32) / float(max_lag)
        lag_indices = torch.arange(1, max_lag + 1, dtype=torch.float32)
        self.register_buffer("relative_positions", relative_positions, persistent=False)
        self.register_buffer("lag_indices", lag_indices, persistent=False)

    def forward(self, z_i:torch.Tensor, lagged_sequence:Optional[torch.Tensor]=None) -> LagGateOutput:
        """执行 Lag Gate 前向并返回分布、解释量和可选上下文。
        Run the lag-gate forward pass and return the distribution, interpretability signal, and optional context.

        如果提供 lagged_sequence，要求其形状为 [B, K, d_model]，这样 omega 才能直接沿 lag 维做加权求和。
        If lagged_sequence is provided, it must have shape [B, K, d_model] so omega can weight the lag axis directly.
        """

        if z_i.dim() != 2 or z_i.size(-1) != 1:
            raise ValueError(f"Expected z_i with shape [B, 1], got {tuple(z_i.shape)}")

        # 第一步：由 z_i 生成原始 logits，并对较远的 lag 加入固定惩罚。
        # Step 1: generate raw logits from z_i and add a fixed penalty to farther lag positions.
        logits = self.logit_network(z_i)
        logits = logits - self.lag_bias_strength * self.relative_positions.unsqueeze(0)

        # 第二步：经温度缩放 softmax 得到合法概率分布 omega。
        # Step 2: apply temperature-scaled softmax to obtain a valid probability distribution omega.
        omega = F.softmax(logits / self.temperature, dim=-1)

        # 第三步：把离散分布映射成期望滞后位置，作为可解释输出 k_star。
        # Step 3: map the discrete distribution into an expected lag position as interpretable k_star.
        k_star = torch.sum(omega * self.lag_indices.unsqueeze(0), dim=-1)

        lag_context = None
        if lagged_sequence is not None:
            if lagged_sequence.dim() != 3:
                raise ValueError(
                    f"Expected lagged_sequence with shape [B, K, d_model], got {tuple(lagged_sequence.shape)}"
                )
            if lagged_sequence.size(1) != self.max_lag:
                raise ValueError(
                    f"Expected lagged_sequence to have {self.max_lag} lag positions, got {lagged_sequence.size(1)}"
                )
            if lagged_sequence.size(-1) != self.d_model:
                raise ValueError(
                    f"Expected lagged_sequence feature size {self.d_model}, got {lagged_sequence.size(-1)}"
                )

            # 第四步：沿 lag 维对历史窗口做加权求和，得到后续 backbone 可直接消费的上下文。
            # Step 4: weight and sum the lag window across the lag axis to obtain a context vector for the backbone.
            lag_context = torch.einsum("bk,bkd->bd", omega, lagged_sequence)

        return LagGateOutput(omega=omega, k_star=k_star, lag_context=lag_context, logits=logits)


__all__ = [
    "LagGateOutput",
    "GatedLinearUnit",
    "GatedResidualNetwork",
    "ScalarGatedResidualNetwork",
    "ScaleInvariantLagGate",
]