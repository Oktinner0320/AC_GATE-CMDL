"""CMDL Step 3 端到端模型组装模块。

This module assembles the AC encoder, lag gate, LSTM backbone, and regression
head into a sequence-to-sequence model over the valid time range t >= max_lag.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from config.cmdl_config import CMDLConfig
from model.ac_encoder import AdaptiveACEncoder
from model.backbone import UniversalPanelBackbone
from model.lag_gate import ScaleInvariantLagGate


@dataclass(slots=True)
class CMDLModelOutput:
	"""Step 3 模型的结构化输出。
	Structured outputs returned by the Step 3 CMDL model.
	"""

	y_pred: torch.Tensor
	omega: torch.Tensor
	z_i: torch.Tensor
	p_hat_i: torch.Tensor
	k_star: torch.Tensor
	lag_context_sequence: torch.Tensor
	backbone_sequence: torch.Tensor


class CMDLModel(nn.Module):
	"""用于 Step 3 合成数据 smoke test 的最小端到端 CMDL 模型。
	Minimal end-to-end CMDL model for the Step 3 synthetic smoke test.
	"""

	def __init__(self, cfg: CMDLConfig) -> None:
		super().__init__()
		self.cfg = cfg
		# 模块按训练时的数据流顺序声明，便于后续实验脚本直接复用。
		# Modules are declared in the same order as the training data flow for easier reuse.
		self.ac_encoder = AdaptiveACEncoder(
			n_proxies=cfg.n_proxies,
			detach_reconstruction=getattr(cfg, "reconstruction_detach", True),
		)
		# 输入适配层先把原始时序特征投影到统一的 d_model 空间。
		# The input adapter projects raw sequential features into the shared d_model space first.
		self.input_adapter = nn.Sequential(
			nn.Linear(cfg.seq_features, cfg.d_model),
			nn.LayerNorm(cfg.d_model),
			nn.GELU(),
		)
		# 实体嵌入显式承载横截面固定效应，与静态特征投影并行进入 backbone。
		# Entity embeddings carry cross-sectional fixed effects alongside projected static features.
		self.entity_embedding = nn.Embedding(cfg.n_entities, cfg.d_model)
		# Lag gate 只先学习实体级滞后分布；真正的逐时点聚合在 forward 中完成。
		# The lag gate only learns an entity-level lag distribution; per-step aggregation happens in forward.
		self.lag_gate = ScaleInvariantLagGate(
			max_lag=cfg.max_lag,
			d_model=cfg.d_model,
			temperature=cfg.temperature,
			dropout=cfg.dropout,
			lag_bias_strength=cfg.lag_bias_strength,
			omega_transform=getattr(cfg, "omega_transform", "softmax"),
		)
		self.backbone = UniversalPanelBackbone(
			d_model=cfg.d_model,
			static_dim=cfg.static_dim,
			lstm_layers=cfg.lstm_layers,
			dropout=cfg.dropout,
		)
		self.regression_head = nn.Linear(cfg.d_model, 1)

	def _build_lagged_windows(self, sequence: torch.Tensor) -> torch.Tensor:
		"""构造按 [lag1, lag2, ..., lagK] 排列的 rolling 历史窗口。
		Create rolling lag windows ordered as [lag1, lag2, ..., lagK].
		"""

		if sequence.dim() != 3:
			raise ValueError(f"Expected sequence with shape [B, T, D], got {tuple(sequence.shape)}")
		if sequence.size(1) <= self.cfg.max_lag:
			raise ValueError("Sequence length must be greater than max_lag to build lagged windows")

		windows = [
			sequence[:, time_index - self.cfg.max_lag : time_index, :]
			for time_index in range(self.cfg.max_lag, sequence.size(1))
		]
		lagged_windows = torch.stack(windows, dim=1)
		# slice 得到的窗口是从“最远 lag -> 最近 lag”，这里翻转后与 omega 的 [1..K] 语义对齐。
		# Slices are collected from farthest to nearest lag; flip aligns them with omega's [1..K] semantics.
		return torch.flip(lagged_windows, dims=[2])

	def forward(
		self,
		entity_ids: torch.Tensor,
		X_it: torch.Tensor,
		p_i: torch.Tensor,
		s_i: torch.Tensor,
		macro_controls: Optional[torch.Tensor] = None,
	) -> CMDLModelOutput:
		if entity_ids.dim() != 1:
			raise ValueError(f"Expected entity_ids with shape [B], got {tuple(entity_ids.shape)}")
		if X_it.dim() != 3 or X_it.size(-1) != self.cfg.seq_features:
			raise ValueError(
				f"Expected X_it with shape [B, T, {self.cfg.seq_features}], got {tuple(X_it.shape)}"
			)
		if p_i.dim() != 2 or p_i.size(-1) != self.cfg.n_proxies:
			raise ValueError(
				f"Expected p_i with shape [B, {self.cfg.n_proxies}], got {tuple(p_i.shape)}"
			)
		if s_i.dim() != 2 or s_i.size(-1) != self.cfg.static_dim:
			raise ValueError(
				f"Expected s_i with shape [B, {self.cfg.static_dim}], got {tuple(s_i.shape)}"
			)

		batch_size = entity_ids.size(0)
		if X_it.size(0) != batch_size or p_i.size(0) != batch_size or s_i.size(0) != batch_size:
			raise ValueError("entity_ids, X_it, p_i, and s_i must agree on batch size")
		if macro_controls is not None and macro_controls.shape[:2] != X_it.shape[:2]:
			raise ValueError("macro_controls must align with X_it on the batch and time dimensions")

		# 先从 proxy 提炼实体级 z_i，再把原始时间序列映射到 backbone 所需宽度。
		# First compress proxies into entity-level z_i, then project the raw sequence to the backbone width.
		encoder_output = self.ac_encoder(p_i)
		adapted_sequence = self.input_adapter(X_it)
		# 这里仅请求 omega / k*，不在 LagGate 内部做时序聚合，避免重复展开窗口。
		# Request only omega / k* here and keep temporal aggregation outside the lag gate to avoid repeated window expansion.
		lag_gate_output = self.lag_gate(encoder_output.z_i)

		# 用共享的实体级 omega 对每个有效时间步的 K 阶历史窗口做加权求和。
		# Apply the shared entity-level omega to every valid time-step's K-lag window.
		lagged_windows = self._build_lagged_windows(adapted_sequence)
		lag_context_sequence = torch.einsum("bk,btkd->btd", lag_gate_output.omega, lagged_windows)
		# 前 max_lag 个时间步没有完整历史窗口，因此直接作为 warm-up 丢弃。
		# The first max_lag steps do not have a complete history window, so they are dropped as warm-up.
		current_sequence = adapted_sequence[:, self.cfg.max_lag :, :]
		valid_macro_controls = None if macro_controls is None else macro_controls[:, self.cfg.max_lag :, :]
		entity_context = self.entity_embedding(entity_ids)

		# Backbone 只消费有效时间段，并输出逐时点隐藏状态供回归头使用。
		# The backbone only consumes the valid time range and returns per-step hidden states for the regression head.
		backbone_output = self.backbone(
			current_sequence=current_sequence,
			lag_context_sequence=lag_context_sequence,
			entity_embedding=entity_context,
			static_features=s_i,
			z_i=encoder_output.z_i,
			macro_controls=valid_macro_controls,
		)
		y_pred = self.regression_head(backbone_output.sequence).squeeze(-1)

		return CMDLModelOutput(
			y_pred=y_pred,
			omega=lag_gate_output.omega,
			z_i=encoder_output.z_i,
			p_hat_i=encoder_output.p_hat_i,
			k_star=lag_gate_output.k_star,
			lag_context_sequence=lag_context_sequence,
			backbone_sequence=backbone_output.sequence,
		)


__all__ = ["CMDLModel", "CMDLModelOutput"]
