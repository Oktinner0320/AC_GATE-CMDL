"""Step 3 CMDL 模型的损失函数工具。
Loss utilities for the Step 3 CMDL model.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class DomainAgnosticLossOutput:
	"""便于日志记录和测试断言的损失拆分结果。
	Structured loss breakdown for logging and testing.
	"""

	total_loss: torch.Tensor
	task_loss: torch.Tensor
	recon_loss: torch.Tensor
	anchor_recon_loss: torch.Tensor
	omega_entropy_penalty: torch.Tensor
	omega_entropy_band_violation_share: torch.Tensor
	z_anchor_loss: torch.Tensor


class DomainAgnosticLoss(nn.Module):
	"""有效时间段上的任务 MSE 与 proxy 重构 MSE 之和。
	Task MSE plus proxy reconstruction MSE over the valid time range.
	"""

	def __init__(
		self,
		lambda_r: float = 0.1,
		warmup_steps: int = 0,
		recon_loss_mode: str = "all",
		anchor_proxy_index: int = 0,
		anchor_recon_weight: float = 1.0,
		lambda_omega_entropy: float = 0.0,
		omega_entropy_min: float | None = None,
		omega_entropy_max: float | None = None,
		lambda_z_anchor: float = 0.0,
		z_anchor_target_sign: float = 1.0,
	) -> None:
		super().__init__()
		if lambda_r < 0.0:
			raise ValueError("lambda_r must be non-negative")
		if warmup_steps < 0:
			raise ValueError("warmup_steps must be non-negative")
		if recon_loss_mode not in {"all", "anchor_only", "anchor_weighted"}:
			raise ValueError("recon_loss_mode must be one of: all, anchor_only, anchor_weighted")
		if anchor_proxy_index < 0:
			raise ValueError("anchor_proxy_index must be non-negative")
		if anchor_recon_weight <= 0.0:
			raise ValueError("anchor_recon_weight must be positive")
		if lambda_omega_entropy < 0.0:
			raise ValueError("lambda_omega_entropy must be non-negative")
		if omega_entropy_min is not None and omega_entropy_min < 0.0:
			raise ValueError("omega_entropy_min must be non-negative when provided")
		if omega_entropy_max is not None and omega_entropy_max < 0.0:
			raise ValueError("omega_entropy_max must be non-negative when provided")
		if (
			omega_entropy_min is not None
			and omega_entropy_max is not None
			and omega_entropy_min > omega_entropy_max
		):
			raise ValueError("omega_entropy_min cannot exceed omega_entropy_max")
		if lambda_z_anchor < 0.0:
			raise ValueError("lambda_z_anchor must be non-negative")
		if z_anchor_target_sign == 0.0:
			raise ValueError("z_anchor_target_sign must be non-zero")

		self.lambda_r = lambda_r
		self.warmup_steps = warmup_steps
		self.recon_loss_mode = recon_loss_mode
		self.anchor_proxy_index = anchor_proxy_index
		self.anchor_recon_weight = anchor_recon_weight
		self.lambda_omega_entropy = lambda_omega_entropy
		self.omega_entropy_min = omega_entropy_min
		self.omega_entropy_max = omega_entropy_max
		self.lambda_z_anchor = lambda_z_anchor
		self.z_anchor_target_sign = float(1.0 if z_anchor_target_sign > 0.0 else -1.0)

	def _reconstruction_loss(self, p_hat_i: torch.Tensor, p_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		if self.anchor_proxy_index >= p_i.size(1):
			raise ValueError(
				f"anchor_proxy_index {self.anchor_proxy_index} is out of bounds for {p_i.size(1)} proxies"
			)

		anchor_error = (p_hat_i[:, self.anchor_proxy_index] - p_i[:, self.anchor_proxy_index]) ** 2
		anchor_recon_loss = anchor_error.mean()
		if self.recon_loss_mode == "anchor_only":
			return anchor_recon_loss, anchor_recon_loss

		squared_error = (p_hat_i - p_i) ** 2
		if self.recon_loss_mode == "anchor_weighted":
			weights = torch.ones(p_i.size(1), dtype=p_i.dtype, device=p_i.device)
			weights[self.anchor_proxy_index] = self.anchor_recon_weight
			recon_loss = (squared_error * weights.unsqueeze(0)).sum() / (p_i.size(0) * weights.sum())
			return recon_loss, anchor_recon_loss

		return squared_error.mean(), anchor_recon_loss

	def _omega_entropy_penalty(self, omega: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
		if omega is None:
			zero = torch.tensor(0.0)
			return zero, zero
		if self.lambda_omega_entropy == 0.0:
			zero = torch.tensor(0.0, device=omega.device, dtype=omega.dtype)
			return zero, zero
		if omega.dim() != 2:
			raise ValueError(f"Expected omega with shape [B, K], got {tuple(omega.shape)}")

		clipped = omega.clamp_min(1e-8)
		entropy = -(clipped * clipped.log()).sum(dim=-1)
		violation = torch.zeros_like(entropy)
		if self.omega_entropy_min is not None:
			violation = violation + F.relu(float(self.omega_entropy_min) - entropy)
		if self.omega_entropy_max is not None:
			violation = violation + F.relu(entropy - float(self.omega_entropy_max))
		penalty = violation.pow(2).mean()
		violation_share = (violation > 0.0).to(dtype=omega.dtype).mean()
		return penalty, violation_share

	def _z_anchor_loss(self, z_i: torch.Tensor | None, p_i: torch.Tensor) -> torch.Tensor:
		if z_i is None or self.lambda_z_anchor == 0.0:
			return torch.tensor(0.0, device=p_i.device, dtype=p_i.dtype)
		if z_i.dim() != 2 or z_i.size(-1) != 1:
			raise ValueError(f"Expected z_i with shape [B, 1], got {tuple(z_i.shape)}")
		if self.anchor_proxy_index >= p_i.size(1):
			raise ValueError(
				f"anchor_proxy_index {self.anchor_proxy_index} is out of bounds for {p_i.size(1)} proxies"
			)

		z_values = z_i.squeeze(-1)
		anchor = p_i[:, self.anchor_proxy_index]
		z_centered = z_values - z_values.mean()
		anchor_centered = anchor - anchor.mean()
		denominator = z_centered.pow(2).mean().sqrt() * anchor_centered.pow(2).mean().sqrt()
		correlation = (z_centered * anchor_centered).mean() / denominator.clamp_min(1e-8)
		adjusted = self.z_anchor_target_sign * correlation
		return F.relu(0.0 - adjusted)

	def forward(
		self,
		y_pred: torch.Tensor,
		y_true: torch.Tensor,
		p_hat_i: torch.Tensor,
		p_i: torch.Tensor,
		omega: torch.Tensor | None = None,
		z_i: torch.Tensor | None = None,
	) -> DomainAgnosticLossOutput:
		if y_pred.dim() != 2:
			raise ValueError(f"Expected y_pred with shape [B, T_valid], got {tuple(y_pred.shape)}")
		if y_true.dim() != 2:
			raise ValueError(f"Expected y_true with shape [B, T], got {tuple(y_true.shape)}")
		if p_hat_i.shape != p_i.shape:
			raise ValueError(f"Expected p_hat_i and p_i to match, got {tuple(p_hat_i.shape)} vs {tuple(p_i.shape)}")

		# y_true 既支持已经裁掉 warm-up 的输入，也支持完整长度输入后在这里对齐。
		# y_true can either be pre-trimmed or be aligned here by removing the warm-up prefix.
		if y_true.size(1) == y_pred.size(1):
			aligned_target = y_true
		elif y_true.size(1) == y_pred.size(1) + self.warmup_steps:
			aligned_target = y_true[:, self.warmup_steps :]
		else:
			raise ValueError(
				"y_true must either match y_pred length or exceed it exactly by warmup_steps; "
				f"got {tuple(y_true.shape)} vs {tuple(y_pred.shape)}"
			)

		# total = task + lambda_r * recon，与 plan.md 中的 Step 3 定义保持一致。
		# total = task + lambda_r * recon, matching the Step 3 definition in plan.md.
		task_loss = F.mse_loss(y_pred, aligned_target)
		recon_loss, anchor_recon_loss = self._reconstruction_loss(p_hat_i, p_i)
		omega_entropy_penalty, omega_entropy_band_violation_share = self._omega_entropy_penalty(omega)
		omega_entropy_penalty = omega_entropy_penalty.to(device=y_pred.device, dtype=y_pred.dtype)
		omega_entropy_band_violation_share = omega_entropy_band_violation_share.to(device=y_pred.device, dtype=y_pred.dtype)
		z_anchor_loss = self._z_anchor_loss(z_i, p_i)
		total_loss = (
			task_loss
			+ self.lambda_r * recon_loss
			+ self.lambda_omega_entropy * omega_entropy_penalty
			+ self.lambda_z_anchor * z_anchor_loss
		)
		return DomainAgnosticLossOutput(
			total_loss=total_loss,
			task_loss=task_loss,
			recon_loss=recon_loss,
			anchor_recon_loss=anchor_recon_loss,
			omega_entropy_penalty=omega_entropy_penalty,
			omega_entropy_band_violation_share=omega_entropy_band_violation_share,
			z_anchor_loss=z_anchor_loss,
		)


__all__ = ["DomainAgnosticLoss", "DomainAgnosticLossOutput"]
