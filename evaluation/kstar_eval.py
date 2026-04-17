"""Step 4 k*、z 与 omega 的专项评估工具。

Specialized evaluation helpers for Step 4 k* recovery, z identification, and omega diagnostics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from evaluation.metrics import compute_mae, compute_mse, compute_r2, compute_spearman


def _to_numpy(value: Any) -> np.ndarray:
	"""Convert tensors or array-like inputs into squeezed float64 arrays.

	将张量或类数组输入转换为压缩后的 float64 数组。
	"""

	if isinstance(value, torch.Tensor):
		array = value.detach().cpu().numpy()
	else:
		array = np.asarray(value)
	return np.asarray(np.squeeze(array), dtype=np.float64)


def evaluate_kstar(k_pred: Any, k_true: Any) -> dict[str, float]:
	"""Evaluate predicted optimal lags against ground truth.

	评估预测得到的最优滞后与真实 k* 之间的一致性。
	"""

	spearman_rho, spearman_p = compute_spearman(k_pred, k_true)
	return {
		"mae": compute_mae(k_pred, k_true),
		"rmse": float(np.sqrt(compute_mse(k_pred, k_true))),
		"spearman_rho": spearman_rho,
		"spearman_p": spearman_p,
	}


def evaluate_z_identification(
	z_pred: Any,
	z_true: Any,
	p_pred: Any,
	p_true: Any,
) -> dict[str, float]:
	"""Evaluate whether learned z preserves proxy semantics and latent ordering.

	评估学习到的 z 是否保留 proxy 语义以及潜变量排序关系。
	"""

	z_spearman_rho, z_spearman_p = compute_spearman(z_pred, z_true)
	return {
		"z_spearman_rho": z_spearman_rho,
		"z_spearman_p": z_spearman_p,
		"proxy_recon_r2": compute_r2(p_pred, p_true),
	}


def evaluate_omega_distribution(omega: Any, kstar_true: Any) -> dict[str, float]:
	"""Return diagnostic statistics for omega sharpness and peak placement.

	返回 omega 分布尖锐程度与峰值位置的诊断统计量。
	"""

	omega_array = _to_numpy(omega)
	kstar_array = _to_numpy(kstar_true).reshape(-1)

	if omega_array.ndim != 2:
		raise ValueError(f"Expected omega with shape [N, K], got {omega_array.shape}")
	if omega_array.shape[0] != kstar_array.shape[0]:
		raise ValueError(
			"omega and kstar_true must agree on the entity dimension; "
			f"got {omega_array.shape[0]} vs {kstar_array.shape[0]}"
		)

	# Clip tiny probabilities to keep entropy numerically stable.
	# 对极小概率做裁剪，以保证熵计算的数值稳定性。
	clipped_omega = np.clip(omega_array, 1e-8, 1.0)
	entropy = -np.sum(clipped_omega * np.log(clipped_omega), axis=1)
	predicted_peak = np.argmax(omega_array, axis=1) + 1
	true_peak = np.rint(kstar_array).astype(np.int64)

	return {
		"entropy_mean": float(np.mean(entropy)),
		"peak_accuracy": float(np.mean(predicted_peak == true_peak)),
	}


__all__ = ["evaluate_kstar", "evaluate_omega_distribution", "evaluate_z_identification"]
