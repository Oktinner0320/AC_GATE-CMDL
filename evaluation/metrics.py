"""Step 4 通用评估指标。
Common metric helpers used by synthetic experiments and downstream analysis.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def _to_numpy(value: Any) -> np.ndarray:
	"""Convert tensors or array-like inputs into float64 numpy arrays.

	将张量或类数组输入统一转换为 float64 的 numpy 数组。
	"""

	if isinstance(value, torch.Tensor):
		array = value.detach().cpu().numpy()
	else:
		array = np.asarray(value)

	if array.ndim == 0:
		array = array.reshape(1)
	return np.asarray(array, dtype=np.float64)


def _prepare_pair(pred: Any, true: Any) -> tuple[np.ndarray, np.ndarray]:
	"""Align prediction and target arrays while preserving 2D metric inputs.

	对齐预测值与真值数组，并尽量保留二维输入的结构。
	"""

	pred_array = np.squeeze(_to_numpy(pred))
	true_array = np.squeeze(_to_numpy(true))

	if pred_array.shape != true_array.shape:
		if pred_array.size != true_array.size:
			raise ValueError(
				"pred and true must have matching shapes or the same number of elements; "
				f"got {pred_array.shape} vs {true_array.shape}"
			)
		pred_array = pred_array.reshape(true_array.shape)

	return pred_array, true_array


def compute_mse(pred: Any, true: Any) -> float:
	"""Compute mean squared error.

	计算均方误差。
	"""

	pred_array, true_array = _prepare_pair(pred, true)
	return float(np.mean(np.square(pred_array - true_array)))


def compute_mae(pred: Any, true: Any) -> float:
	"""Compute mean absolute error.

	计算平均绝对误差。
	"""

	pred_array, true_array = _prepare_pair(pred, true)
	return float(np.mean(np.abs(pred_array - true_array)))


def compute_r2(pred: Any, true: Any) -> float:
	"""Compute R-squared with safe fallbacks for degenerate targets.

	计算 R²，并为退化目标值提供安全回退逻辑。
	"""

	pred_array, true_array = _prepare_pair(pred, true)
	if true_array.size < 2 or np.allclose(true_array, true_array.flat[0]):
		return 0.0

	# Degenerate targets can trigger warnings or NaNs; normalize them to a safe score.
	# 退化目标值可能触发警告或 NaN，这里统一回退为安全分数。
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		score = r2_score(true_array, pred_array, multioutput="uniform_average")

	if not np.isfinite(score):
		return 0.0
	return float(score)


def compute_spearman(pred: Any, true: Any) -> tuple[float, float]:
	"""Compute Spearman rank correlation and p-value.

	计算 Spearman 秩相关系数及其 p 值。
	"""

	pred_array, true_array = _prepare_pair(pred, true)
	pred_flat = pred_array.reshape(-1)
	true_flat = true_array.reshape(-1)

	if pred_flat.size < 2:
		return 0.0, 1.0
	if np.allclose(pred_flat, pred_flat[0]) or np.allclose(true_flat, true_flat[0]):
		return 0.0, 1.0

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		rho, p_value = spearmanr(pred_flat, true_flat)

	if not np.isfinite(rho):
		rho = 0.0
	if not np.isfinite(p_value):
		p_value = 1.0
	return float(rho), float(p_value)


__all__ = ["compute_mae", "compute_mse", "compute_r2", "compute_spearman"]
