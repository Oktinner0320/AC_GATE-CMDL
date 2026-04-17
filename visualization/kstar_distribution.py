"""Step 4 k* 预测散点图工具。

Scatter plot utilities for predicted-versus-true k* analysis in Step 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.metrics import compute_mae, compute_spearman


def _to_numpy(value: Any) -> np.ndarray:
	"""Convert tensors or array-like inputs into squeezed float64 arrays.

	将张量或类数组输入转换为压缩后的 float64 数组。
	"""

	if isinstance(value, torch.Tensor):
		array = value.detach().cpu().numpy()
	else:
		array = np.asarray(value)
	return np.asarray(np.squeeze(array), dtype=np.float64)


def plot_kstar_scatter(
	k_pred: Any,
	k_true: Any,
	save_path: str | Path | None,
	z_values: Any | None = None,
):
	"""Plot predicted versus true k* values with optional z quantile coloring.

	绘制预测 k* 与真实 k* 的散点图，并可按 z 分位数着色。
	"""

	k_pred_array = _to_numpy(k_pred).reshape(-1)
	k_true_array = _to_numpy(k_true).reshape(-1)

	if k_pred_array.shape != k_true_array.shape:
		raise ValueError(f"k_pred and k_true must match; got {k_pred_array.shape} vs {k_true_array.shape}")

	mae = compute_mae(k_pred_array, k_true_array)
	spearman_rho, _ = compute_spearman(k_pred_array, k_true_array)

	fig, ax = plt.subplots(figsize=(7, 6))
	if z_values is not None:
		z_array = _to_numpy(z_values).reshape(-1)
		if z_array.shape != k_true_array.shape:
			raise ValueError("z_values must align with k_pred and k_true")
		quantiles = np.quantile(z_array, [0.25, 0.50, 0.75])
		groups = np.digitize(z_array, quantiles, right=True)
		palette = ["#D1495B", "#EDAE49", "#66A182", "#2E4057"]
		labels = ["Q1", "Q2", "Q3", "Q4"]
		# Color-code entities by z quantile to reveal whether error varies by latent regime.
		# 按 z 分位数着色实体，以观察误差是否随潜在状态区间变化。
		for group_index in range(4):
			mask = groups == group_index
			ax.scatter(
				k_true_array[mask],
				k_pred_array[mask],
				s=28,
				alpha=0.8,
				color=palette[group_index],
				label=labels[group_index],
			)
		ax.legend(frameon=False, title="z quantile")
	else:
		ax.scatter(k_true_array, k_pred_array, s=28, alpha=0.8, color="#2E4057")

	lower = min(k_true_array.min(), k_pred_array.min()) - 0.5
	upper = max(k_true_array.max(), k_pred_array.max()) + 0.5
	ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1.0)
	ax.set_xlim(lower, upper)
	ax.set_ylim(lower, upper)
	ax.set_xlabel("True k*")
	ax.set_ylabel("Predicted k*")
	ax.set_title("Predicted vs true k*")
	ax.text(
		0.04,
		0.96,
		f"MAE = {mae:.3f}\nSpearman rho = {spearman_rho:.3f}",
		transform=ax.transAxes,
		ha="left",
		va="top",
		bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
	)
	ax.grid(alpha=0.2)

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.tight_layout()
		fig.savefig(save_path, dpi=200, bbox_inches="tight")

	return ax


__all__ = ["plot_kstar_scatter"]
