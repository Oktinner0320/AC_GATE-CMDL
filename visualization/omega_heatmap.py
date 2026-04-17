"""Step 4 omega 热力图工具。

Omega heatmap utilities for Step 4 synthetic experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def _to_numpy(value: Any) -> np.ndarray:
	"""Convert tensors or array-like inputs into squeezed float64 arrays.

	将张量或类数组输入转换为压缩后的 float64 数组。
	"""

	if isinstance(value, torch.Tensor):
		array = value.detach().cpu().numpy()
	else:
		array = np.asarray(value)
	return np.asarray(np.squeeze(array), dtype=np.float64)


def plot_omega_heatmap(
	omega: Any,
	z_values: Any,
	kstar_true: Any,
	save_path: str | Path | None,
):
	"""Plot omega as a heatmap after sorting entities by descending z.

	按 z 从高到低排序实体后绘制 omega 热力图。
	"""

	omega_array = _to_numpy(omega)
	z_array = _to_numpy(z_values).reshape(-1)
	kstar_array = _to_numpy(kstar_true).reshape(-1)

	if omega_array.ndim != 2:
		raise ValueError(f"Expected omega with shape [N, K], got {omega_array.shape}")
	if omega_array.shape[0] != z_array.shape[0] or omega_array.shape[0] != kstar_array.shape[0]:
		raise ValueError("omega, z_values, and kstar_true must have the same number of entities")

	# Order entities by z so the lag pattern is easier to inspect visually.
	# 按 z 排序实体，便于直观看到滞后模式随 z 的变化。
	sort_index = np.argsort(z_array)[::-1]
	omega_sorted = omega_array[sort_index]
	kstar_sorted = kstar_array[sort_index]

	figure_width = max(6.0, 0.9 * omega_sorted.shape[1])
	figure_height = min(14.0, max(6.0, 0.045 * omega_sorted.shape[0] + 1.5))
	fig, ax = plt.subplots(figsize=(figure_width, figure_height))

	sns.heatmap(
		omega_sorted,
		ax=ax,
		cmap="YlOrRd",
		xticklabels=np.arange(1, omega_sorted.shape[1] + 1),
		yticklabels=False,
		cbar_kws={"label": "omega"},
	)

	x_coords = kstar_sorted - 0.5
	y_coords = np.arange(kstar_sorted.shape[0]) + 0.5
	# Overlay the true k* location on top of the heatmap for comparison.
	# 在热力图上叠加真实 k* 位置，便于和学习到的权重峰值对比。
	ax.scatter(
		x_coords,
		y_coords,
		s=14,
		c="black",
		edgecolors="white",
		linewidths=0.3,
	)
	ax.set_xlabel("Lag index k")
	ax.set_ylabel("Entities sorted by z")
	ax.set_title("Omega heatmap ordered by z")

	if save_path is not None:
		save_path = Path(save_path)
		save_path.parent.mkdir(parents=True, exist_ok=True)
		fig.tight_layout()
		fig.savefig(save_path, dpi=200, bbox_inches="tight")

	return ax


__all__ = ["plot_omega_heatmap"]
