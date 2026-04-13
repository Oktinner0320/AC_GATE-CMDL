"""CMDL 合成数据生成模块，用于 Step 1 机制验证。
Synthetic panel generator for Step 1 mechanism verification in CMDL.
"""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from config.cmdl_config import CMDLConfig


@dataclass(slots=True)
class SyntheticPanel:
	"""合成面板数据容器，统一封装模型输入与 ground truth。
	Container for synthetic panel tensors and their corresponding ground truth.
	"""

	X_it: torch.Tensor         # 时序输入张量 [N, T, F] | Sequential inputs.
	p_i: torch.Tensor          # 实体级代理变量 [N, M] | Entity-level proxy variables.
	s_i: torch.Tensor          # 实体级静态特征 [N, S] | Static entity features.
	Y_it: torch.Tensor         # 目标序列 [N, T] | Generated target sequence.
	z_true: torch.Tensor       # 真实潜变量 [N] | Ground-truth latent moderator.
	kstar_true: torch.Tensor   # 真实最优滞后 [N] | Ground-truth optimal lag.
	entity_ids: torch.Tensor   # 实体编号 | Entity identifiers.
	time_index: torch.Tensor   # 时间索引 | Time index.
	metadata: dict[str, Any]   # 生成元信息 | Generation metadata.


def _compute_kstar(z_true: np.ndarray, max_lag: int, scenario: str) -> np.ndarray:
	"""根据 z_true 计算实体级最优滞后。
	Compute entity-specific optimal lag values from the latent score.
	"""

	if scenario == "linear":
		raw = np.rint(3.0 + 7.0 * (1.0 - z_true))
	elif scenario == "nonlinear":
		raw = np.rint(10.0 * np.square(1.0 - z_true))
	else:
		raise ValueError(f"Unsupported scenario: {scenario}")

	return np.clip(raw.astype(np.float32), 1.0, float(max_lag))


def _generate_proxies(z_true: np.ndarray, cfg: CMDLConfig, rng: np.random.Generator) -> np.ndarray:
	"""从 z_true 派生带噪声代理变量，保留可学习相关性。
	Derive noisy proxy variables from z_true while preserving learnable signal.
	"""

	signals = [
		z_true,
		np.sqrt(np.clip(z_true, 1e-6, None)),
		1.0 - 0.7 * z_true,
		np.log1p(2.0 * z_true),
		0.5 + 0.8 * np.square(z_true),
		0.2 + 0.6 * z_true + 0.2 * np.square(z_true),
	]
	proxies = np.stack(signals[: cfg.n_proxies], axis=-1)
	proxies += rng.normal(0.0, cfg.noise_std * 0.35, size=proxies.shape)
	return proxies.astype(np.float32)


def _generate_static_features(z_true: np.ndarray, cfg: CMDLConfig, rng: np.random.Generator) -> np.ndarray:
	"""生成静态实体特征，其中部分维度与 z_true 相关。
	Generate static entity features with both correlated and independent dimensions.
	"""

	columns = [
		z_true + rng.normal(0.0, cfg.noise_std * 0.40, size=z_true.shape),
		rng.normal(0.0, 1.0, size=z_true.shape),
	]
	while len(columns) < cfg.static_dim:
		columns.append(0.4 * z_true + rng.normal(0.0, 0.8, size=z_true.shape))
	return np.stack(columns[: cfg.static_dim], axis=-1).astype(np.float32)


def _generate_inputs(z_true: np.ndarray, cfg: CMDLConfig, rng: np.random.Generator) -> np.ndarray:
	"""生成 AR 风格的输入序列，作为后续滞后聚合的来源。
	Generate autoregressive input sequences used by the lag aggregation mechanism.
	"""

	n_entities = cfg.n_entities
	x = np.zeros((n_entities, cfg.seq_length, cfg.seq_features), dtype=np.float32)
	entity_shift = (z_true - 0.5).reshape(-1, 1)

	for feature_index in range(cfg.seq_features):
		innovations = rng.normal(0.0, 1.0, size=(n_entities, cfg.seq_length)).astype(np.float32)
		feature_scale = 1.0 + 0.2 * feature_index
		x[:, 0, feature_index] = feature_scale * (0.6 * innovations[:, 0] + 0.4 * entity_shift[:, 0])

		for time_step in range(1, cfg.seq_length):
			x[:, time_step, feature_index] = (
				0.65 * x[:, time_step - 1, feature_index]
				+ 0.35 * feature_scale * innovations[:, time_step]
				+ 0.10 * entity_shift[:, 0]
			)

	return x.astype(np.float32)


def _build_lag_weights(kstar_true: np.ndarray, max_lag: int) -> np.ndarray:
	"""围绕 kstar_true 构造平滑滞后权重分布。
	Build smooth lag-weight distributions centered around kstar_true.
	"""

	lag_index = np.arange(1, max_lag + 1, dtype=np.float32)
	scaled_distance = (lag_index[None, :] - kstar_true[:, None]) / 1.25
	weights = np.exp(-0.5 * np.square(scaled_distance))
	# 归一化后每个实体得到一条合法的 lag 概率分布。
	weights /= weights.sum(axis=1, keepdims=True)
	return weights.astype(np.float32)


def _generate_targets(
	X_it: np.ndarray,
	lag_weights: np.ndarray,
	z_true: np.ndarray,
	s_i: np.ndarray,
	cfg: CMDLConfig,
	rng: np.random.Generator,
) -> np.ndarray:
	"""用实体特有的 lag 权重生成目标序列。
	Generate targets using entity-specific lag-weighted historical context.
	"""

	padded = np.pad(X_it, ((0, 0), (cfg.max_lag, 0), (0, 0)), mode="constant")
	# 为每个时间点截取长度为 K 的历史窗口，窗口方向随后会翻转以对齐 lag 编号。
	windows = np.stack(
		[padded[:, time_step : time_step + cfg.max_lag, :] for time_step in range(cfg.seq_length)],
		axis=1,
	)
	# lag_weights 按离 k* 的距离衰减，突出最优滞后及其邻近位置。
	weighted_windows = windows * lag_weights[:, None, ::-1, None]
	lag_context = weighted_windows.sum(axis=2)

	feature_coeffs = np.linspace(1.0, 0.7, cfg.seq_features, dtype=np.float32)
	signal = np.einsum("ntf,f->nt", lag_context, feature_coeffs)
	contemporaneous = np.einsum("ntf,f->nt", X_it, 0.15 * feature_coeffs)
	target = signal + contemporaneous
	# 额外叠加少量 z_true 与静态项，避免目标完全退化为单一线性卷积。
	target += 0.25 * z_true[:, None]
	target += 0.10 * s_i[:, 0:1]
	target += rng.normal(0.0, cfg.noise_std, size=target.shape).astype(np.float32)
	return target.astype(np.float32)


def generate_cmdl_synthetic(cfg: CMDLConfig | None = None, **overrides: Any) -> SyntheticPanel:
	"""生成一份可直接送入后续模块的合成面板数据。
	Generate a synthetic panel dataset ready for downstream CMDL modules.
	"""

	if cfg is None:
		cfg = CMDLConfig.from_domain("synthetic", **overrides)
	elif overrides:
		merged = cfg.to_dict()
		merged.update(overrides)
		cfg = CMDLConfig(**merged)

	rng = np.random.default_rng(cfg.seed)
	z_true = rng.uniform(0.0, 1.0, size=cfg.n_entities).astype(np.float32)
	kstar_true = _compute_kstar(z_true, cfg.max_lag, cfg.scenario)
	p_i = _generate_proxies(z_true, cfg, rng)
	s_i = _generate_static_features(z_true, cfg, rng)
	X_it = _generate_inputs(z_true, cfg, rng)
	lag_weights = _build_lag_weights(kstar_true, cfg.max_lag)
	Y_it = _generate_targets(X_it, lag_weights, z_true, s_i, cfg, rng)

	return SyntheticPanel(
		X_it=torch.tensor(X_it, dtype=torch.float32),
		p_i=torch.tensor(p_i, dtype=torch.float32),
		s_i=torch.tensor(s_i, dtype=torch.float32),
		Y_it=torch.tensor(Y_it, dtype=torch.float32),
		z_true=torch.tensor(z_true, dtype=torch.float32),
		kstar_true=torch.tensor(kstar_true, dtype=torch.float32),
		entity_ids=torch.arange(cfg.n_entities, dtype=torch.long),
		time_index=torch.arange(cfg.seq_length, dtype=torch.long),
		metadata={
			"domain": cfg.domain,
			"scenario": cfg.scenario,
			"max_lag": cfg.max_lag,
			"warmup_steps": cfg.max_lag,
			"seed": cfg.seed,
		},
	)


def summarize_synthetic_data(panel: SyntheticPanel) -> dict[str, float]:
	"""输出关键统计量，便于快速检查生成质量。
	Summarize key statistics for a quick synthetic data sanity check.
	"""

	z_true = panel.z_true.detach().cpu().numpy()
	kstar_true = panel.kstar_true.detach().cpu().numpy()
	p_i = panel.p_i.detach().cpu().numpy()

	summary = {
		"n_entities": float(panel.X_it.shape[0]),
		"seq_length": float(panel.X_it.shape[1]),
		"seq_features": float(panel.X_it.shape[2]),
		"z_min": float(z_true.min()),
		"z_max": float(z_true.max()),
		"kstar_min": float(kstar_true.min()),
		"kstar_max": float(kstar_true.max()),
		"proxy_mean": float(p_i.mean()),
		"proxy_std": float(p_i.std()),
	}
	return summary


def plot_z_vs_kstar(panel: SyntheticPanel, ax: plt.Axes | None = None) -> plt.Axes:
	"""绘制 z_true 与 kstar_true 的关系散点图。
	Plot a scatter chart of z_true against kstar_true.
	"""

	if ax is None:
		_, ax = plt.subplots(figsize=(8, 6))

	z_true = panel.z_true.detach().cpu().numpy()
	kstar_true = panel.kstar_true.detach().cpu().numpy()

	ax.scatter(z_true, kstar_true, alpha=0.75, s=28)
	ax.set_xlabel("z_true")
	ax.set_ylabel("kstar_true")
	ax.set_title(f"Synthetic lag structure ({panel.metadata['scenario']})")
	ax.grid(alpha=0.25)
	return ax


if __name__ == "__main__":
	# 直接运行文件时，打印摘要并显示 ground-truth 可视化结果。
	panel = generate_cmdl_synthetic()
	print(summarize_synthetic_data(panel))
	plot_z_vs_kstar(panel)
	plt.show()


__all__ = ["SyntheticPanel", "generate_cmdl_synthetic", "plot_z_vs_kstar", "summarize_synthetic_data"]
