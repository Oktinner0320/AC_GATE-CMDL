"""CMDL 配置模块，集中定义 Core 阶段所需超参数。
Configuration module for CMDL core experiments and synthetic data presets.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal


DomainLiteral = Literal["synthetic", "shadow", "energy", "economics"]
ScenarioLiteral = Literal["linear", "nonlinear"]


@dataclass(slots=True)
class CMDLConfig:
	"""CMDL 核心配置对象，统一管理模型与数据参数。
	Central configuration object for model hyperparameters and data settings.
	"""

	domain: DomainLiteral = "synthetic"     # 数据域预设名称 | Domain preset name.
	max_lag: int = 10                       # 最大滞后阶数 | Maximum lag horizon.
	d_model: int = 64                       # 主干网络隐藏维度 | Hidden size for downstream model blocks.
	n_proxies: int = 3                      # AC 代理变量数量 | Number of AC proxy variables.
	lambda_r: float = 0.1                   # 代理重构损失权重 | Weight of proxy reconstruction loss.
	temperature: float = 1.0                # 门控 softmax 温度 | Softmax temperature for lag gate.
	lag_bias_strength: float = 1.0          # 相对位置偏置强度 | Strength of relative position bias in lag gate.
	n_entities: int = 200                   # 实体数量 | Number of entities in a panel.
	seq_length: int = 30                    # 时间序列长度 | Sequence length per entity.
	seq_features: int = 1                   # 时序输入特征数 | Number of sequential input features.
	static_dim: int = 2                     # 静态特征维度 | Number of static entity features.
	noise_std: float = 0.15                 # 合成数据噪声强度 | Noise level used in synthetic generation.
	seed: int = 42                          # 随机种子 | Random seed for reproducibility.
	scenario: ScenarioLiteral = "linear"    # 合成场景类型 | Synthetic scenario type.

	def __post_init__(self) -> None:
		self.validate()

	def validate(self) -> None:
		"""校验配置约束，避免无效组合进入训练流程。
		Validate configuration constraints before later modules consume them.
		"""

		valid_domains = {"synthetic", "shadow", "energy", "economics"}
		valid_scenarios = {"linear", "nonlinear"}

		if self.domain not in valid_domains:
			raise ValueError(f"Unsupported domain: {self.domain}")
		if self.scenario not in valid_scenarios:
			raise ValueError(f"Unsupported synthetic scenario: {self.scenario}")
		# 序列长度必须大于滞后窗口，否则无法构造有效历史上下文。
		if self.max_lag < 1:
			raise ValueError("max_lag must be at least 1")
		if self.seq_length <= self.max_lag:
			raise ValueError("seq_length must be greater than max_lag")
		if self.seq_features < 1:
			raise ValueError("seq_features must be at least 1")
		if self.static_dim < 1:
			raise ValueError("static_dim must be at least 1")
		if self.n_proxies < 1:
			raise ValueError("n_proxies must be at least 1")
		if self.n_entities < 1:
			raise ValueError("n_entities must be at least 1")
		if self.lambda_r < 0.0:
			raise ValueError("lambda_r must be non-negative")
		if self.temperature <= 0.0:
			raise ValueError("temperature must be positive")
		if self.lag_bias_strength < 0.0:
			raise ValueError("lag_bias_strength must be non-negative")
		if self.noise_std < 0.0:
			raise ValueError("noise_std must be non-negative")

	@classmethod
	def from_domain(cls, domain: DomainLiteral, **overrides: Any) -> "CMDLConfig":
		"""按域名加载默认配置，并允许局部覆盖。
		Build a preset configuration for a domain and apply optional overrides.
		"""

		presets: dict[str, dict[str, Any]] = {
			# synthetic 直接对齐 Step 1 的机制验证规模。
			"synthetic": {
				"domain": "synthetic",
				"max_lag": 10,
				"d_model": 64,
				"n_proxies": 3,
				"lambda_r": 0.1,
				"temperature": 1.0,
				"lag_bias_strength": 1.0,
				"n_entities": 200,
				"seq_length": 30,
				"seq_features": 1,
				"static_dim": 2,
				"noise_std": 0.15,
				"seed": 42,
				"scenario": "linear",
			},
			# shadow 对齐 Medina & Schneider (2018) 影子经济面板。
			# 158 国 × 1991-2015 (25 年)，proxy: 治理质量 / 金融包容度 / 执法效率。
			"shadow": {
				"domain": "shadow",
				"max_lag": 10,
				"d_model": 64,
				"n_proxies": 3,
				"lambda_r": 0.1,
				"temperature": 1.0,
				"lag_bias_strength": 1.0,
				"n_entities": 158,
				"seq_length": 25,
				"seq_features": 1,
				"static_dim": 2,
				"noise_std": 0.10,
				"seed": 42,
				"scenario": "linear",
			},
			# energy 与 economics 作为泛化验证域。
			"energy": {
				"domain": "energy",
				"max_lag": 10,
				"d_model": 64,
				"n_proxies": 3,
				"lambda_r": 0.1,
				"temperature": 1.0,
				"lag_bias_strength": 1.0,
				"n_entities": 180,
				"seq_length": 35,
				"seq_features": 1,
				"static_dim": 2,
				"noise_std": 0.10,
				"seed": 42,
				"scenario": "linear",
			},
			"economics": {
				"domain": "economics",
				"max_lag": 10,
				"d_model": 64,
				"n_proxies": 1,
				"lambda_r": 0.1,
				"temperature": 1.0,
				"lag_bias_strength": 1.0,
				"n_entities": 150,
				"seq_length": 40,
				"seq_features": 1,
				"static_dim": 2,
				"noise_std": 0.10,
				"seed": 42,
				"scenario": "linear",
			},
		}

		if domain not in presets:
			raise ValueError(f"Unsupported domain preset: {domain}")

		preset = presets[domain].copy()
		preset.update(overrides)
		return cls(**preset)

	def to_dict(self) -> dict[str, Any]:
		"""导出为字典，便于日志记录或实验追踪。
		Serialize config into a plain dictionary for logging and tracking.
		"""

		return asdict(self)


__all__ = ["CMDLConfig", "DomainLiteral", "ScenarioLiteral"]
