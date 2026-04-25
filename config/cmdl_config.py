"""CMDL 配置模块，集中定义 Core 阶段所需超参数。
Configuration module for CMDL core experiments and synthetic data presets.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal


DomainLiteral = Literal["synthetic", "shadow", "energy", "economics"]
ScenarioLiteral = Literal["linear", "nonlinear"]
OmegaTransformLiteral = Literal["softmax", "sparsemax"]


@dataclass(slots=True)
class CMDLConfig:
	"""CMDL 核心配置对象，统一管理模型与数据参数。
	Central configuration object for model hyperparameters and data settings.
	"""

	domain: DomainLiteral = "synthetic"     # 数据域预设名称 | Domain preset name.
	max_lag: int = 10                       # 最大滞后阶数 | Maximum lag horizon.
	d_model: int = 64                       # 主干网络隐藏维度 | Hidden size for downstream model blocks.
	n_proxies: int = 3                      # AC 代理变量数量 | Number of AC proxy variables.
	lambda_r: float = 1.0                   # 代理重构损失权重 | Weight of proxy reconstruction loss.
	temperature: float = 1.0                # 门控 softmax 温度 | Softmax temperature for lag gate.
	lag_bias_strength: float = 0.0          # 相对位置偏置强度 | Strength of relative position bias in lag gate.
	n_entities: int = 200                   # 实体数量 | Number of entities in a panel.
	seq_length: int = 30                    # 时间序列长度 | Sequence length per entity.
	seq_features: int = 1                   # 时序输入特征数 | Number of sequential input features.
	static_dim: int = 2                     # 静态特征维度 | Number of static entity features.
	lstm_layers: int = 2                    # LSTM 层数 | Number of LSTM layers in the backbone.
	dropout: float = 0.05                  # 通用 dropout 比例 | Shared dropout rate for Step 3 blocks.
	noise_std: float = 0.15                 # 合成数据噪声强度 | Noise level used in synthetic generation.
	reconstruction_detach: bool = True      # proxy 重构是否截断 z 梯度 | Whether proxy reconstruction detaches z.
	omega_transform: OmegaTransformLiteral = "softmax"  # lag logits 到 omega 的映射 | Mapping from lag logits to omega.
	lambda_omega_entropy: float = 0.0       # omega 熵区间正则权重 | Entropy-band penalty weight for omega.
	omega_entropy_min: float | None = None  # omega 熵下界 | Lower entropy bound for omega.
	omega_entropy_max: float | None = None  # omega 熵上界 | Upper entropy bound for omega.
	lambda_z_anchor: float = 0.0            # z-anchor 方向弱约束权重 | Weak z-anchor alignment weight.
	z_anchor_target_sign: float = 1.0       # z 与 anchor proxy 的预期方向 | Expected z-anchor direction.
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
		if self.omega_transform not in {"softmax", "sparsemax"}:
			raise ValueError(f"Unsupported omega_transform: {self.omega_transform}")
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
		# Step 3 新增的 backbone 超参数也在配置层统一校验。
		# Step 3 backbone hyperparameters are validated centrally in the config layer.
		if self.lstm_layers < 1:
			raise ValueError("lstm_layers must be at least 1")
		if self.lambda_r < 0.0:
			raise ValueError("lambda_r must be non-negative")
		if not 0.0 <= self.dropout < 1.0:
			raise ValueError("dropout must be in [0, 1)")
		if self.temperature <= 0.0:
			raise ValueError("temperature must be positive")
		if self.lag_bias_strength < 0.0:
			raise ValueError("lag_bias_strength must be non-negative")
		if self.noise_std < 0.0:
			raise ValueError("noise_std must be non-negative")
		if self.lambda_omega_entropy < 0.0:
			raise ValueError("lambda_omega_entropy must be non-negative")
		if self.omega_entropy_min is not None and self.omega_entropy_min < 0.0:
			raise ValueError("omega_entropy_min must be non-negative when provided")
		if self.omega_entropy_max is not None and self.omega_entropy_max < 0.0:
			raise ValueError("omega_entropy_max must be non-negative when provided")
		if (
			self.omega_entropy_min is not None
			and self.omega_entropy_max is not None
			and self.omega_entropy_min > self.omega_entropy_max
		):
			raise ValueError("omega_entropy_min cannot exceed omega_entropy_max")
		if self.lambda_z_anchor < 0.0:
			raise ValueError("lambda_z_anchor must be non-negative")
		if self.z_anchor_target_sign == 0.0:
			raise ValueError("z_anchor_target_sign must be non-zero")

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
				"lambda_r": 1.0,
				"temperature": 1.0,
				"lag_bias_strength": 0.0,
				"n_entities": 200,
				"seq_length": 30,
				"seq_features": 1,
				"static_dim": 2,
				"lstm_layers": 2,
				"dropout": 0.05,
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
				"lstm_layers": 2,
				"dropout": 0.05,
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
				"lstm_layers": 2,
				"dropout": 0.05,
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
				"lstm_layers": 2,
				"dropout": 0.05,
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


__all__ = ["CMDLConfig", "DomainLiteral", "OmegaTransformLiteral", "ScenarioLiteral"]
