"""吸收能力编码器模块。
Absorptive-capacity encoder used in Step 2.

该模块把实体级代理变量 p_i 压缩为标量潜变量 z_i，随后再从 z_i 重构回代理空间，
为后续的代理重构正则提供直接监督信号。
This module compresses entity-level proxy variables p_i into a scalar latent score z_i,
and reconstructs the proxy space from z_i to support later reconstruction regularization.
"""

from dataclasses import dataclass
import torch
from torch import nn


@dataclass(slots=True)
class ACEncoderOutput:
	"""AC 编码器的结构化输出。
	Structured outputs returned by the AC encoder.

	z_i 是标量吸收能力得分，p_hat_i 是从 z_i 反推得到的代理重构结果。
	z_i is the scalar absorptive-capacity score, while p_hat_i is the reconstructed proxy vector.
	"""

	z_i: torch.Tensor
	p_hat_i: torch.Tensor


class AdaptiveACEncoder(nn.Module):
	"""将实体级代理变量编码为标量 AC 得分并执行重构。
	Encode entity-level proxies into a scalar AC score and reconstruct them.

	整体流程分为两段：
	1. 编码器 MLP 从多维 proxy 中提炼共享隐藏表示；
	2. latent head 压缩为 z_i，reconstructor 再从 z_i 回推原始 proxy。
	The overall flow has two stages:
	1. an MLP encoder extracts a shared hidden representation from the proxy vector;
	2. the latent head compresses it into z_i, and the reconstructor maps z_i back to proxy space.
	"""

	def __init__(self, n_proxies:int, hidden_dim:int=32, bottleneck_dim:int=16) -> None:
		super().__init__()
		if n_proxies < 1:
			raise ValueError("n_proxies must be at least 1")
		if hidden_dim < 1 or bottleneck_dim < 1:
			raise ValueError("hidden dimensions must be positive")

		self.n_proxies = n_proxies
		# 主编码器先做特征混合与非线性变换，再把信息压到较小瓶颈层。
		# The main encoder mixes proxy features and compresses them into a smaller bottleneck.
		self.encoder = nn.Sequential(nn.Linear(n_proxies, hidden_dim), nn.LayerNorm(hidden_dim),
									nn.GELU(), nn.Linear(hidden_dim, bottleneck_dim), nn.GELU())
		# latent_head 负责输出标量 z_i；重构头负责把 z_i 投影回 proxy 空间。
		# The latent head emits scalar z_i, and the reconstruction head maps z_i back to proxies.
		self.latent_head = nn.Linear(bottleneck_dim, 1)
		self.proxy_reconstructor = nn.Linear(1, n_proxies)

	def forward(self, proxies:torch.Tensor) -> ACEncoderOutput:
		"""执行一次完整的编码与重构前向过程。
		Run the full encode-and-reconstruct forward pass.

		输入张量必须是 [B, M]，其中 B 是 batch 大小，M 是 proxy 维度。
		The input tensor must be shaped as [B, M], where B is batch size and M is proxy dimension.
		"""

		if proxies.dim() != 2:
			raise ValueError(f"Expected proxies with shape [B, M], got {tuple(proxies.shape)}")
		if proxies.size(-1) != self.n_proxies:
			raise ValueError(
				f"Expected {self.n_proxies} proxy features, got {proxies.size(-1)}"
			)

		# hidden 表示聚合多维 proxy 信息；z_i 是后续 Lag Gate 使用的实体级条件变量。
		# hidden aggregates proxy information; z_i becomes the entity-level conditioning signal for the lag gate.
		hidden = self.encoder(proxies)
		z_i = self.latent_head(hidden)
		# p_hat_i 用于后续 reconstruction loss；detach 防止 recon 梯度干扰 encoder 表示学习。
		# p_hat_i feeds reconstruction loss; detach prevents recon gradients from disturbing encoder representations.
		p_hat_i = self.proxy_reconstructor(z_i.detach())
		return ACEncoderOutput(z_i=z_i, p_hat_i=p_hat_i)


__all__ = ["ACEncoderOutput", "AdaptiveACEncoder"]
