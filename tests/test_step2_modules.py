"""Step 2 测试脚本。
Step 2 test suite for the AC encoder and lag gate.

该文件覆盖三层验证：
1. 单元级形状契约；
2. Lag Gate 的概率分布与梯度回传；
3. 合成数据上的最小集成链路。
This file covers three layers of checks:
1. unit-level shape contracts;
2. probability and gradient behavior of the lag gate;
3. a minimal end-to-end integration path on synthetic data.
"""

import unittest
import os
import sys

# 直接运行测试文件时，解释器会把 tests 目录当作起点，
# 这里显式把工作区根目录加回 sys.path，确保 config/model/data 等包可以被导入。
# When the test file is executed directly, Python starts from the tests directory.
# Add the workspace root back to sys.path so project modules remain importable.
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Windows 上部分 PyTorch 依赖会触发重复 OpenMP runtime 报错；
# 测试阶段允许继续运行，避免把环境问题误判成模型逻辑问题。
# Some Windows PyTorch setups hit duplicate OpenMP runtime errors;
# allow execution to continue during tests so environment issues do not mask model logic.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

from config.cmdl_config import CMDLConfig
from data.synthetic.generate import generate_cmdl_synthetic
from model.ac_encoder import AdaptiveACEncoder
from model.lag_gate import ScaleInvariantLagGate


class AdaptiveACEncoderTest(unittest.TestCase):
    """验证 AC 编码器的基础张量契约。
    Validate the basic tensor contract of the AC encoder.
    """

    def test_encoder_returns_expected_shapes(self) -> None:
        """编码器应返回标量 z_i 和同维度代理重构。
        The encoder should return scalar z_i and same-width proxy reconstructions.
        """

        encoder = AdaptiveACEncoder(n_proxies=3)
        proxy_batch = torch.randn(6, 3)

        # 执行前向传播，检查输出是否符合 Step 2 的接口定义。
        # Run the forward pass and verify the Step 2 output contract.
        output = encoder(proxy_batch)

        self.assertEqual(tuple(output.z_i.shape), (6, 1))
        self.assertEqual(tuple(output.p_hat_i.shape), (6, 3))


class ScaleInvariantLagGateTest(unittest.TestCase):
    """验证 Lag Gate 的分布合法性和可训练性。
    Validate the lag gate's distributional correctness and trainability.
    """

    def test_gate_outputs_valid_distribution(self) -> None:
        """omega 应为合法概率分布，k_star 应落在有效滞后范围内。
        omega should be a valid probability distribution and k_star should stay within the lag range.
        """

        gate = ScaleInvariantLagGate(max_lag=5, d_model=8, temperature=0.7, lag_bias_strength=0.5)
        z_i = torch.randn(4, 1)
        lagged_sequence = torch.randn(4, 5, 8)

        # 给定标量 z_i 和长度为 K 的历史窗口，Lag Gate 需要同时返回分布和聚合上下文。
        # Given scalar z_i and a K-step lagged window, the lag gate should return both the distribution and context.
        output = gate(z_i, lagged_sequence)

        self.assertEqual(tuple(output.omega.shape), (4, 5))
        self.assertEqual(tuple(output.k_star.shape), (4,))
        self.assertEqual(tuple(output.lag_context.shape), (4, 8))
        # 每个样本的 omega 都应在 lag 维度上归一化到 1。
        # Each sample's omega should sum to 1 across the lag dimension.
        torch.testing.assert_close(output.omega.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)
        self.assertGreaterEqual(output.k_star.min().item(), 1.0)
        self.assertLessEqual(output.k_star.max().item(), 5.0)

    def test_gate_backpropagates_to_latent_score(self) -> None:
        """梯度应能从输出回传到输入 z_i，证明门控路径可训练。
        Gradients should flow back to z_i, proving the gating path is trainable.
        """

        gate = ScaleInvariantLagGate(max_lag=6, d_model=4)
        z_i = torch.randn(3, 1, requires_grad=True)
        lagged_sequence = torch.randn(3, 6, 4)

        output = gate(z_i, lagged_sequence)
        # 用一个简单的可导目标组合 lag_context 和 k_star，检查反向传播是否打通。
        # Build a simple differentiable objective from lag_context and k_star to verify backpropagation.
        loss = output.lag_context.pow(2).mean() + output.k_star.mean()
        loss.backward()

        self.assertIsNotNone(z_i.grad)
        self.assertGreater(z_i.grad.abs().sum().item(), 0.0)

    def test_sparsemax_transform_outputs_sparse_distribution(self) -> None:
        """sparsemax 选项应保持概率合法，并允许精确零权重。"""

        gate = ScaleInvariantLagGate(max_lag=5, d_model=4, omega_transform="sparsemax", dropout=0.0)
        z_i = torch.tensor([[-3.0], [0.0], [3.0]])

        output = gate(z_i)

        self.assertEqual(tuple(output.omega.shape), (3, 5))
        torch.testing.assert_close(output.omega.sum(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.any(output.omega == 0.0))


class Step2IntegrationTest(unittest.TestCase):
    """在合成数据上验证编码器和门控器的最小集成链路。
    Validate the minimal integration path between the encoder and lag gate on synthetic data.
    """

    def test_encoder_and_gate_run_on_synthetic_batch(self) -> None:
        """Step 2 两个核心模块应能在同一 batch 上完成前向与反向。
        The two Step 2 core modules should complete a forward and backward pass on the same batch.
        """

        # 使用小规模 synthetic 配置，保证测试足够快，同时保留真实接口形状。
        # Use a small synthetic configuration to keep the test fast while preserving real interface shapes.
        cfg = CMDLConfig.from_domain(
            "synthetic",
            max_lag=6,
            n_entities=8,
            seq_length=12,
            n_proxies=3,
            seq_features=1,
        )
        panel = generate_cmdl_synthetic(cfg)
        encoder = AdaptiveACEncoder(n_proxies=cfg.n_proxies)
        gate = ScaleInvariantLagGate(
            max_lag=cfg.max_lag,
            d_model=cfg.seq_features,
            temperature=cfg.temperature,
            lag_bias_strength=cfg.lag_bias_strength,
        )

        proxy_batch = panel.p_i[:4]
        lagged_sequence = panel.X_it[:4, : cfg.max_lag, :]

    # 先由 proxy 得到 z_i，再让 Lag Gate 基于 z_i 生成 omega 和 lag_context。
    # First derive z_i from proxies, then let the lag gate generate omega and lag_context from z_i.
        encoder_output = encoder(proxy_batch)
        gate_output = gate(encoder_output.z_i, lagged_sequence)

    # 这个损失不是训练目标本身，只是用于验证整个 Step 2 计算图可反向传播。
    # This is not the final training loss; it is only used to confirm the Step 2 graph is differentiable end-to-end.
        loss = gate_output.lag_context.pow(2).mean() + encoder_output.p_hat_i.pow(2).mean()
        loss.backward()

        self.assertEqual(tuple(gate_output.omega.shape), (4, cfg.max_lag))
        self.assertEqual(tuple(gate_output.lag_context.shape), (4, cfg.seq_features))
        self.assertGreaterEqual(gate_output.k_star.min().item(), 1.0)
        self.assertLessEqual(gate_output.k_star.max().item(), float(cfg.max_lag))
        self.assertTrue(any(parameter.grad is not None for parameter in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()