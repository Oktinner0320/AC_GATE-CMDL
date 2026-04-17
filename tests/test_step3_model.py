"""Step 3 的 backbone、整模和损失模块测试。
Step 3 tests for the backbone, assembled model, and loss module.
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    # 直接运行测试文件时，手动把工作区根目录加回 import 路径。
    # When running the file directly, add the workspace root back to the import path.
    sys.path.insert(0, WORKSPACE_ROOT)

# Windows + PyTorch 在部分环境下会触发重复 OpenMP runtime 报错；测试阶段允许继续。
# Some Windows + PyTorch setups hit duplicate OpenMP runtime warnings; allow tests to continue.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from config.cmdl_config import CMDLConfig
from data.synthetic.generate import generate_cmdl_synthetic
from model.backbone import UniversalPanelBackbone
from model.cmdl_model import CMDLModel
from model.loss import DomainAgnosticLoss


class UniversalPanelBackboneTest(unittest.TestCase):
    """验证 Step 3 backbone 的 shape 契约。
    Validate shape contracts of the Step 3 backbone.
    """

    def test_backbone_returns_expected_shapes(self) -> None:
        backbone = UniversalPanelBackbone(d_model=16, static_dim=2, lstm_layers=2, dropout=0.0)
        current_sequence = torch.randn(4, 6, 16)
        lag_context_sequence = torch.randn(4, 6, 16)
        entity_embedding = torch.randn(4, 16)
        static_features = torch.randn(4, 2)
        z_i = torch.randn(4, 1)

        output = backbone(
            current_sequence=current_sequence,
            lag_context_sequence=lag_context_sequence,
            entity_embedding=entity_embedding,
            static_features=static_features,
            z_i=z_i,
        )

        self.assertEqual(tuple(output.sequence.shape), (4, 6, 16))
        self.assertEqual(tuple(output.final_state.shape), (4, 16))


class CMDLModelIntegrationTest(unittest.TestCase):
    """验证 Step 3 前向、loss 对齐和短程优化行为。
    Validate Step 3 forward, loss alignment, and short optimization behavior.
    """

    @staticmethod
    def _build_small_batch() -> tuple[CMDLConfig, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 小规模 synthetic batch 用于快速回归，不追求统计意义，只验证接口与梯度链路。
        # A small synthetic batch keeps regression tests fast; the goal is interface and gradient validation.
        cfg = CMDLConfig.from_domain(
            "synthetic",
            max_lag=4,
            n_entities=8,
            seq_length=12,
            d_model=16,
            dropout=0.0,
        )
        panel = generate_cmdl_synthetic(cfg)
        entity_ids = panel.entity_ids[:4]
        X_it = panel.X_it[:4]
        p_i = panel.p_i[:4]
        s_i = panel.s_i[:4]
        y_true = panel.Y_it[:4]
        return cfg, entity_ids, X_it, p_i, s_i, y_true

    def test_model_forward_returns_expected_shapes(self) -> None:
        cfg, entity_ids, X_it, p_i, s_i, _ = self._build_small_batch()
        model = CMDLModel(cfg)

        output = model(entity_ids=entity_ids, X_it=X_it, p_i=p_i, s_i=s_i)

        # 这里显式检查 warm-up 截断后的时间长度，防止未来改动破坏对齐逻辑。
        # Explicitly check the post-warm-up time length so future changes do not break alignment.
        self.assertEqual(tuple(output.y_pred.shape), (4, cfg.seq_length - cfg.max_lag))
        self.assertEqual(tuple(output.omega.shape), (4, cfg.max_lag))
        self.assertEqual(tuple(output.z_i.shape), (4, 1))
        self.assertEqual(tuple(output.p_hat_i.shape), (4, cfg.n_proxies))
        self.assertEqual(tuple(output.k_star.shape), (4,))
        self.assertEqual(tuple(output.lag_context_sequence.shape), (4, cfg.seq_length - cfg.max_lag, cfg.d_model))
        torch.testing.assert_close(output.omega.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_loss_aligns_with_warmup_steps(self) -> None:
        cfg, entity_ids, X_it, p_i, s_i, y_true = self._build_small_batch()
        model = CMDLModel(cfg)
        criterion = DomainAgnosticLoss(lambda_r=cfg.lambda_r, warmup_steps=cfg.max_lag)

        output = model(entity_ids=entity_ids, X_it=X_it, p_i=p_i, s_i=s_i)
        losses = criterion(output.y_pred, y_true, output.p_hat_i, p_i)

        self.assertEqual(losses.total_loss.ndim, 0)
        self.assertEqual(losses.task_loss.ndim, 0)
        self.assertEqual(losses.recon_loss.ndim, 0)
        self.assertGreaterEqual(losses.total_loss.item(), losses.task_loss.item())

    def test_end_to_end_optimization_reduces_loss(self) -> None:
        torch.manual_seed(7)
        cfg, entity_ids, X_it, p_i, s_i, y_true = self._build_small_batch()
        model = CMDLModel(cfg)
        criterion = DomainAgnosticLoss(lambda_r=cfg.lambda_r, warmup_steps=cfg.max_lag)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        loss_values = []
        for _ in range(6):
            # 这里用极短的优化回合确认整条计算图既能反传也能朝正确方向更新。
            # A very short optimization loop is enough to verify both backpropagation and update direction.
            optimizer.zero_grad()
            output = model(entity_ids=entity_ids, X_it=X_it, p_i=p_i, s_i=s_i)
            losses = criterion(output.y_pred, y_true, output.p_hat_i, p_i)
            loss_values.append(losses.total_loss.item())
            losses.total_loss.backward()
            optimizer.step()

        self.assertLess(loss_values[-1], loss_values[0])


if __name__ == "__main__":
    unittest.main()