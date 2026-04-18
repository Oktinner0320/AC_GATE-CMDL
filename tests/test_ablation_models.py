"""Regression tests for the synthetic ablation model variants.

覆盖 no_ac_encoder 和 uniform_lag 两个变体的核心行为约束。
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from config.cmdl_config import CMDLConfig
from data.synthetic.generate import generate_cmdl_synthetic
from experiments.run_ablation import NoACEncoderCMDLModel, UniformLagCMDLModel


class AblationModelTest(unittest.TestCase):
    """Validate the most important contracts of the ablation variants.

    验证 ablation 变体的关键 shape 和分布约束。
    """

    @staticmethod
    def _build_small_panel() -> tuple[CMDLConfig, object]:
        cfg = CMDLConfig.from_domain(
            "synthetic",
            max_lag=4,
            n_entities=8,
            seq_length=12,
            d_model=16,
            dropout=0.0,
        )
        panel = generate_cmdl_synthetic(cfg)
        return cfg, panel

    def test_no_ac_encoder_shares_z_and_omega(self) -> None:
        cfg, panel = self._build_small_panel()
        model = NoACEncoderCMDLModel(cfg)

        output = model(
            entity_ids=panel.entity_ids,
            X_it=panel.X_it,
            p_i=panel.p_i,
            s_i=panel.s_i,
        )

        self.assertEqual(tuple(output.z_i.shape), (cfg.n_entities, 1))
        self.assertEqual(tuple(output.omega.shape), (cfg.n_entities, cfg.max_lag))
        self.assertTrue(torch.allclose(output.z_i, output.z_i[:1].expand_as(output.z_i)))
        self.assertTrue(torch.allclose(output.omega, output.omega[:1].expand_as(output.omega)))

    def test_uniform_lag_outputs_uniform_weights(self) -> None:
        cfg, panel = self._build_small_panel()
        model = UniformLagCMDLModel(cfg)

        output = model(
            entity_ids=panel.entity_ids,
            X_it=panel.X_it,
            p_i=panel.p_i,
            s_i=panel.s_i,
        )

        expected = torch.full((cfg.n_entities, cfg.max_lag), 1.0 / cfg.max_lag)
        torch.testing.assert_close(output.omega, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            output.k_star,
            torch.full((cfg.n_entities,), (cfg.max_lag + 1) / 2.0),
            atol=1e-6,
            rtol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()