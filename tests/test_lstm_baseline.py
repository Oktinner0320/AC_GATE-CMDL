"""Regression tests for the synthetic plain-LSTM baseline.

覆盖 baseline 的前向 shape 契约，以及 post-hoc lag profile 的归一化行为。
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

from baselines.lstm_baseline import PlainLSTMBaseline
from config.cmdl_config import CMDLConfig
from data.synthetic.generate import generate_cmdl_synthetic
from experiments.run_lstm_baseline import compute_posthoc_lag_profile


class PlainLSTMBaselineTest(unittest.TestCase):
    """Validate forward contracts and post-hoc lag diagnostics.

    验证 baseline 的前向输出契约和 post-hoc lag 解释结果。
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

    def test_forward_returns_expected_shapes(self) -> None:
        cfg, panel = self._build_small_panel()
        model = PlainLSTMBaseline(cfg)

        output = model(entity_ids=panel.entity_ids[:4], X_it=panel.X_it[:4], s_i=panel.s_i[:4])

        self.assertEqual(tuple(output.y_pred.shape), (4, cfg.seq_length - cfg.max_lag))
        self.assertEqual(tuple(output.sequence.shape), (4, cfg.seq_length - cfg.max_lag, cfg.d_model))
        self.assertEqual(tuple(output.final_state.shape), (4, cfg.d_model))

    def test_posthoc_lag_profile_is_normalized(self) -> None:
        torch.manual_seed(7)
        cfg, panel = self._build_small_panel()
        model = PlainLSTMBaseline(cfg)

        lag_profile, pseudo_k_star = compute_posthoc_lag_profile(model, panel)

        self.assertEqual(tuple(lag_profile.shape), (cfg.n_entities, cfg.max_lag))
        self.assertEqual(tuple(pseudo_k_star.shape), (cfg.n_entities,))
        torch.testing.assert_close(
            lag_profile.sum(dim=-1),
            torch.ones(cfg.n_entities),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertTrue(torch.all(pseudo_k_star >= 1.0).item())
        self.assertTrue(torch.all(pseudo_k_star <= float(cfg.max_lag)).item())


if __name__ == "__main__":
    unittest.main()