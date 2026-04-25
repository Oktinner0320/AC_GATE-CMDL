"""Tests for shared real-data diagnostics and forecast calibration helpers."""

from types import SimpleNamespace
import unittest

import torch

from baselines.panel_ols import add_forecast_calibration, evaluate_grouped_ardl_baseline
from evaluation.realdata_diagnostics import build_realdata_diagnostics, proxy_metadata_payload


class RealDataDiagnosticsTest(unittest.TestCase):
    def test_sign_adjusted_anchor_and_per_proxy_metrics(self) -> None:
        metadata = {
            "proxy_columns": ["proxy_anchor", "proxy_aux"],
            "anchor_proxy_name": "proxy_anchor",
            "anchor_proxy_index": 0,
            "anchor_expected_sign": -1.0,
            "proxy_expected_signs": [-1.0, -1.0],
        }
        kstar = torch.tensor([1.0, 2.0, 3.0, 4.0])
        proxies = torch.tensor(
            [
                [4.0, 10.0],
                [3.0, 20.0],
                [2.0, 30.0],
                [1.0, 40.0],
            ]
        )
        metrics = build_realdata_diagnostics(
            effective_kstar=kstar,
            proxies=proxies,
            metadata=metadata,
            prefix="kstar",
            proxy_predictions=proxies,
        )

        self.assertAlmostEqual(metrics["kstar_proxy_spearman_rho"], -1.0)
        self.assertAlmostEqual(metrics["kstar_proxy_spearman_adjusted_rho"], 1.0)
        self.assertAlmostEqual(metrics["kstar_proxy_2_spearman_rho"], 1.0)
        self.assertAlmostEqual(metrics["proxy_anchor_recon_r2"], 1.0)
        self.assertIn("z_anchor_adjusted_rho", build_realdata_diagnostics(kstar, proxies, metadata, z_values=kstar))

        payload = proxy_metadata_payload(metadata, n_proxies=2)
        self.assertEqual(payload["anchor_proxy_name"], "proxy_anchor")
        self.assertEqual(payload["auxiliary_proxy_names"], ["proxy_aux"])

    def test_forecast_calibration_adds_simple_baselines(self) -> None:
        panel = SimpleNamespace(
            X_it=torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6, 1),
            p_i=torch.tensor([[0.1], [0.9]], dtype=torch.float32),
            s_i=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            Y_it=torch.tensor(
                [
                    [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                    [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
                ],
                dtype=torch.float32,
            ),
        )
        enriched = add_forecast_calibration({"r2": 0.25}, panel, panel, max_lag=2)

        self.assertIn("baseline_persistence_r2", enriched)
        self.assertIn("baseline_panel_ols_r2", enriched)
        self.assertIn("baseline_grouped_ardl_r2", enriched)
        self.assertIn("target_variance", enriched)
        self.assertIn("r2_delta_vs_persistence", enriched)
        self.assertIn("r2_delta_vs_grouped_ardl", enriched)

    def test_grouped_ardl_baseline_reports_lag_summary(self) -> None:
        panel = SimpleNamespace(
            X_it=torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8, 1),
            p_i=torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
            s_i=torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32),
            Y_it=torch.tensor(
                [
                    [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                    [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
                    [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7],
                ],
                dtype=torch.float32,
            ),
            metadata={"anchor_proxy_index": 0},
        )

        metrics = evaluate_grouped_ardl_baseline(panel, panel, max_lag=2)

        self.assertIn("baseline_grouped_ardl_r2", metrics)
        self.assertIn("baseline_grouped_ardl_group_count", metrics)
        self.assertGreaterEqual(metrics["baseline_grouped_ardl_group_count"], 1.0)


if __name__ == "__main__":
    unittest.main()