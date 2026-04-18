"""Regression tests for unified synthetic comparison tables.

覆盖 CMDL、plain LSTM 与 ablation summary 的统一读取和指标归一化逻辑。
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


from evaluation.synthetic_comparison import (
    build_identification_table,
    build_recovery_table,
    build_synthetic_comparison,
)


class SyntheticComparisonTest(unittest.TestCase):
    """Validate unified synthetic comparison loading and normalization.

    验证统一 synthetic comparison 工具的读取与指标对齐行为。
    """

    @staticmethod
    def _write_summary(root: Path, run_name: str, payload: dict) -> None:
        run_dir = root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_comparison_tables_normalize_family_specific_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            cmdl_root = temporary_root / "cmdl"
            baseline_root = temporary_root / "baseline"
            ablation_root = temporary_root / "ablation"

            self._write_summary(
                cmdl_root,
                "E1a_linear",
                {
                    "experiment": "E1a_linear",
                    "scenario": "linear",
                    "tracking_backend": "json",
                    "best_epoch": 12,
                    "best_val_task_loss": 0.11,
                    "metrics": {
                        "task_loss": 0.08,
                        "recon_loss": 0.01,
                        "kstar_mae": 0.9,
                        "kstar_spearman_rho": 0.95,
                        "proxy_recon_r2": 0.91,
                        "z_spearman_rho": 0.97,
                        "omega_entropy_mean": 1.8,
                        "omega_peak_accuracy": 0.4,
                    },
                },
            )
            self._write_summary(
                cmdl_root,
                "E1b_identification",
                {
                    "experiment": "E1b_identification",
                    "scenario": "linear",
                    "metrics": {
                        "proxy_recon_r2": 0.93,
                        "z_spearman_rho": 0.98,
                    },
                },
            )
            self._write_summary(
                baseline_root,
                "LSTM_linear",
                {
                    "experiment": "LSTM_linear",
                    "model": "plain_lstm",
                    "scenario": "linear",
                    "tracking_backend": "json",
                    "best_epoch": 3,
                    "best_val_task_loss": 0.25,
                    "metrics": {
                        "task_loss": 0.23,
                        "posthoc_kstar_mae": 2.2,
                        "posthoc_kstar_spearman_rho": 0.06,
                        "posthoc_profile_entropy_mean": 2.0,
                        "posthoc_profile_peak_accuracy": 0.02,
                    },
                },
            )
            self._write_summary(
                ablation_root,
                "no_ac_encoder_linear_seed42",
                {
                    "experiment": "no_ac_encoder_linear_seed42",
                    "variant": "no_ac_encoder",
                    "scenario": "linear",
                    "tracking_backend": "json",
                    "best_epoch": 5,
                    "best_val_task_loss": 0.4,
                    "metrics": {
                        "task_loss": 0.35,
                        "kstar_mae": 3.1,
                        "kstar_spearman_rho": 0.12,
                        "proxy_recon_r2": -0.2,
                        "z_spearman_rho": 0.0,
                        "omega_entropy_mean": 2.1,
                        "omega_peak_accuracy": 0.1,
                    },
                },
            )

            comparison = build_synthetic_comparison(
                cmdl_root=cmdl_root,
                baseline_root=baseline_root,
                ablation_root=ablation_root,
            )
            self.assertEqual(len(comparison), 4)

            baseline_row = comparison.loc[comparison["experiment"] == "LSTM_linear"].iloc[0]
            self.assertEqual(baseline_row["display_name"], "Plain LSTM")
            self.assertAlmostEqual(baseline_row["effective_kstar_mae"], 2.2)
            self.assertAlmostEqual(baseline_row["effective_lag_peak_accuracy"], 0.02)

            cmdl_row = comparison.loc[comparison["experiment"] == "E1a_linear"].iloc[0]
            self.assertEqual(cmdl_row["display_name"], "CMDL")
            self.assertAlmostEqual(cmdl_row["effective_kstar_mae"], 0.9)
            self.assertAlmostEqual(cmdl_row["proxy_signal_r2"], 0.91)

            ablation_row = comparison.loc[comparison["experiment"] == "no_ac_encoder_linear_seed42"].iloc[0]
            self.assertEqual(ablation_row["display_name"], "No AC Encoder")
            self.assertAlmostEqual(ablation_row["effective_kstar_spearman_rho"], 0.12)

            recovery_table = build_recovery_table(comparison)
            identification_table = build_identification_table(comparison)

            self.assertEqual(len(recovery_table), 3)
            self.assertNotIn("Plain LSTM", identification_table["display_name"].tolist())
            self.assertIn("CMDL", identification_table["display_name"].tolist())
            self.assertIn("No AC Encoder", identification_table["display_name"].tolist())


if __name__ == "__main__":
    unittest.main()