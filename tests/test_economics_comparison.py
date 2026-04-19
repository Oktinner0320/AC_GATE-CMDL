"""Regression tests for unified economics comparison tables.

覆盖 economics CMDL、plain LSTM 与 ablation summary 的统一读取和指标归一化逻辑。
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


from evaluation.economics_comparison import (  # noqa: E402
    build_economics_comparison,
    build_interpretability_table,
    build_task_table,
)


class EconomicsComparisonTest(unittest.TestCase):
    """Validate unified economics comparison loading and normalization.

    验证统一 economics comparison 工具的读取与指标对齐行为。
    """

    @staticmethod
    def _build_data_payload() -> dict[str, object]:
        return {
            "source_path": "C:/tmp/economics_cleaned_long.csv",
            "target_column": "ctfp",
            "stats_end_year": 2007,
            "year_start": 1980,
            "year_end": 2023,
            "train_end_year": 2007,
            "val_end_year": 2013,
            "n_entities": 105,
            "full_seq_length": 44,
            "train_years": list(range(1980, 2008)),
            "val_years": list(range(1998, 2014)),
            "test_years": list(range(2004, 2024)),
        }

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
                "E4_economics_cmdl_seed0",
                {
                    "experiment": "E4_economics_cmdl_seed0",
                    "tracking_backend": "json",
                    "device": "cpu",
                    "best_epoch": 12,
                    "best_val_task_loss": 0.11,
                    "config": {
                        "domain": "economics",
                        "seed": 0,
                        "max_lag": 10,
                        "d_model": 64,
                        "n_entities": 105,
                        "seq_length": 44,
                        "seq_features": 1,
                        "n_proxies": 1,
                        "static_dim": 2,
                        "lambda_r": 0.1,
                        "temperature": 1.0,
                        "lag_bias_strength": 1.0,
                        "lstm_layers": 2,
                        "dropout": 0.05,
                        "noise_std": 0.1,
                        "scenario": "linear",
                    },
                    "data": self._build_data_payload(),
                    "metrics": {
                        "train": {"task_loss": 0.20, "mse": 0.20, "mae": 0.30, "r2": 0.60},
                        "val": {"task_loss": 0.11, "mse": 0.11, "mae": 0.21, "r2": 0.72},
                        "test": {
                            "total_loss": 0.14,
                            "task_loss": 0.13,
                            "recon_loss": 0.01,
                            "mse": 0.13,
                            "mae": 0.24,
                            "r2": 0.70,
                            "proxy_recon_r2": 0.91,
                            "kstar_proxy_spearman_rho": 0.85,
                            "kstar_proxy_spearman_p": 0.01,
                            "kstar_mean": 5.7,
                            "kstar_std": 1.2,
                            "omega_entropy_mean": 1.7,
                        },
                    },
                },
            )
            self._write_summary(
                baseline_root,
                "E4_economics_lstm_seed0",
                {
                    "experiment": "E4_economics_lstm_seed0",
                    "model": "plain_lstm",
                    "tracking_backend": "json",
                    "device": "cpu",
                    "best_epoch": 9,
                    "best_val_task_loss": 0.18,
                    "posthoc_lag_method": "lag_occlusion",
                    "config": {
                        "domain": "economics",
                        "seed": 0,
                        "max_lag": 10,
                        "d_model": 64,
                        "n_entities": 105,
                        "seq_length": 44,
                        "seq_features": 1,
                        "n_proxies": 1,
                        "static_dim": 2,
                        "lambda_r": 0.1,
                        "temperature": 1.0,
                        "lag_bias_strength": 1.0,
                        "lstm_layers": 2,
                        "dropout": 0.05,
                        "noise_std": 0.1,
                        "scenario": "linear",
                    },
                    "data": self._build_data_payload(),
                    "metrics": {
                        "train": {"task_loss": 0.25, "mse": 0.25, "mae": 0.33, "r2": 0.55},
                        "val": {"task_loss": 0.18, "mse": 0.18, "mae": 0.27, "r2": 0.63},
                        "test": {
                            "task_loss": 0.17,
                            "mse": 0.17,
                            "mae": 0.29,
                            "r2": 0.66,
                            "posthoc_kstar_proxy_spearman_rho": 0.42,
                            "posthoc_kstar_proxy_spearman_p": 0.05,
                            "posthoc_kstar_mean": 6.3,
                            "posthoc_kstar_std": 1.7,
                            "lag_profile_entropy_mean": 2.2,
                        },
                    },
                },
            )
            self._write_summary(
                ablation_root,
                "E4_economics_ablation_no_ac_encoder_seed0",
                {
                    "experiment": "E4_economics_ablation_no_ac_encoder_seed0",
                    "variant": "no_ac_encoder",
                    "tracking_backend": "json",
                    "device": "cpu",
                    "best_epoch": 7,
                    "best_val_task_loss": 0.31,
                    "effective_lambda_r": 0.0,
                    "config": {
                        "domain": "economics",
                        "seed": 0,
                        "max_lag": 10,
                        "d_model": 64,
                        "n_entities": 105,
                        "seq_length": 44,
                        "seq_features": 1,
                        "n_proxies": 1,
                        "static_dim": 2,
                        "lambda_r": 0.1,
                        "temperature": 1.0,
                        "lag_bias_strength": 1.0,
                        "lstm_layers": 2,
                        "dropout": 0.05,
                        "noise_std": 0.1,
                        "scenario": "linear",
                    },
                    "data": self._build_data_payload(),
                    "diagnostics": {
                        "proxy_refit": {
                            "status": "skipped_rank_deficient",
                            "applied": False,
                            "metrics_interpretable": False,
                            "reason": "rank_deficient_design",
                            "design_rank": 1,
                            "design_columns": 2,
                            "latent_std": 0.0,
                            "proxy_std": 0.12,
                        }
                    },
                    "metrics": {
                        "train": {"task_loss": 0.40, "mse": 0.40, "mae": 0.47, "r2": 0.30},
                        "val": {"task_loss": 0.31, "mse": 0.31, "mae": 0.39, "r2": 0.21},
                        "test": {
                            "total_loss": 0.35,
                            "task_loss": 0.35,
                            "recon_loss": 0.08,
                            "mse": 0.35,
                            "mae": 0.41,
                            "r2": 0.18,
                            "proxy_recon_r2": float("nan"),
                            "proxy_metric_valid": 0.0,
                            "kstar_proxy_spearman_rho": float("nan"),
                            "kstar_proxy_spearman_p": float("nan"),
                            "kstar_mean": 5.0,
                            "kstar_std": 0.0,
                            "kstar_proxy_metric_valid": 0.0,
                            "omega_entropy_mean": 2.1,
                        },
                    },
                },
            )

            comparison = build_economics_comparison(
                cmdl_root=cmdl_root,
                baseline_root=baseline_root,
                ablation_root=ablation_root,
            )

            self.assertEqual(len(comparison), 3)
            self.assertTrue({"test_r2", "test_effective_kstar_proxy_spearman_rho"}.issubset(comparison.columns))

            cmdl_row = comparison.loc[comparison["family"] == "cmdl"].iloc[0]
            self.assertEqual(cmdl_row["display_name"], "CMDL")
            self.assertEqual(cmdl_row["lag_method"], "learned_omega")
            self.assertAlmostEqual(cmdl_row["test_proxy_signal_r2"], 0.91)

            baseline_row = comparison.loc[comparison["family"] == "plain_lstm"].iloc[0]
            self.assertEqual(baseline_row["display_name"], "Plain LSTM")
            self.assertEqual(baseline_row["lag_method"], "lag_occlusion")
            self.assertTrue(pd.isna(baseline_row["test_proxy_signal_r2"]))
            self.assertAlmostEqual(baseline_row["test_effective_kstar_proxy_spearman_rho"], 0.42)
            self.assertAlmostEqual(baseline_row["test_effective_lag_entropy_mean"], 2.2)

            ablation_row = comparison.loc[comparison["family"] == "ablation"].iloc[0]
            self.assertEqual(ablation_row["display_name"], "No AC Encoder")
            self.assertEqual(ablation_row["variant"], "no_ac_encoder")
            self.assertAlmostEqual(ablation_row["effective_lambda_r"], 0.0)
            self.assertEqual(ablation_row["proxy_refit_status"], "skipped_rank_deficient")
            self.assertFalse(bool(ablation_row["proxy_metric_interpretable"]))
            self.assertTrue(pd.isna(ablation_row["test_proxy_signal_r2"]))
            self.assertTrue(pd.isna(ablation_row["test_effective_kstar_proxy_spearman_rho"]))

            task_table = build_task_table(comparison)
            interpretability_table = build_interpretability_table(comparison)

            self.assertEqual(task_table.iloc[0]["display_name"], "CMDL")
            self.assertEqual(len(task_table), 3)
            self.assertEqual(len(interpretability_table), 3)
            self.assertEqual(interpretability_table.iloc[0]["display_name"], "CMDL")

    def test_empty_roots_return_empty_tables_with_stable_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            empty_root = Path(temporary_dir) / "missing"
            comparison = build_economics_comparison(cmdl_root=empty_root)
            task_table = build_task_table(comparison)
            interpretability_table = build_interpretability_table(comparison)

            self.assertTrue(comparison.empty)
            self.assertEqual(
                task_table.columns.tolist(),
                [
                    "display_name",
                    "family",
                    "variant",
                    "seed",
                    "experiment",
                    "target_column",
                    "best_epoch",
                    "best_val_task_loss",
                    "test_r2",
                    "test_mae",
                    "test_mse",
                ],
            )
            self.assertEqual(
                interpretability_table.columns.tolist(),
                [
                    "display_name",
                    "family",
                    "variant",
                    "seed",
                    "experiment",
                    "target_column",
                    "lag_method",
                    "proxy_refit_status",
                    "proxy_metric_interpretable",
                    "test_effective_kstar_proxy_spearman_rho",
                    "test_effective_kstar_proxy_metric_valid",
                    "test_effective_kstar_mean",
                    "test_effective_kstar_std",
                    "test_effective_lag_entropy_mean",
                    "test_proxy_signal_r2",
                    "test_proxy_signal_metric_valid",
                ],
            )


if __name__ == "__main__":
    unittest.main()