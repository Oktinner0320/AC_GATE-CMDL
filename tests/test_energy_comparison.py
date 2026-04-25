"""Regression tests for unified energy comparison tables.

覆盖 energy CMDL、plain LSTM 与 ablation summary 的统一读取和指标归一化逻辑。
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


from evaluation.energy_comparison import (  # noqa: E402
    build_energy_comparison,
    build_interpretability_table,
    build_mechanism_result_log,
    build_mechanism_summary_table,
    build_per_proxy_audit_summary_table,
    build_per_proxy_alignment_table,
    build_task_table,
)


class EnergyComparisonTest(unittest.TestCase):
    """Validate unified energy comparison loading and normalization."""

    @staticmethod
    def _build_data_payload() -> dict[str, object]:
        return {
            "source_path": "C:/tmp/energy_wgi.csv",
            "treatment_column": "renewables_share_energy",
            "target_column": "co2_per_unit_energy",
            "feature_bundle": "minimal",
            "seq_feature_columns": ["x_t"],
            "proxy_columns": [
                "proxy_government_effectiveness",
                "proxy_regulatory_quality",
                "proxy_rule_of_law",
            ],
            "anchor_proxy_name": "proxy_government_effectiveness",
            "anchor_proxy_index": 0,
            "anchor_expected_sign": -1.0,
            "auxiliary_proxy_names": ["proxy_regulatory_quality", "proxy_rule_of_law"],
            "proxy_aggregate_name": "mean_wgi_proxy",
            "static_columns": ["static_log_population", "static_log_gdp_per_capita"],
            "stats_end_year": 2011,
            "year_start": 1996,
            "year_end": 2023,
            "train_end_year": 2011,
            "val_end_year": 2017,
            "n_entities": 88,
            "full_seq_length": 28,
            "train_years": list(range(1996, 2012)),
            "val_years": list(range(2002, 2018)),
            "test_years": list(range(2008, 2024)),
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
                "E3_energy_cmdl_seed0",
                {
                    "experiment": "E3_energy_cmdl_seed0",
                    "tracking_backend": "json",
                    "device": "cpu",
                    "best_epoch": 12,
                    "best_val_task_loss": 0.11,
                    "config": {
                        "domain": "energy",
                        "seed": 0,
                        "max_lag": 10,
                        "d_model": 64,
                        "n_entities": 88,
                        "seq_length": 28,
                        "seq_features": 1,
                        "n_proxies": 3,
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
                            "baseline_persistence_r2": 0.60,
                            "baseline_panel_ols_r2": 0.68,
                            "baseline_grouped_ardl_r2": 0.69,
                            "baseline_best_simple_r2": 0.68,
                            "r2_delta_vs_persistence": 0.10,
                            "r2_delta_vs_panel_ols": 0.02,
                            "r2_delta_vs_grouped_ardl": 0.01,
                            "proxy_recon_r2": 0.91,
                            "proxy_metric_valid": 1.0,
                            "kstar_proxy_spearman_rho": 0.85,
                            "kstar_proxy_spearman_p": 0.01,
                            "kstar_proxy_spearman_adjusted_rho": 0.85,
                            "kstar_proxy_mean_spearman_adjusted_rho": 0.84,
                            "kstar_proxy_government_effectiveness_spearman_adjusted_rho": 0.86,
                            "kstar_proxy_regulatory_quality_spearman_adjusted_rho": 0.83,
                            "kstar_proxy_rule_of_law_spearman_adjusted_rho": 0.82,
                            "kstar_mean": 5.7,
                            "kstar_std": 1.2,
                            "kstar_proxy_metric_valid": 1.0,
                            "omega_entropy_mean": 1.7,
                            "omega_top1_share": 0.40,
                            "z_std": 0.30,
                            "z_proxy_spearman_adjusted_rho": 0.70,
                            "z_anchor_adjusted_rho": 0.71,
                            "omega_entropy_penalty": 0.02,
                            "omega_entropy_band_violation_share": 0.25,
                            "z_anchor_loss": 0.0,
                            "lag_gate_sensitivity_range": 0.50,
                        },
                    },
                },
            )
            self._write_summary(
                baseline_root,
                "E3_energy_lstm_seed0",
                {
                    "experiment": "E3_energy_lstm_seed0",
                    "model": "plain_lstm",
                    "tracking_backend": "json",
                    "device": "cpu",
                    "best_epoch": 9,
                    "best_val_task_loss": 0.18,
                    "posthoc_lag_method": "lag_occlusion",
                    "config": {
                        "domain": "energy",
                        "seed": 0,
                        "max_lag": 10,
                        "d_model": 64,
                        "n_entities": 88,
                        "seq_length": 28,
                        "seq_features": 1,
                        "n_proxies": 3,
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
                            "baseline_persistence_r2": 0.60,
                            "baseline_panel_ols_r2": 0.68,
                            "baseline_grouped_ardl_r2": 0.69,
                            "baseline_best_simple_r2": 0.68,
                            "r2_delta_vs_persistence": 0.06,
                            "r2_delta_vs_panel_ols": -0.02,
                            "r2_delta_vs_grouped_ardl": -0.03,
                            "posthoc_kstar_proxy_spearman_rho": 0.42,
                            "posthoc_kstar_proxy_spearman_p": 0.05,
                            "posthoc_kstar_mean": 6.3,
                            "posthoc_kstar_std": 1.7,
                            "lag_profile_entropy_mean": 2.2,
                            "kstar_proxy_metric_valid": 1.0,
                        },
                    },
                },
            )
            self._write_summary(
                ablation_root,
                "E3_energy_ablation_no_ac_encoder_seed0",
                {
                    "experiment": "E3_energy_ablation_no_ac_encoder_seed0",
                    "variant": "no_ac_encoder",
                    "tracking_backend": "json",
                    "device": "cpu",
                    "best_epoch": 7,
                    "best_val_task_loss": 0.31,
                    "effective_lambda_r": 0.0,
                    "config": {
                        "domain": "energy",
                        "seed": 0,
                        "max_lag": 10,
                        "d_model": 64,
                        "n_entities": 88,
                        "seq_length": 28,
                        "seq_features": 1,
                        "n_proxies": 3,
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
                        "ablation": {
                            "same_architecture_as_full_cmdl": False,
                            "matched_init_to_full_cmdl": False,
                            "causal_ablation_validity": "architecture_changed",
                        },
                        "proxy_refit": {
                            "status": "skipped_rank_deficient",
                            "applied": False,
                            "metrics_interpretable": False,
                            "reason": "rank_deficient_design",
                            "design_rank": 1,
                            "design_columns": 2,
                            "latent_std": 0.0,
                            "proxy_std": 0.12,
                        },
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
                            "baseline_persistence_r2": 0.60,
                            "baseline_panel_ols_r2": 0.68,
                            "baseline_grouped_ardl_r2": 0.69,
                            "baseline_best_simple_r2": 0.68,
                            "r2_delta_vs_persistence": -0.42,
                            "r2_delta_vs_panel_ols": -0.50,
                            "r2_delta_vs_grouped_ardl": -0.51,
                            "proxy_recon_r2": float("nan"),
                            "proxy_metric_valid": 0.0,
                            "kstar_proxy_spearman_rho": float("nan"),
                            "kstar_proxy_spearman_p": float("nan"),
                            "kstar_mean": 5.0,
                            "kstar_std": 0.0,
                            "kstar_proxy_metric_valid": 0.0,
                            "omega_entropy_mean": 2.1,
                            "omega_top1_share": 1.0,
                        },
                    },
                },
            )

            comparison = build_energy_comparison(
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
            self.assertEqual(cmdl_row["feature_bundle"], "minimal")

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
            self.assertFalse(bool(ablation_row["matched_init_to_full_cmdl"]))
            self.assertEqual(ablation_row["causal_ablation_validity"], "architecture_changed")

            task_table = build_task_table(comparison)
            interpretability_table = build_interpretability_table(comparison)

            self.assertEqual(task_table.iloc[0]["display_name"], "CMDL")
            self.assertEqual(len(task_table), 3)
            self.assertEqual(len(interpretability_table), 3)
            self.assertEqual(interpretability_table.iloc[0]["display_name"], "CMDL")

            per_proxy_table = build_per_proxy_alignment_table(comparison)
            per_proxy_audit_summary = build_per_proxy_audit_summary_table(comparison)
            mechanism_summary = build_mechanism_summary_table(comparison)
            result_log = build_mechanism_result_log(comparison)

            self.assertEqual(len(per_proxy_table), 3)
            self.assertEqual(len(per_proxy_audit_summary), 3)
            self.assertEqual(
                per_proxy_table["proxy_name"].tolist(),
                ["government_effectiveness", "regulatory_quality", "rule_of_law"],
            )
            self.assertAlmostEqual(float(per_proxy_table["adjusted_rho"].min()), 0.82)
            self.assertTrue((per_proxy_audit_summary["positive_seed_share"] == 1.0).all())
            self.assertIn("test_lag_gate_sensitivity_range_mean", mechanism_summary.columns)
            answers = dict(zip(result_log["layer"], result_log["answer"]))
            self.assertEqual(answers["forecast_calibration"], "yes")
            self.assertEqual(answers["simple_baseline_calibration"], "yes")
            self.assertEqual(answers["ac_gate_per_proxy"], "yes")

    def test_empty_roots_return_empty_tables_with_stable_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            empty_root = Path(temporary_dir) / "missing"
            comparison = build_energy_comparison(cmdl_root=empty_root)
            task_table = build_task_table(comparison)
            interpretability_table = build_interpretability_table(comparison)
            per_proxy_table = build_per_proxy_alignment_table(comparison)
            per_proxy_audit_summary = build_per_proxy_audit_summary_table(comparison)
            mechanism_summary = build_mechanism_summary_table(comparison)
            result_log = build_mechanism_result_log(comparison)

            self.assertTrue(comparison.empty)
            self.assertTrue(per_proxy_table.empty)
            self.assertTrue(per_proxy_audit_summary.empty)
            self.assertTrue(mechanism_summary.empty)
            self.assertTrue(result_log.empty)
            self.assertEqual(
                task_table.columns.tolist(),
                [
                    "display_name",
                    "family",
                    "variant",
                    "seed",
                    "experiment",
                    "target_column",
                    "feature_bundle",
                    "best_epoch",
                    "best_val_task_loss",
                    "test_r2",
                    "test_baseline_persistence_r2",
                    "test_baseline_panel_ols_r2",
                    "test_baseline_grouped_ardl_r2",
                    "test_baseline_best_simple_r2",
                    "test_r2_delta_vs_persistence",
                    "test_r2_delta_vs_panel_ols",
                    "test_r2_delta_vs_grouped_ardl",
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
                    "feature_bundle",
                    "lag_method",
                    "anchor_proxy_name",
                    "anchor_expected_sign",
                    "proxy_refit_status",
                    "proxy_metric_interpretable",
                    "test_effective_kstar_proxy_spearman_rho",
                    "test_effective_kstar_proxy_spearman_adjusted_rho",
                    "test_effective_kstar_proxy_mean_spearman_rho",
                    "test_effective_kstar_proxy_mean_spearman_adjusted_rho",
                    "test_effective_kstar_proxy_metric_valid",
                    "test_effective_kstar_mean",
                    "test_effective_kstar_std",
                    "test_effective_lag_entropy_mean",
                    "test_effective_lag_entropy_std",
                    "test_effective_lag_top1_share",
                    "test_z_std",
                    "test_z_proxy_spearman_adjusted_rho",
                    "test_z_anchor_adjusted_rho",
                    "test_omega_entropy_penalty",
                    "test_omega_entropy_band_violation_share",
                    "test_z_anchor_loss",
                    "test_lag_gate_sensitivity_slope",
                    "test_lag_gate_sensitivity_range",
                    "test_proxy_signal_r2",
                    "test_proxy_signal_metric_valid",
                    "omega_transform",
                    "lambda_omega_entropy",
                    "omega_entropy_min",
                    "omega_entropy_max",
                    "lambda_z_anchor",
                    "z_anchor_target_sign",
                ],
            )


if __name__ == "__main__":
    unittest.main()