"""Regression tests for the economics-domain ablation runner.

覆盖 economics ablation 的 smoke 运行、逐 run 写盘与汇总表生成。
"""

from argparse import Namespace
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from experiments.run_economics import setup_experiment  # noqa: E402
from experiments.run_economics_ablation import (  # noqa: E402
    aggregate_results,
    build_ablation_decision_log,
    prepare_variant_setup,
    run_suite,
)


class EconomicsAblationTest(unittest.TestCase):
    """Validate the economics ablation suite on a small smoke fixture.

    验证 economics ablation 在小型夹具上的冒烟运行与写盘行为。
    """

    @staticmethod
    def _write_fixture_csv(target_path: Path, year_start: int = 1980, year_end: int = 2004) -> None:
        years = list(range(year_start, year_end + 1))
        rows: list[dict[str, float | int | str]] = []
        country_specs = [
            ("AAA", "Alpha", 1.00, 0.80),
            ("BBB", "Beta", 1.20, 0.95),
            ("CCC", "Gamma", 1.40, 1.10),
            ("DROP", "Dropout", 0.90, 0.70),
        ]

        for country_index, (country_code, country_name, hc_base, tfp_base) in enumerate(country_specs):
            for step, year in enumerate(years):
                ck_value = 90.0 + 15.0 * country_index + 2.5 * step
                rgdpna_value = 55.0 + 8.0 * country_index + 1.8 * step
                row = {
                    "countrycode": country_code,
                    "country": country_name,
                    "year": year,
                    "hc": hc_base + 0.015 * step,
                    "ck": ck_value,
                    "rgdpna": rgdpna_value,
                    "ctfp": tfp_base + 0.030 * step,
                    "rtfpna": 1.10 + 0.020 * country_index + 0.015 * step,
                    "emp": 10.0 + 2.0 * country_index + 0.10 * step,
                    "avh": 1800.0 + 20.0 * country_index + 1.0 * step,
                    "labsh": 0.45 + 0.01 * country_index + 0.001 * step,
                }

                if country_code == "BBB" and year in {1983, 1984}:
                    row["ctfp"] = np.nan
                if country_code == "DROP" and year in {1981, 1982, 1983, 1984, 1985}:
                    row["hc"] = np.nan
                    row["ck"] = np.nan
                    row["rgdpna"] = np.nan
                    row["ctfp"] = np.nan

                rows.append(row)

        pd.DataFrame(rows).to_csv(target_path, index=False)

    def test_run_suite_smoke_writes_variant_outputs_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            csv_path = temporary_root / "pwt_fixture_long.csv"
            output_root = temporary_root / "economics_ablation_outputs"
            self._write_fixture_csv(csv_path)

            args = Namespace(
                variant="all",
                seeds=[0],
                csv_path=str(csv_path),
                year_start=1980,
                year_end=2004,
                train_end_year=1997,
                val_end_year=2000,
                target_column="ctfp",
                max_missing_share=0.20,
                lr=1e-3,
                epochs=1,
                patience=1,
                lambda_r=0.1,
                temperature=1.0,
                lag_bias_strength=1.0,
                grad_clip=1.0,
                output_dir=str(output_root),
                experiment_prefix="economics_ablation",
                device="cpu",
                disable_mlflow=True,
                log_every=1,
                smoke=True,
            )

            summary_frame, aggregated = run_suite(args)

            self.assertEqual(sorted(summary_frame["variant"].tolist()), [
                "no_ac_encoder",
                "no_recon_regularization",
                "uniform_lag",
            ])
            self.assertTrue((output_root / "ablation_results.csv").exists())
            self.assertTrue((output_root / "ablation_results.json").exists())
            self.assertTrue((output_root / "ablation_results_aggregated.csv").exists())
            self.assertTrue((output_root / "ablation_mechanism_summary.csv").exists())
            self.assertTrue((output_root / "ablation_decision_log.csv").exists())
            self.assertEqual(len(aggregated), 3)
            self.assertTrue((aggregated["n_seeds"] == 1).all())

            no_recon_run_dir = output_root / "economics_ablation_no_recon_regularization_seed0"
            no_recon_summary = json.loads((no_recon_run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(no_recon_summary["variant"], "no_recon_regularization")
            self.assertAlmostEqual(no_recon_summary["effective_lambda_r"], 0.0)
            self.assertTrue(no_recon_summary["diagnostics"]["ablation"]["matched_init_to_full_cmdl"])
            self.assertEqual(
                no_recon_summary["diagnostics"]["ablation"]["causal_ablation_validity"],
                "matched_init_effective_lambda_only",
            )

            no_ac_summary = json.loads(
                (output_root / "economics_ablation_no_ac_encoder_seed0" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                no_ac_summary["diagnostics"]["proxy_refit"]["status"],
                "skipped_rank_deficient",
            )
            self.assertFalse(no_ac_summary["diagnostics"]["proxy_refit"]["applied"])
            self.assertFalse(no_ac_summary["diagnostics"]["proxy_refit"]["metrics_interpretable"])
            self.assertTrue(np.isnan(no_ac_summary["metrics"]["test"]["proxy_recon_r2"]))
            self.assertTrue(np.isnan(no_ac_summary["metrics"]["test"]["kstar_proxy_spearman_rho"]))
            no_ac_row = summary_frame.loc[summary_frame["variant"] == "no_ac_encoder"].iloc[0]
            self.assertEqual(no_ac_row["proxy_refit_status"], "skipped_rank_deficient")
            self.assertFalse(bool(no_ac_row["proxy_metric_interpretable"]))

            for variant in ["no_ac_encoder", "uniform_lag", "no_recon_regularization"]:
                run_dir = output_root / f"economics_ablation_{variant}_seed0"
                self.assertTrue((run_dir / "summary.json").exists())
                self.assertTrue((run_dir / "predictions.csv").exists())
                self.assertTrue((run_dir / "history.csv").exists())
                self.assertTrue((run_dir / "best_model.pt").exists())

    def test_aggregate_results_reports_multiseed_positive_shares(self) -> None:
        rows = [
            {
                "variant": "no_ac_encoder",
                "target_column": "ctfp",
                "feature_bundle": "effective_labor_aware",
                "seed": 0,
                "test_kstar_proxy_spearman_adjusted_rho": 0.20,
                "test_kstar_std": 0.00,
                "test_lag_gate_sensitivity_range": 0.00,
                "test_proxy_recon_r2": np.nan,
            },
            {
                "variant": "no_ac_encoder",
                "target_column": "ctfp",
                "feature_bundle": "effective_labor_aware",
                "seed": 1,
                "test_kstar_proxy_spearman_adjusted_rho": -0.10,
                "test_kstar_std": 0.00,
                "test_lag_gate_sensitivity_range": 0.00,
                "test_proxy_recon_r2": np.nan,
            },
            {
                "variant": "uniform_lag",
                "target_column": "ctfp",
                "feature_bundle": "effective_labor_aware",
                "seed": 0,
                "test_kstar_proxy_spearman_adjusted_rho": 0.30,
                "test_kstar_std": 0.00,
                "test_lag_gate_sensitivity_range": 0.10,
                "test_proxy_recon_r2": 0.20,
            },
            {
                "variant": "uniform_lag",
                "target_column": "ctfp",
                "feature_bundle": "effective_labor_aware",
                "seed": 1,
                "test_kstar_proxy_spearman_adjusted_rho": 0.40,
                "test_kstar_std": 0.00,
                "test_lag_gate_sensitivity_range": 0.10,
                "test_proxy_recon_r2": 0.25,
            },
        ]

        aggregated = aggregate_results(rows)
        decision_log = build_ablation_decision_log(pd.DataFrame(rows))

        no_ac_row = aggregated.loc[aggregated["variant"] == "no_ac_encoder"].iloc[0]
        uniform_row = aggregated.loc[aggregated["variant"] == "uniform_lag"].iloc[0]
        self.assertEqual(int(no_ac_row["n_seeds"]), 2)
        self.assertAlmostEqual(no_ac_row["test_kstar_proxy_spearman_adjusted_rho_positive_share"], 0.5)
        self.assertAlmostEqual(uniform_row["test_kstar_proxy_spearman_adjusted_rho_positive_share"], 1.0)

        uniform_decisions = decision_log.loc[decision_log["variant"] == "uniform_lag"]
        answers = dict(zip(uniform_decisions["layer"], uniform_decisions["answer"]))
        self.assertEqual(answers["mechanism_direction"], "yes")
        self.assertEqual(answers["lag_heterogeneity"], "no")

    def test_run_suite_smoke_supports_effective_labor_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            csv_path = temporary_root / "pwt_fixture_long.csv"
            output_root = temporary_root / "economics_ablation_outputs_effective_labor"
            self._write_fixture_csv(csv_path)

            args = Namespace(
                variant="all",
                seeds=[0],
                csv_path=str(csv_path),
                year_start=1980,
                year_end=2004,
                train_end_year=1997,
                val_end_year=2000,
                target_column="ctfp",
                feature_bundle="effective_labor_aware",
                max_missing_share=0.20,
                lr=1e-3,
                epochs=1,
                patience=1,
                lambda_r=0.1,
                temperature=1.0,
                lag_bias_strength=1.0,
                grad_clip=1.0,
                output_dir=str(output_root),
                experiment_prefix="economics_ablation_effective_labor",
                device="cpu",
                disable_mlflow=True,
                log_every=1,
                smoke=True,
            )

            summary_frame, aggregated = run_suite(args)

            self.assertEqual(len(summary_frame), 3)
            self.assertEqual(len(aggregated), 3)
            for variant in ["no_ac_encoder", "uniform_lag", "no_recon_regularization"]:
                summary = json.loads(
                    (
                        output_root / f"economics_ablation_effective_labor_{variant}_seed0" / "summary.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(summary["data"]["feature_bundle"], "effective_labor_aware")

    def test_no_recon_prepare_variant_setup_reuses_full_cmdl_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            csv_path = temporary_root / "pwt_fixture_long.csv"
            self._write_fixture_csv(csv_path)

            base_args = Namespace(
                csv_path=str(csv_path),
                year_start=1980,
                year_end=2004,
                train_end_year=1997,
                val_end_year=2000,
                target_column="ctfp",
                max_missing_share=0.20,
                seed=42,
                lr=1e-3,
                epochs=1,
                patience=1,
                lambda_r=0.1,
                temperature=1.0,
                lag_bias_strength=1.0,
                grad_clip=1.0,
                output_dir=str(temporary_root / "outputs"),
                experiment_name="economics_full_reference",
                experiment_prefix="economics_ablation",
                device="cpu",
                disable_mlflow=True,
                log_every=1,
                smoke=True,
            )

            full_setup = setup_experiment(Namespace(**vars(base_args)))
            no_recon_setup, _, effective_lambda_r, ablation_diagnostics = prepare_variant_setup(
                args=base_args,
                variant="no_recon_regularization",
                seed=42,
            )

            self.assertEqual(effective_lambda_r, 0.0)
            self.assertTrue(ablation_diagnostics["matched_init_to_full_cmdl"])
            self.assertEqual(
                ablation_diagnostics["causal_ablation_validity"],
                "matched_init_effective_lambda_only",
            )
            for parameter_name, reference_tensor in full_setup.model.state_dict().items():
                torch.testing.assert_close(
                    no_recon_setup.model.state_dict()[parameter_name],
                    reference_tensor,
                    atol=0.0,
                    rtol=0.0,
                )


if __name__ == "__main__":
    unittest.main()