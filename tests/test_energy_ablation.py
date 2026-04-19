"""Regression tests for the energy-domain ablation runner.

覆盖 energy ablation 的 smoke 运行、逐 run 写盘与汇总表生成。
"""

from argparse import Namespace
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from data.energy.download import download_energy_table  # noqa: E402
from experiments.run_energy import setup_experiment  # noqa: E402
from experiments.run_energy_ablation import prepare_variant_setup, run_suite  # noqa: E402
from tests.test_energy_loader import EnergyLoaderTest  # noqa: E402


class EnergyAblationTest(unittest.TestCase):
    """Validate the energy ablation suite on a smoke fixture."""

    @staticmethod
    def _build_merged_fixture(temporary_root: Path, year_start: int = 1996, year_end: int = 2023) -> Path:
        energy_source = temporary_root / "owid_energy_fixture.csv"
        governance_source = temporary_root / "wgi_fixture.csv"
        merged_path = temporary_root / "energy_wgi.csv"
        EnergyLoaderTest._write_energy_fixture(energy_source, year_start=year_start, year_end=year_end)
        EnergyLoaderTest._write_governance_fixture(governance_source, year_start=year_start, year_end=year_end)
        download_energy_table(
            energy_source=str(energy_source),
            governance_source=str(governance_source),
            output_path=merged_path,
            year_start=year_start,
            year_end=year_end,
            force=True,
        )
        return merged_path

    def test_run_suite_smoke_writes_variant_outputs_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            csv_path = self._build_merged_fixture(temporary_root)
            output_root = temporary_root / "energy_ablation_outputs"

            args = Namespace(
                variant="all",
                seeds=[0],
                csv_path=str(csv_path),
                year_start=1996,
                year_end=2023,
                train_end_year=2011,
                val_end_year=2017,
                treatment_column="renewables_share_energy",
                target_column="co2_per_unit_energy",
                feature_bundle="minimal",
                max_missing_share=0.15,
                lr=1e-3,
                epochs=1,
                patience=1,
                lambda_r=0.1,
                temperature=1.0,
                lag_bias_strength=1.0,
                grad_clip=1.0,
                output_dir=str(output_root),
                experiment_prefix="energy_ablation",
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
            self.assertEqual(len(aggregated), 3)

            no_recon_run_dir = output_root / "energy_ablation_no_recon_regularization_seed0"
            no_recon_summary = json.loads((no_recon_run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(no_recon_summary["variant"], "no_recon_regularization")
            self.assertAlmostEqual(no_recon_summary["effective_lambda_r"], 0.0)
            self.assertTrue(no_recon_summary["diagnostics"]["ablation"]["matched_init_to_full_cmdl"])
            self.assertEqual(
                no_recon_summary["diagnostics"]["ablation"]["causal_ablation_validity"],
                "matched_init_effective_lambda_only",
            )

            no_ac_summary = json.loads(
                (output_root / "energy_ablation_no_ac_encoder_seed0" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(no_ac_summary["diagnostics"]["proxy_refit"]["status"], "skipped_rank_deficient")
            self.assertFalse(no_ac_summary["diagnostics"]["proxy_refit"]["applied"])
            self.assertFalse(no_ac_summary["diagnostics"]["proxy_refit"]["metrics_interpretable"])
            self.assertTrue(np.isnan(no_ac_summary["metrics"]["test"]["proxy_recon_r2"]))
            self.assertTrue(np.isnan(no_ac_summary["metrics"]["test"]["kstar_proxy_spearman_rho"]))
            no_ac_row = summary_frame.loc[summary_frame["variant"] == "no_ac_encoder"].iloc[0]
            self.assertEqual(no_ac_row["proxy_refit_status"], "skipped_rank_deficient")
            self.assertFalse(bool(no_ac_row["proxy_metric_interpretable"]))

            for variant in ["no_ac_encoder", "uniform_lag", "no_recon_regularization"]:
                run_dir = output_root / f"energy_ablation_{variant}_seed0"
                self.assertTrue((run_dir / "summary.json").exists())
                self.assertTrue((run_dir / "predictions.csv").exists())
                self.assertTrue((run_dir / "history.csv").exists())
                self.assertTrue((run_dir / "best_model.pt").exists())

    def test_no_recon_prepare_variant_setup_reuses_full_cmdl_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            csv_path = self._build_merged_fixture(temporary_root)

            base_args = Namespace(
                csv_path=str(csv_path),
                year_start=1996,
                year_end=2023,
                train_end_year=2011,
                val_end_year=2017,
                treatment_column="renewables_share_energy",
                target_column="co2_per_unit_energy",
                feature_bundle="minimal",
                max_missing_share=0.15,
                seed=42,
                lr=1e-3,
                epochs=1,
                patience=1,
                lambda_r=0.1,
                temperature=1.0,
                lag_bias_strength=1.0,
                grad_clip=1.0,
                output_dir=str(temporary_root / "outputs"),
                experiment_name="energy_full_reference",
                experiment_prefix="energy_ablation",
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