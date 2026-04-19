"""Regression tests for the energy-domain runner and plain-LSTM baseline.

覆盖 energy CMDL runner 与 plain-LSTM baseline 的 smoke 运行与写盘行为。
"""

from argparse import Namespace
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from data.energy.download import download_energy_table  # noqa: E402
from experiments.run_energy import run_experiment  # noqa: E402
from experiments.run_energy_lstm_baseline import run_experiment as run_baseline_experiment  # noqa: E402
from tests.test_energy_loader import EnergyLoaderTest  # noqa: E402


class EnergyRunnerTest(unittest.TestCase):
    """Validate smoke execution for the energy CMDL runner and baseline."""

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

    def test_run_energy_smoke_writes_summary_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            csv_path = self._build_merged_fixture(temporary_root)
            output_root = temporary_root / "outputs"

            args = Namespace(
                csv_path=str(csv_path),
                year_start=1996,
                year_end=2023,
                train_end_year=2011,
                val_end_year=2017,
                treatment_column="renewables_share_energy",
                target_column="co2_per_unit_energy",
                max_missing_share=0.15,
                seed=42,
                lr=1e-3,
                epochs=1,
                patience=1,
                lambda_r=0.1,
                temperature=1.0,
                lag_bias_strength=1.0,
                grad_clip=1.0,
                output_dir=str(output_root),
                experiment_name="energy_smoke",
                device="cpu",
                disable_mlflow=True,
                log_every=1,
                smoke=True,
            )

            summary = run_experiment(args)
            run_dir = output_root / "energy_smoke"

            self.assertEqual(summary["experiment"], "energy_smoke")
            self.assertEqual(summary["data"]["treatment_column"], "renewables_share_energy")
            self.assertEqual(summary["data"]["target_column"], "co2_per_unit_energy")
            self.assertEqual(summary["data"]["proxy_columns"], [
                "proxy_government_effectiveness",
                "proxy_regulatory_quality",
                "proxy_rule_of_law",
            ])
            self.assertIn("test", summary["metrics"])
            self.assertTrue(np.isfinite(summary["metrics"]["test"]["mse"]))
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "predictions.csv").exists())
            self.assertTrue((run_dir / "history.csv").exists())
            self.assertTrue((run_dir / "best_model.pt").exists())

    def test_run_energy_lstm_baseline_smoke_writes_summary_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            csv_path = self._build_merged_fixture(temporary_root)
            output_root = temporary_root / "baseline_outputs"

            args = Namespace(
                csv_path=str(csv_path),
                year_start=1996,
                year_end=2023,
                train_end_year=2011,
                val_end_year=2017,
                treatment_column="renewables_share_energy",
                target_column="co2_per_unit_energy",
                max_missing_share=0.15,
                seed=42,
                lr=1e-3,
                epochs=1,
                patience=1,
                grad_clip=1.0,
                output_dir=str(output_root),
                experiment_name="energy_lstm_smoke",
                device="cpu",
                disable_mlflow=True,
                log_every=1,
                smoke=True,
            )

            summary = run_baseline_experiment(args)
            run_dir = output_root / "energy_lstm_smoke"

            self.assertEqual(summary["model"], "plain_lstm")
            self.assertEqual(summary["data"]["treatment_column"], "renewables_share_energy")
            self.assertIn("test", summary["metrics"])
            self.assertTrue(np.isfinite(summary["metrics"]["test"]["mse"]))
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "predictions.csv").exists())
            self.assertTrue((run_dir / "history.csv").exists())
            self.assertTrue((run_dir / "best_model.pt").exists())


if __name__ == "__main__":
    unittest.main()