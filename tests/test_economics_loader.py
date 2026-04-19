"""Regression tests for the economics-domain loader and temporal splitting.

覆盖 PWT loader 的张量契约、缺失值过滤与按时间切分逻辑。
"""

from argparse import Namespace
import os
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch

from data.economics.download import download_pwt_table
from data.economics.economics_loader import (
    build_temporal_splits,
    get_prediction_years,
    load_economics_panel,
)
from experiments.run_economics_lstm_baseline import run_experiment as run_baseline_experiment
from experiments.run_economics import run_experiment


class EconomicsLoaderTest(unittest.TestCase):
    """Validate economics loader output contracts and split windows.

    验证 economics loader 的输出契约与时间切分窗口。
    """

    @staticmethod
    def _write_fixture_csv(target_path: Path, year_start: int = 1980, year_end: int = 1991) -> None:
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

    def test_loader_returns_balanced_dense_panel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "pwt_fixture.csv"
            self._write_fixture_csv(csv_path)

            panel = load_economics_panel(
                csv_path=csv_path,
                year_start=1980,
                year_end=1991,
                max_missing_share=0.20,
            )

            self.assertEqual(tuple(panel.X_it.shape), (3, 12, 1))
            self.assertEqual(tuple(panel.Y_it.shape), (3, 12))
            self.assertEqual(tuple(panel.p_i.shape), (3, 1))
            self.assertEqual(tuple(panel.s_i.shape), (3, 2))
            self.assertEqual(panel.entity_codes, ["AAA", "BBB", "CCC"])
            self.assertEqual(panel.metadata["years"], list(range(1980, 1992)))
            torch.testing.assert_close(panel.entity_ids, torch.arange(3, dtype=torch.long))
            self.assertTrue(torch.isfinite(panel.X_it).all().item())
            self.assertTrue(torch.isfinite(panel.Y_it).all().item())
            self.assertTrue(torch.isfinite(panel.p_i).all().item())
            self.assertTrue(torch.isfinite(panel.s_i).all().item())

    def test_loader_uses_train_window_stats_for_scaling_and_static_features(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            base_csv_path = temporary_root / "pwt_fixture_base.csv"
            shifted_csv_path = temporary_root / "pwt_fixture_shifted.csv"
            self._write_fixture_csv(base_csv_path, year_start=1980, year_end=1991)

            shifted_frame = pd.read_csv(base_csv_path)
            future_mask = (shifted_frame["countrycode"] == "AAA") & (shifted_frame["year"] >= 1988)
            shifted_frame.loc[future_mask, "hc"] = shifted_frame.loc[future_mask, "hc"] + 50.0
            shifted_frame.loc[future_mask, "ck"] = shifted_frame.loc[future_mask, "ck"] * 25.0
            shifted_frame.loc[future_mask, "rgdpna"] = shifted_frame.loc[future_mask, "rgdpna"] * 0.20
            shifted_frame.loc[future_mask, "ctfp"] = shifted_frame.loc[future_mask, "ctfp"] + 100.0
            shifted_frame.to_csv(shifted_csv_path, index=False)

            base_panel = load_economics_panel(
                csv_path=base_csv_path,
                year_start=1980,
                year_end=1991,
                stats_end_year=1987,
                max_missing_share=0.20,
            )
            shifted_panel = load_economics_panel(
                csv_path=shifted_csv_path,
                year_start=1980,
                year_end=1991,
                stats_end_year=1987,
                max_missing_share=0.20,
            )

            torch.testing.assert_close(base_panel.p_i, shifted_panel.p_i, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(base_panel.s_i, shifted_panel.s_i, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(base_panel.X_it[:, :8, :], shifted_panel.X_it[:, :8, :], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(base_panel.Y_it[:, :8], shifted_panel.Y_it[:, :8], atol=1e-6, rtol=1e-6)

    def test_download_utility_reads_excel_data_sheet_and_caches_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            csv_fixture = temporary_root / "pwt_fixture.csv"
            xlsx_fixture = temporary_root / "pwt_fixture.xlsx"
            cached_output = temporary_root / "cached_pwt.csv"
            self._write_fixture_csv(csv_fixture)

            dataframe = pd.read_csv(csv_fixture)
            with pd.ExcelWriter(xlsx_fixture) as writer:
                pd.DataFrame({"note": ["fixture"]}).to_excel(writer, sheet_name="Info", index=False)
                pd.DataFrame({"note": ["fixture"]}).to_excel(writer, sheet_name="Legend", index=False)
                dataframe.to_excel(writer, sheet_name="Data", index=False)

            local_path = download_pwt_table(
                source=str(xlsx_fixture),
                output_path=cached_output,
                sheet_name="Data",
                force=True,
            )
            cached_frame = pd.read_csv(local_path)

            self.assertTrue(local_path.exists())
            self.assertEqual(local_path.suffix.lower(), ".csv")
            self.assertTrue({"countrycode", "year", "ctfp", "hc", "ck", "rgdpna"}.issubset(cached_frame.columns))
            self.assertEqual(len(cached_frame), len(dataframe))

    def test_temporal_splits_keep_context_years(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            csv_path = Path(temporary_dir) / "pwt_fixture.csv"
            self._write_fixture_csv(csv_path)

            panel = load_economics_panel(
                csv_path=csv_path,
                year_start=1980,
                year_end=1991,
                max_missing_share=0.20,
            )
            train_panel, val_panel, test_panel = build_temporal_splits(
                panel=panel,
                max_lag=4,
                train_end_year=1987,
                val_end_year=1989,
            )

            self.assertEqual(train_panel.metadata["years"], [1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987])
            self.assertEqual(val_panel.metadata["years"], [1984, 1985, 1986, 1987, 1988, 1989])
            self.assertEqual(test_panel.metadata["years"], [1986, 1987, 1988, 1989, 1990, 1991])
            self.assertEqual(get_prediction_years(val_panel, 4), [1988, 1989])
            self.assertEqual(get_prediction_years(test_panel, 4), [1990, 1991])

    def test_run_economics_smoke_writes_summary_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            csv_path = temporary_root / "pwt_fixture_long.csv"
            output_root = temporary_root / "outputs"
            self._write_fixture_csv(csv_path, year_start=1980, year_end=2004)

            args = Namespace(
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
                output_dir=str(output_root),
                experiment_name="economics_smoke",
                device="cpu",
                disable_mlflow=True,
                log_every=1,
                smoke=True,
            )

            summary = run_experiment(args)
            run_dir = output_root / "economics_smoke"

            self.assertEqual(summary["experiment"], "economics_smoke")
            self.assertIn("test", summary["metrics"])
            self.assertTrue(np.isfinite(summary["metrics"]["test"]["mse"]))
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "predictions.csv").exists())
            self.assertTrue((run_dir / "history.csv").exists())

    def test_run_economics_lstm_baseline_smoke_writes_summary_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            csv_path = temporary_root / "pwt_fixture_long.csv"
            output_root = temporary_root / "baseline_outputs"
            self._write_fixture_csv(csv_path, year_start=1980, year_end=2004)

            args = Namespace(
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
                grad_clip=1.0,
                output_dir=str(output_root),
                experiment_name="economics_lstm_smoke",
                device="cpu",
                disable_mlflow=True,
                log_every=1,
                smoke=True,
            )

            summary = run_baseline_experiment(args)
            run_dir = output_root / "economics_lstm_smoke"

            self.assertEqual(summary["model"], "plain_lstm")
            self.assertIn("test", summary["metrics"])
            self.assertTrue(np.isfinite(summary["metrics"]["test"]["mse"]))
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "predictions.csv").exists())
            self.assertTrue((run_dir / "history.csv").exists())


if __name__ == "__main__":
    unittest.main()