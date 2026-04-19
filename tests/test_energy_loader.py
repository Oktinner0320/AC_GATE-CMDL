"""Regression tests for the minimal energy-domain downloader and loader.

覆盖 energy raw-data merge、张量契约、训练窗口统计和时间切分逻辑。
"""

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


from config.cmdl_config import CMDLConfig
from data.energy.download import download_energy_table
from data.energy.energy_loader import build_temporal_splits, get_prediction_years, load_energy_panel


class EnergyLoaderTest(unittest.TestCase):
	"""Validate the energy-domain raw merge and tensor contract."""

	@staticmethod
	def _write_energy_fixture(target_path: Path, year_start: int = 1996, year_end: int = 2004) -> None:
		years = list(range(year_start, year_end + 1))
		rows: list[dict[str, float | int | str]] = []
		country_specs = [
			("AAA", "Alpha", 12.0, 0.92),
			("BBB", "Beta", 18.0, 0.88),
			("CCC", "Gamma", 24.0, 0.84),
			("DRP", "Dropout", 10.0, 0.96),
		]

		for country_index, (entity_code, entity_name, renewable_base, co2_base) in enumerate(country_specs):
			for step, year in enumerate(years):
				rows.append(
					{
						"iso_code": entity_code,
						"country": entity_name,
						"year": year,
						"renewables_share_energy": renewable_base + 1.5 * step,
						"co2_per_unit_energy": co2_base - 0.012 * step - 0.01 * country_index,
						"population": 5_000_000 + 400_000 * country_index + 30_000 * step,
						"gdp": 60_000_000_000 + 7_000_000_000 * country_index + 600_000_000 * step,
					}
				)

		for step, year in enumerate(years):
			rows.append(
				{
					"iso_code": "OWID_WRL",
					"country": "World",
					"year": year,
					"renewables_share_energy": 20.0 + step,
					"co2_per_unit_energy": 0.90 - 0.01 * step,
					"population": 1_000_000_000,
					"gdp": 1_000_000_000_000,
				}
			)

		frame = pd.DataFrame(rows)
		frame.loc[(frame["iso_code"] == "BBB") & (frame["year"].isin([1998, 1999])), "co2_per_unit_energy"] = np.nan
		frame.loc[(frame["iso_code"] == "BBB") & (frame["year"] == 1999), "renewables_share_energy"] = np.nan
		frame.loc[(frame["iso_code"] == "DRP") & (frame["year"].isin([1997, 1998, 1999, 2000, 2001])), "co2_per_unit_energy"] = np.nan
		frame.loc[(frame["iso_code"] == "DRP") & (frame["year"].isin([1997, 1998, 1999, 2000, 2001])), "renewables_share_energy"] = np.nan
		frame.to_csv(target_path, index=False)

	@staticmethod
	def _write_governance_fixture(target_path: Path, year_start: int = 1996, year_end: int = 2004) -> None:
		years = list(range(year_start, year_end + 1))
		rows: list[dict[str, float | int | str]] = []
		country_specs = [
			("AAA", "Alpha", 0.4),
			("BBB", "Beta", -0.1),
			("CCC", "Gamma", 0.8),
			("DRP", "Dropout", -0.5),
		]

		for country_index, (entity_code, entity_name, governance_base) in enumerate(country_specs):
			for step, year in enumerate(years):
				rows.append(
					{
						"entity_code": entity_code,
						"entity_name": entity_name,
						"year": year,
						"government_effectiveness": governance_base + 0.02 * step,
						"regulatory_quality": governance_base + 0.03 * step + 0.1,
						"rule_of_law": governance_base + 0.01 * step + 0.2,
					}
				)

		frame = pd.DataFrame(rows)
		frame.loc[(frame["entity_code"] == "BBB") & (frame["year"] == 1999), "regulatory_quality"] = np.nan
		frame.loc[
			(frame["entity_code"] == "DRP") & (frame["year"].isin([1997, 1998, 1999, 2000, 2001])),
			["government_effectiveness", "regulatory_quality", "rule_of_law"],
		] = np.nan
		frame.to_csv(target_path, index=False)

	def test_download_utility_merges_local_sources_and_filters_aggregates(self) -> None:
		with tempfile.TemporaryDirectory() as temporary_dir:
			temporary_root = Path(temporary_dir)
			energy_source = temporary_root / "owid_energy_fixture.csv"
			governance_source = temporary_root / "wgi_fixture.csv"
			output_path = temporary_root / "merged" / "energy_wgi.csv"
			self._write_energy_fixture(energy_source)
			self._write_governance_fixture(governance_source)

			saved_path = download_energy_table(
				energy_source=str(energy_source),
				governance_source=str(governance_source),
				output_path=output_path,
				year_start=1996,
				year_end=2004,
				force=True,
			)
			merged_frame = pd.read_csv(saved_path)

			self.assertTrue(saved_path.exists())
			self.assertEqual(saved_path, output_path.resolve())
			self.assertEqual(sorted(merged_frame["entity_code"].unique().tolist()), ["AAA", "BBB", "CCC", "DRP"])
			self.assertTrue(
				{
					"entity_code",
					"entity_name",
					"year",
					"renewables_share_energy",
					"co2_per_unit_energy",
					"population",
					"gdp",
					"government_effectiveness",
					"regulatory_quality",
					"rule_of_law",
				}.issubset(merged_frame.columns)
			)

	def test_loader_returns_balanced_dense_panel(self) -> None:
		with tempfile.TemporaryDirectory() as temporary_dir:
			temporary_root = Path(temporary_dir)
			energy_source = temporary_root / "owid_energy_fixture.csv"
			governance_source = temporary_root / "wgi_fixture.csv"
			merged_path = temporary_root / "energy_wgi.csv"
			self._write_energy_fixture(energy_source)
			self._write_governance_fixture(governance_source)
			download_energy_table(
				energy_source=str(energy_source),
				governance_source=str(governance_source),
				output_path=merged_path,
				year_start=1996,
				year_end=2004,
				force=True,
			)

			small_cfg = CMDLConfig.from_domain("energy", max_lag=3)
			panel = load_energy_panel(
				csv_path=merged_path,
				cfg=small_cfg,
				year_start=1996,
				year_end=2004,
				stats_end_year=2001,
				max_missing_share=0.20,
			)

			self.assertEqual(tuple(panel.X_it.shape), (3, 9, 1))
			self.assertEqual(tuple(panel.Y_it.shape), (3, 9))
			self.assertEqual(tuple(panel.p_i.shape), (3, 3))
			self.assertEqual(tuple(panel.s_i.shape), (3, 2))
			self.assertEqual(panel.entity_codes, ["AAA", "BBB", "CCC"])
			self.assertEqual(panel.metadata["years"], list(range(1996, 2005)))
			self.assertEqual(panel.metadata["feature_bundle"], "minimal")
			self.assertEqual(panel.metadata["seq_feature_columns"], ["x_t"])
			self.assertEqual(
				panel.metadata["proxy_columns"],
				["proxy_government_effectiveness", "proxy_regulatory_quality", "proxy_rule_of_law"],
			)
			torch.testing.assert_close(panel.entity_ids, torch.arange(3, dtype=torch.long))
			self.assertTrue(torch.isfinite(panel.X_it).all().item())
			self.assertTrue(torch.isfinite(panel.Y_it).all().item())
			self.assertTrue(torch.isfinite(panel.p_i).all().item())
			self.assertTrue(torch.isfinite(panel.s_i).all().item())

	def test_loader_uses_train_window_stats_for_scaling_and_proxy_summaries(self) -> None:
		with tempfile.TemporaryDirectory() as temporary_dir:
			temporary_root = Path(temporary_dir)
			energy_source = temporary_root / "owid_energy_fixture.csv"
			governance_source = temporary_root / "wgi_fixture.csv"
			base_path = temporary_root / "energy_wgi_base.csv"
			shifted_path = temporary_root / "energy_wgi_shifted.csv"
			self._write_energy_fixture(energy_source)
			self._write_governance_fixture(governance_source)
			download_energy_table(
				energy_source=str(energy_source),
				governance_source=str(governance_source),
				output_path=base_path,
				year_start=1996,
				year_end=2004,
				force=True,
			)

			shifted_frame = pd.read_csv(base_path)
			future_mask = (shifted_frame["entity_code"] == "AAA") & (shifted_frame["year"] >= 2002)
			shifted_frame.loc[future_mask, "renewables_share_energy"] = shifted_frame.loc[future_mask, "renewables_share_energy"] + 50.0
			shifted_frame.loc[future_mask, "co2_per_unit_energy"] = shifted_frame.loc[future_mask, "co2_per_unit_energy"] + 5.0
			shifted_frame.loc[future_mask, "population"] = shifted_frame.loc[future_mask, "population"] * 10.0
			shifted_frame.loc[future_mask, "gdp"] = shifted_frame.loc[future_mask, "gdp"] * 30.0
			shifted_frame.loc[future_mask, "government_effectiveness"] = shifted_frame.loc[future_mask, "government_effectiveness"] + 8.0
			shifted_frame.loc[future_mask, "regulatory_quality"] = shifted_frame.loc[future_mask, "regulatory_quality"] - 7.0
			shifted_frame.loc[future_mask, "rule_of_law"] = shifted_frame.loc[future_mask, "rule_of_law"] + 6.0
			shifted_frame.to_csv(shifted_path, index=False)

			small_cfg = CMDLConfig.from_domain("energy", max_lag=3)
			base_panel = load_energy_panel(
				csv_path=base_path,
				cfg=small_cfg,
				year_start=1996,
				year_end=2004,
				stats_end_year=2001,
				max_missing_share=0.20,
			)
			shifted_panel = load_energy_panel(
				csv_path=shifted_path,
				cfg=small_cfg,
				year_start=1996,
				year_end=2004,
				stats_end_year=2001,
				max_missing_share=0.20,
			)

			torch.testing.assert_close(base_panel.p_i, shifted_panel.p_i, atol=1e-6, rtol=1e-6)
			torch.testing.assert_close(base_panel.s_i, shifted_panel.s_i, atol=1e-6, rtol=1e-6)
			torch.testing.assert_close(base_panel.X_it[:, :6, :], shifted_panel.X_it[:, :6, :], atol=1e-6, rtol=1e-6)
			torch.testing.assert_close(base_panel.Y_it[:, :6], shifted_panel.Y_it[:, :6], atol=1e-6, rtol=1e-6)

	def test_temporal_splits_keep_context_years(self) -> None:
		with tempfile.TemporaryDirectory() as temporary_dir:
			temporary_root = Path(temporary_dir)
			energy_source = temporary_root / "owid_energy_fixture.csv"
			governance_source = temporary_root / "wgi_fixture.csv"
			merged_path = temporary_root / "energy_wgi.csv"
			self._write_energy_fixture(energy_source)
			self._write_governance_fixture(governance_source)
			download_energy_table(
				energy_source=str(energy_source),
				governance_source=str(governance_source),
				output_path=merged_path,
				year_start=1996,
				year_end=2004,
				force=True,
			)

			small_cfg = CMDLConfig.from_domain("energy", max_lag=3)
			panel = load_energy_panel(
				csv_path=merged_path,
				cfg=small_cfg,
				year_start=1996,
				year_end=2004,
				stats_end_year=2001,
				max_missing_share=0.20,
			)
			train_panel, val_panel, test_panel = build_temporal_splits(
				panel=panel,
				max_lag=3,
				train_end_year=2000,
				val_end_year=2002,
			)

			self.assertEqual(train_panel.metadata["years"], [1996, 1997, 1998, 1999, 2000])
			self.assertEqual(val_panel.metadata["years"], [1998, 1999, 2000, 2001, 2002])
			self.assertEqual(test_panel.metadata["years"], [2000, 2001, 2002, 2003, 2004])
			self.assertEqual(get_prediction_years(val_panel, 3), [2001, 2002])
			self.assertEqual(get_prediction_years(test_panel, 3), [2003, 2004])


if __name__ == "__main__":
	unittest.main()