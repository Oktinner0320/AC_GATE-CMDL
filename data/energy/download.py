"""Energy-domain raw-data download and merge utility.

该脚本为 energy-CO2 域生成稳定的本地缓存表：
1. 读取 OWID energy 原始表；
2. 拉取或读取 WGI 三个治理维度；
3. 统一国家代码并合并为单一 CSV，供后续 loader 直接使用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd


DEFAULT_OWID_ENERGY_SOURCE = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
DEFAULT_ENERGY_OUTPUT_PATH = Path(__file__).resolve().parent / "raw" / "energy_wgi_merged.csv"
DEFAULT_YEAR_START = 1996
DEFAULT_YEAR_END = 2023
DEFAULT_WGI_INDICATORS = {
	"GE.EST": "government_effectiveness",
	"RQ.EST": "regulatory_quality",
	"RL.EST": "rule_of_law",
}
_WORLD_BANK_API_ROOT = "https://api.worldbank.org/v2/country/all/indicator"


def _looks_like_excel_source(source: str) -> bool:
	"""Infer whether a URL or path should be read as Excel."""

	normalized = source.lower()
	return normalized.endswith(".xlsx") or normalized.endswith(".xls")


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for the energy download step."""

	parser = argparse.ArgumentParser(description="Download and merge the energy-domain raw tables.")
	parser.add_argument(
		"--energy-source",
		type=str,
		default=DEFAULT_OWID_ENERGY_SOURCE,
		help="OWID energy CSV URL or local CSV path.",
	)
	parser.add_argument(
		"--governance-source",
		type=str,
		default=None,
		help="Optional local governance CSV/Excel path. When omitted, WGI is fetched from the World Bank API.",
	)
	parser.add_argument("--year-start", type=int, default=DEFAULT_YEAR_START)
	parser.add_argument("--year-end", type=int, default=DEFAULT_YEAR_END)
	parser.add_argument(
		"--output",
		type=str,
		default=str(DEFAULT_ENERGY_OUTPUT_PATH),
		help="Destination CSV path for the merged energy cache.",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Overwrite the existing merged cache if it already exists.",
	)
	return parser.parse_args()


def _read_json_url(url: str) -> list[Any]:
	"""Read one JSON payload from a URL."""

	try:
		with urllib.request.urlopen(url) as response:
			return json.loads(response.read().decode("utf-8"))
	except urllib.error.URLError as error:
		raise RuntimeError(f"Failed to fetch URL: {url}") from error


def _load_table_source(source: str) -> pd.DataFrame:
	"""Load a CSV or Excel source into a dataframe."""

	if _looks_like_excel_source(source):
		return pd.read_excel(source)
	return pd.read_csv(source, encoding="utf-8")


def load_energy_source(
	source: str = DEFAULT_OWID_ENERGY_SOURCE,
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
) -> pd.DataFrame:
	"""Load and standardize the OWID energy source."""

	frame = _load_table_source(source).copy()
	code_column = "iso_code" if "iso_code" in frame.columns else "entity_code"
	name_column = "country" if "country" in frame.columns else "entity_name"
	required_columns = {
		code_column,
		name_column,
		"year",
		"renewables_share_energy",
		"co2_per_unit_energy",
		"population",
		"gdp",
	}
	missing_columns = sorted(required_columns.difference(frame.columns))
	if missing_columns:
		raise ValueError(f"Missing required energy columns: {missing_columns}")

	frame["entity_code"] = frame[code_column].astype(str).str.strip().str.upper()
	frame["entity_name"] = frame[name_column].fillna(frame[code_column]).astype(str)
	frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
	for column_name in [
		"renewables_share_energy",
		"co2_per_unit_energy",
		"population",
		"gdp",
	]:
		frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")

	frame = frame.dropna(subset=["entity_code", "year"])
	frame["year"] = frame["year"].astype(int)
	frame = frame[(frame["year"] >= year_start) & (frame["year"] <= year_end)].copy()
	frame = frame[frame["entity_code"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
	frame = frame.drop_duplicates(subset=["entity_code", "year"], keep="last")
	frame = frame.sort_values(["entity_code", "year"]).reset_index(drop=True)

	return frame.loc[
		:,
		[
			"entity_code",
			"entity_name",
			"year",
			"renewables_share_energy",
			"co2_per_unit_energy",
			"population",
			"gdp",
		],
	]


def _world_bank_indicator_url(indicator: str, page: int, per_page: int) -> str:
	"""Build the World Bank API URL for one indicator page."""

	query = urllib.parse.urlencode(
		{
			"format": "json",
			"per_page": per_page,
			"page": page,
		}
	)
	return f"{_WORLD_BANK_API_ROOT}/{indicator}?{query}"


def _fetch_wgi_indicator(indicator: str, alias: str, year_start: int, year_end: int) -> pd.DataFrame:
	"""Fetch one WGI indicator panel from the World Bank API."""

	per_page = 20000
	first_page = _read_json_url(_world_bank_indicator_url(indicator=indicator, page=1, per_page=per_page))
	if not isinstance(first_page, list) or len(first_page) != 2:
		raise RuntimeError(f"Unexpected World Bank API response for indicator {indicator}")

	metadata, rows = first_page
	pages = int(metadata.get("pages", 1))
	all_rows = list(rows or [])
	for page in range(2, pages + 1):
		payload = _read_json_url(_world_bank_indicator_url(indicator=indicator, page=page, per_page=per_page))
		if not isinstance(payload, list) or len(payload) != 2:
			raise RuntimeError(f"Unexpected paged response for indicator {indicator}, page {page}")
		all_rows.extend(payload[1] or [])

	records: list[dict[str, Any]] = []
	for row in all_rows:
		entity_code = str(row.get("countryiso3code") or "").strip().upper()
		if len(entity_code) != 3:
			continue
		year_value = row.get("date")
		if year_value is None:
			continue
		try:
			year = int(year_value)
		except (TypeError, ValueError):
			continue
		if year < year_start or year > year_end:
			continue
		records.append(
			{
				"entity_code": entity_code,
				"entity_name": str((row.get("country") or {}).get("value") or entity_code),
				"year": year,
				alias: row.get("value"),
			}
		)

	if not records:
		raise RuntimeError(f"No WGI rows were returned for indicator {indicator}")

	frame = pd.DataFrame(records)
	frame[alias] = pd.to_numeric(frame[alias], errors="coerce")
	return frame.groupby(["entity_code", "entity_name", "year"], as_index=False).last()


def fetch_wgi_panel(
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
	indicator_map: dict[str, str] | None = None,
) -> pd.DataFrame:
	"""Fetch the three WGI proxy panels and merge them into one dataframe."""

	resolved_indicator_map = DEFAULT_WGI_INDICATORS if indicator_map is None else indicator_map
	merged_frame: pd.DataFrame | None = None
	for indicator, alias in resolved_indicator_map.items():
		indicator_frame = _fetch_wgi_indicator(
			indicator=indicator,
			alias=alias,
			year_start=year_start,
			year_end=year_end,
		)
		if merged_frame is None:
			merged_frame = indicator_frame
		else:
			merged_frame = merged_frame.merge(
				indicator_frame.loc[:, ["entity_code", "year", alias]],
				on=["entity_code", "year"],
				how="outer",
			)

	if merged_frame is None:
		raise RuntimeError("WGI fetch returned no data")

	merged_frame = merged_frame.sort_values(["entity_code", "year"]).reset_index(drop=True)
	return merged_frame


def load_governance_source(
	source: str | None = None,
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
) -> pd.DataFrame:
	"""Load the governance source from a local file or the World Bank API."""

	if source is None:
		return fetch_wgi_panel(year_start=year_start, year_end=year_end)

	frame = _load_table_source(source).copy()
	required_columns = {
		"entity_code",
		"year",
		"government_effectiveness",
		"regulatory_quality",
		"rule_of_law",
	}
	missing_columns = sorted(required_columns.difference(frame.columns))
	if missing_columns:
		raise ValueError(f"Missing required governance columns: {missing_columns}")

	if "entity_name" not in frame.columns:
		frame["entity_name"] = frame["entity_code"]

	frame["entity_code"] = frame["entity_code"].astype(str).str.strip().str.upper()
	frame["entity_name"] = frame["entity_name"].fillna(frame["entity_code"]).astype(str)
	frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
	for column_name in ["government_effectiveness", "regulatory_quality", "rule_of_law"]:
		frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")

	frame = frame.dropna(subset=["entity_code", "year"])
	frame["year"] = frame["year"].astype(int)
	frame = frame[(frame["year"] >= year_start) & (frame["year"] <= year_end)].copy()
	frame = frame[frame["entity_code"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
	frame = frame.drop_duplicates(subset=["entity_code", "year"], keep="last")
	frame = frame.sort_values(["entity_code", "year"]).reset_index(drop=True)

	return frame.loc[
		:,
		[
			"entity_code",
			"entity_name",
			"year",
			"government_effectiveness",
			"regulatory_quality",
			"rule_of_law",
		],
	]


def download_energy_table(
	energy_source: str = DEFAULT_OWID_ENERGY_SOURCE,
	governance_source: str | None = None,
	output_path: str | Path = DEFAULT_ENERGY_OUTPUT_PATH,
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
	force: bool = False,
) -> Path:
	"""Download or merge the energy-domain raw tables into one local CSV cache."""

	destination = Path(output_path).resolve()
	if destination.exists() and not force:
		return destination

	energy_frame = load_energy_source(
		source=energy_source,
		year_start=year_start,
		year_end=year_end,
	)
	governance_frame = load_governance_source(
		source=governance_source,
		year_start=year_start,
		year_end=year_end,
	)
	merged_frame = energy_frame.merge(
		governance_frame.loc[
			:,
			[
				"entity_code",
				"year",
				"government_effectiveness",
				"regulatory_quality",
				"rule_of_law",
			],
		],
		on=["entity_code", "year"],
		how="left",
	)
	merged_frame = merged_frame.sort_values(["entity_code", "year"]).reset_index(drop=True)

	destination.parent.mkdir(parents=True, exist_ok=True)
	merged_frame.to_csv(destination, index=False)
	return destination


def main() -> None:
	"""Execute the energy raw-data download step."""

	args = parse_args()
	local_path = download_energy_table(
		energy_source=args.energy_source,
		governance_source=args.governance_source,
		output_path=args.output,
		year_start=args.year_start,
		year_end=args.year_end,
		force=args.force,
	)
	dataframe = pd.read_csv(local_path)
	entity_count = int(dataframe["entity_code"].nunique()) if not dataframe.empty else 0
	year_min = int(dataframe["year"].min()) if not dataframe.empty else args.year_start
	year_max = int(dataframe["year"].max()) if not dataframe.empty else args.year_end
	print(f"Saved energy raw data to: {local_path}")
	print(f"Rows: {len(dataframe)}, Entities: {entity_count}, Years: {year_min}..{year_max}")


if __name__ == "__main__":
	main()


__all__ = [
	"DEFAULT_ENERGY_OUTPUT_PATH",
	"DEFAULT_OWID_ENERGY_SOURCE",
	"DEFAULT_WGI_INDICATORS",
	"DEFAULT_YEAR_END",
	"DEFAULT_YEAR_START",
	"download_energy_table",
	"fetch_wgi_panel",
	"load_energy_source",
	"load_governance_source",
]
