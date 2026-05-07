"""Informal-domain workbook audit and export utility.

This script does not assume that the workbook is already a clean panel.
Instead, it:
1. reads the two-row workbook header and the variable codebook,
2. exports a column inventory and per-block raw CSV files,
3. melts year-like blocks into long CSVs when a block exposes year columns,
4. builds a first RCW candidate panel keyed by year and county.

The observed workbook currently behaves like a set of horizontally concatenated
subtables with different row semantics. Exporting block-level CSVs first keeps
the preprocessing auditable and avoids forcing the wrong global data contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any
import unicodedata

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
	 sys.path.insert(0, str(WORKSPACE_ROOT))


DEFAULT_WORKBOOK_PATH = Path(__file__).resolve().parent / "black_economy_structure - Final 20250819.xlsx"
DEFAULT_CODEBOOK_PATH = Path(__file__).resolve().parent / "variables_table.xlsx"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "processed"
DEFAULT_BLOCK_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "blocks"
ACGATE_OUTPUT_DIRNAME = "acgate_inputs"

YEAR_PATTERN = re.compile(r"^(?P<year>\d{4})(?:\.(?P<suffix>\d+))?$")
MONTH_PATTERN = re.compile(r"^(?P<year>\d{4})M(?P<month>\d{2})$")
COUNTY_CODE_PATTERN = re.compile(r"^(?P<entity_code>[A-Z]{2}\d{2,3})\s+(?P<entity_name>.+)$")
RCW_REGION_CODE_PATTERN = re.compile(r"^SE\d{3}$")
SENTINEL_UNNAMED_PREFIX = "Unnamed:"

ENTITY_CODE_ALIAS_MAP = {
	"SE11": "SE110",
}

TD_MEASURE_LABEL_MAP = {
	"1.1. Direct taxes on labour": "x_td_direct_taxes_on_labour_raw",
	"1.2. Indirect taxes on labour": "x_td_indirect_taxes_on_labour_raw",
	"2. Taxes on capital": "x_td_taxes_on_capital_raw",
	"9. Total tax revenues": "x_td_total_tax_revenues_raw",
}

TD_YEAR_COLUMN_MAP = {
	2019: "td__2019",
	2020: "td__2020",
	2021: "td__2021",
	2022: "td__2022",
	2023: "td__2023",
}

RYDGDP_NUMERATOR_ENTITY_COLUMN = "rydgdp__unnamed_810_level_1"
RYDGDP_NUMERATOR_DESCRIPTOR_COLUMN = "rydgdp__unnamed_811_level_1"

RYDGDP_NUMERATOR_YEAR_COLUMN_MAP = {
	2019: "rydgdp__2019",
	2020: "rydgdp__2020",
	2021: "rydgdp__2021",
	2022: "rydgdp__2022",
	2023: "rydgdp__2023",
}

RYDGDP_NUMERATOR_PER_CAPITA_YEAR_COLUMN_MAP = {
	2019: "rydgdp__2019_1",
	2020: "rydgdp__2020_1",
	2021: "rydgdp__2021_1",
	2022: "rydgdp__2022_1",
	2023: "rydgdp__2023_1",
}

RYDGDP_DENOMINATOR_ENTITY_COLUMN = "rydgdp__unnamed_885_level_1"

RYDGDP_DENOMINATOR_YEAR_COLUMN_MAP = {
	2019: "rydgdp__grdp_current_prices_million_sek_2",
	2020: "rydgdp__unnamed_887_level_1",
	2021: "rydgdp__unnamed_888_level_1",
	2022: "rydgdp__unnamed_889_level_1",
	2023: "rydgdp__unnamed_890_level_1",
}

RYDGDP_DENOMINATOR_PER_CAPITA_YEAR_COLUMN_MAP = {
	2019: "rydgdp__grdp_per_capita_thousand_sek_2",
	2020: "rydgdp__unnamed_897_level_1",
	2021: "rydgdp__unnamed_898_level_1",
	2022: "rydgdp__unnamed_899_level_1",
	2023: "rydgdp__unnamed_900_level_1",
}

RYDGDP_DISPOSABLE_INCOME_LABEL = "disposable income, net"

CURATED_FORMAL_SEQUENCE_RENAME_MAP = {
	"x_rpcyd_total_income_raw": "x_seq_rpcyd_total_income",
	"rydgdp_disposable_income_msek_raw": "x_seq_rydgdp_disposable_income",
	"x_rydgdp_disposable_income_per_capita_raw": "x_seq_rydgdp_disposable_income_per_capita",
	"rydgdp_grdp_msek_raw": "x_seq_rydgdp_grdp",
	"x_rydgdp_grdp_per_capita_raw": "x_seq_rydgdp_grdp_per_capita",
	"x_rydgdp_ratio_raw": "x_seq_rydgdp_ratio",
}

# The workbook uses nearby but not always identical codes relative to the user
# codebook. Keep the observed block code separate, but surface the mapped code.
VARIABLE_CODE_ALIAS_MAP = {
	"RCW": "RCV",
	"PRID": "PRIID",
	"TI": "TII",
	"ISET_ISEMT_ISEFT": "ISET_ISEMT_ISEFT",
}


@dataclass(slots=True)
class InformalExportResult:
	"""Paths written by the informal workbook export utility."""

	audit_json_path: Path
	column_inventory_path: Path
	codebook_registry_path: Path
	block_summary_path: Path
	rcw_candidate_panel_path: Path | None
	rcw_cleaned_panel_path: Path | None
	rcw_model_ready_path: Path | None
	td_candidate_panel_path: Path | None
	rydgdp_candidate_panel_path: Path | None
	rpcyd_panel_path: Path | None
	merged_long_panel_path: Path | None
	acgate_ready_panel_path: Path | None
	acgate_output_dir: Path | None
	acgate_single_feature_panel_path: Path | None
	acgate_multiseq_panel_path: Path | None
	acgate_multiseq_overlap_panel_path: Path | None
	acgate_manifest_path: Path | None
	raw_block_paths: dict[str, Path]
	year_long_block_paths: dict[str, Path]


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for workbook auditing and CSV export."""

	parser = argparse.ArgumentParser(description="Audit and export the informal-domain workbook.")
	parser.add_argument(
		"--workbook-path",
		type=str,
		default=str(DEFAULT_WORKBOOK_PATH),
		help="Path to the mixed-structure informal workbook.",
	)
	parser.add_argument(
		"--codebook-path",
		type=str,
		default=str(DEFAULT_CODEBOOK_PATH),
		help="Path to the variable codebook workbook.",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(DEFAULT_OUTPUT_DIR),
		help="Directory where audit files and CSV exports are written.",
	)
	parser.add_argument(
		"--sheet-name",
		type=str,
		default=None,
		help="Optional sheet name. Defaults to the workbook's first sheet.",
	)
	parser.add_argument(
		"--rcw-min-duplicate-count",
		type=int,
		default=1,
		help="Minimum duplicate row count retained in the RCW candidate panel aggregation.",
	)
	return parser.parse_args()


def _clean_header_cell(value: Any) -> str:
	"""Normalize one header cell into a compact display string."""

	if value is None or pd.isna(value):
		return ""
	text = str(value).replace("\n", " ").strip()
	return re.sub(r"\s+", " ", text)


def _normalize_token(value: str) -> str:
	"""Normalize one token into an uppercase identifier-like string."""

	cleaned = _clean_header_cell(value)
	if not cleaned or cleaned.startswith(SENTINEL_UNNAMED_PREFIX):
		return ""
	normalized = re.sub(r"[^A-Za-z0-9_]+", "_", cleaned).strip("_")
	return normalized.upper()


def _slugify_token(value: str) -> str:
	"""Convert a free-form label into a stable snake_case token."""

	cleaned = _clean_header_cell(value)
	if not cleaned:
		return "missing"
	slug = re.sub(r"[^A-Za-z0-9]+", "_", cleaned).strip("_").lower()
	return slug or "missing"


def _parse_period_label(label: str) -> tuple[int | None, str | None]:
	"""Parse year or year-month labels embedded in second-level headers."""

	cleaned = _clean_header_cell(label)
	month_match = MONTH_PATTERN.match(cleaned)
	if month_match is not None:
		return int(month_match.group("year")), f"{month_match.group('year')}-{month_match.group('month')}"
	year_match = YEAR_PATTERN.match(cleaned)
	if year_match is not None:
		return int(year_match.group("year")), str(year_match.group("year"))
	return None, None


def _first_sheet_name(excel: pd.ExcelFile, requested_sheet_name: str | None) -> str:
	"""Resolve the sheet name to use for one workbook read."""

	if requested_sheet_name is None:
		return str(excel.sheet_names[0])
	if requested_sheet_name not in excel.sheet_names:
		raise ValueError(
			f"Sheet {requested_sheet_name!r} not found. Available sheets: {excel.sheet_names}"
		)
	return requested_sheet_name


def load_variable_registry(codebook_path: str | Path) -> pd.DataFrame:
	"""Load the variable codebook into a normalized registry table."""

	resolved_path = Path(codebook_path).resolve()
	if not resolved_path.exists():
		raise FileNotFoundError(f"Variable codebook not found: {resolved_path}")

	frame = pd.read_excel(resolved_path)
	frame.columns = [_slugify_token(str(column_name)) for column_name in frame.columns]
	required_columns = {"variable", "definition", "source"}
	missing_columns = sorted(required_columns.difference(frame.columns))
	if missing_columns:
		raise ValueError(f"Codebook is missing required columns: {missing_columns}")

	registry = frame.loc[:, ["variable", "definition", "source"]].copy()
	registry = registry.rename(columns={"variable": "variable_code"})
	registry = registry.dropna(subset=["variable_code"]).copy()
	registry["variable_code"] = registry["variable_code"].astype(str).str.strip().str.upper()
	registry = registry.loc[registry["variable_code"].ne("") & registry["variable_code"].ne("NAN")].copy()
	registry["definition"] = registry["definition"].fillna("").astype(str).str.strip()
	registry["source"] = registry["source"].fillna("").astype(str).str.strip()
	registry = registry.drop_duplicates(subset=["variable_code"]).reset_index(drop=True)
	return registry


def load_workbook_frame(
	workbook_path: str | Path,
	sheet_name: str | None = None,
) -> tuple[pd.DataFrame, str]:
	"""Read the mixed-structure workbook using the first two rows as headers."""

	resolved_path = Path(workbook_path).resolve()
	if not resolved_path.exists():
		raise FileNotFoundError(f"Workbook not found: {resolved_path}")

	excel = pd.ExcelFile(resolved_path)
	resolved_sheet_name = _first_sheet_name(excel, sheet_name)
	frame = pd.read_excel(resolved_path, sheet_name=resolved_sheet_name, header=[0, 1])
	return frame, resolved_sheet_name


def flatten_multiindex_columns(
	columns: pd.MultiIndex,
	variable_registry: pd.DataFrame,
	global_key_count: int = 2,
) -> tuple[list[str], pd.DataFrame]:
	"""Flatten workbook headers and build a reusable column inventory."""

	known_codes = set(variable_registry["variable_code"].astype(str))
	flattened_names: list[str] = []
	inventory_rows: list[dict[str, Any]] = []
	seen_names: dict[str, int] = {}

	for column_index, pair in enumerate(columns):
		first_level = _clean_header_cell(pair[0])
		second_level = _clean_header_cell(pair[1])
		observed_block_code = _normalize_token(first_level)
		mapped_variable_code = VARIABLE_CODE_ALIAS_MAP.get(observed_block_code, observed_block_code)
		period_year, period_label = _parse_period_label(second_level)
		is_global_key = column_index < global_key_count

		if is_global_key:
			base_name = _slugify_token(first_level or second_level or f"key_{column_index}")
		elif first_level and second_level:
			base_name = f"{_slugify_token(first_level)}__{_slugify_token(second_level)}"
		else:
			base_name = _slugify_token(first_level or second_level or f"column_{column_index}")

		occurrence_index = seen_names.get(base_name, 0)
		seen_names[base_name] = occurrence_index + 1
		flattened_name = base_name if occurrence_index == 0 else f"{base_name}__dup{occurrence_index + 1}"
		flattened_names.append(flattened_name)

		inventory_rows.append(
			{
				"column_index": int(column_index),
				"flattened_name": flattened_name,
				"first_level": first_level,
				"second_level": second_level,
				"observed_block_code": observed_block_code,
				"mapped_variable_code": mapped_variable_code,
				"is_global_key": bool(is_global_key),
				"is_year_like_subcolumn": bool(period_label is not None),
				"period_year": period_year,
				"period_label": period_label,
				"codebook_match": bool(mapped_variable_code in known_codes),
			}
		)

	inventory = pd.DataFrame(inventory_rows)
	return flattened_names, inventory


def build_block_frame_map(frame: pd.DataFrame, inventory: pd.DataFrame) -> dict[str, pd.DataFrame]:
	"""Split the workbook into top-level block frames while keeping global keys."""

	global_key_columns = inventory.loc[inventory["is_global_key"], "flattened_name"].tolist()
	block_frames: dict[str, pd.DataFrame] = {}
	block_rows = inventory.loc[~inventory["is_global_key"]].copy()

	for observed_block_code, group in block_rows.groupby("observed_block_code", sort=True):
		if not observed_block_code:
			continue
		block_columns = global_key_columns + group["flattened_name"].tolist()
		block_frame = frame.loc[:, block_columns].copy()
		block_frame = block_frame.rename(
			columns={
				global_key_columns[0]: "global_year",
				global_key_columns[1]: "global_county",
			}
		)
		block_frame.insert(0, "workbook_row_index", np.arange(len(block_frame), dtype=np.int64))
		block_frame.insert(1, "observed_block_code", observed_block_code)
		block_frames[observed_block_code] = block_frame

	return block_frames


def build_block_summary(
	block_frames: dict[str, pd.DataFrame],
	inventory: pd.DataFrame,
	variable_registry: pd.DataFrame,
) -> pd.DataFrame:
	"""Summarize block-level structure and density for audit purposes."""

	registry_lookup = variable_registry.set_index("variable_code")["definition"].to_dict()
	rows: list[dict[str, Any]] = []

	for observed_block_code, block_frame in sorted(block_frames.items()):
		group = inventory.loc[inventory["observed_block_code"] == observed_block_code]
		value_columns = group["flattened_name"].tolist()
		value_frame = block_frame.loc[:, value_columns]
		numeric_frame = value_frame.apply(pd.to_numeric, errors="coerce")
		mapped_code = str(group["mapped_variable_code"].mode().iloc[0]) if not group.empty else observed_block_code
		rows.append(
			{
				"observed_block_code": observed_block_code,
				"mapped_variable_code": mapped_code,
				"codebook_match": bool(mapped_code in registry_lookup),
				"definition": registry_lookup.get(mapped_code, ""),
				"column_count": int(len(value_columns)),
				"row_count": int(len(block_frame)),
				"year_like_column_count": int(group["is_year_like_subcolumn"].sum()),
				"non_null_cell_share": float(value_frame.notna().mean().mean()),
				"numeric_cell_share": float(numeric_frame.notna().mean().mean()),
				"sample_second_level_labels": " | ".join(group["second_level"].head(8).astype(str).tolist()),
			}
		)

	return pd.DataFrame(rows).sort_values(["observed_block_code"]).reset_index(drop=True)


def build_year_long_block(
	block_frame: pd.DataFrame,
	block_inventory: pd.DataFrame,
) -> pd.DataFrame:
	"""Expand one year-like block into a sparse long table without a full-frame melt.

	Using DataFrame.melt on the full mixed block duplicates hundreds of object columns and
	can exceed workstation memory on the real workbook. The raw block CSV already preserves
	all descriptors, so the year-long export only keeps the minimal row identifiers plus the
	selected year-like value column metadata.
	"""

	year_columns = block_inventory.loc[
		block_inventory["is_year_like_subcolumn"],
		["flattened_name", "period_year", "period_label"],
	].copy()
	if len(year_columns) < 3:
		return pd.DataFrame()

	identifier_candidates = ["workbook_row_index", "observed_block_code", "global_year", "global_county"]
	identifier_columns = [column_name for column_name in identifier_candidates if column_name in block_frame.columns]
	year_frames: list[pd.DataFrame] = []

	for year_column in year_columns.itertuples(index=False):
		source_column = str(year_column.flattened_name)
		value_raw = block_frame[source_column]
		value_numeric = pd.to_numeric(value_raw, errors="coerce")
		non_empty_mask = value_raw.notna() | value_numeric.notna()
		if not bool(non_empty_mask.any()):
			continue

		piece = block_frame.loc[non_empty_mask, identifier_columns].copy()
		piece["source_column"] = source_column
		piece["period_year"] = year_column.period_year
		piece["period_label"] = year_column.period_label
		piece["value_raw"] = value_raw.loc[non_empty_mask].to_numpy(copy=False)
		piece["value_numeric"] = value_numeric.loc[non_empty_mask].astype(np.float32, copy=False).to_numpy(copy=False)
		year_frames.append(piece.reset_index(drop=True))

	if not year_frames:
		return pd.DataFrame()

	return pd.concat(year_frames, ignore_index=True)


def _parse_county_label(label: str) -> tuple[str, str]:
	"""Split one observed county label into entity code and entity name."""

	cleaned = str(label).strip()
	match = COUNTY_CODE_PATTERN.match(cleaned)
	if match is not None:
		return match.group("entity_code"), match.group("entity_name")
	return cleaned, cleaned


def _classify_entity_level(entity_code: str) -> str:
	"""Map one parsed entity code into a coarse granularity label."""

	cleaned = str(entity_code).strip().upper()
	if RCW_REGION_CODE_PATTERN.fullmatch(cleaned) is not None:
		return "region"
	if re.fullmatch(r"SE\d{2}", cleaned) is not None:
		return "macro_region"
	if re.fullmatch(r"\d{4}", cleaned) is not None:
		return "municipality"
	return "other"


def _coalesce_numeric_columns(frame: pd.DataFrame, candidate_columns: list[str]) -> pd.Series:
	"""Return the first non-null numeric value across several synonymous columns."""

	existing_columns = [column_name for column_name in candidate_columns if column_name in frame.columns]
	if not existing_columns:
		return pd.Series(np.nan, index=frame.index, dtype=float)
	numeric_frame = frame.loc[:, existing_columns].apply(pd.to_numeric, errors="coerce")
	return numeric_frame.bfill(axis=1).iloc[:, 0].astype(float)


def _canonicalize_entity_code(entity_code: str) -> str:
	"""Normalize one entity code across nearby workbook naming variants."""

	cleaned = str(entity_code).strip().upper()
	return ENTITY_CODE_ALIAS_MAP.get(cleaned, cleaned)


def _ascii_fold_text(value: str) -> str:
	"""Fold accents out of text so county-name joins tolerate workbook variants."""

	cleaned = _clean_header_cell(value)
	if not cleaned:
		return ""
	decomposed = unicodedata.normalize("NFKD", cleaned)
	return "".join(character for character in decomposed if not unicodedata.combining(character))


def _normalize_entity_name_key(label: str) -> str:
	"""Map one entity label to a county-name key that ignores formatting variants."""

	cleaned = _ascii_fold_text(label).lower()
	for token in ["the county of", "county of", "production county", " county", " region"]:
		cleaned = cleaned.replace(token, " ")
	cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
	return re.sub(r"\s+", " ", cleaned).strip()


def build_region_lookup(rcw_cleaned_panel: pd.DataFrame) -> dict[str, tuple[str, str]]:
	"""Build a name-based county lookup anchored on the stable RCW region panel."""

	if rcw_cleaned_panel.empty:
		return {}

	lookup: dict[str, tuple[str, str]] = {}
	unique_regions = rcw_cleaned_panel.loc[:, ["entity_code", "entity_name"]].drop_duplicates()
	for row in unique_regions.itertuples(index=False):
		entity_code = _canonicalize_entity_code(str(row.entity_code))
		entity_name = str(row.entity_name).strip()
		candidate_labels = {
			entity_name,
			f"{entity_name} county",
			f"the county of {entity_name}",
			f"{entity_name} region",
		}
		for candidate_label in candidate_labels:
			lookup[_normalize_entity_name_key(candidate_label)] = (entity_code, entity_name)
	return lookup


def _resolve_region_label(label: str, region_lookup: dict[str, tuple[str, str]]) -> tuple[str | None, str | None]:
	"""Resolve one coded or uncoded region label into the canonical RCW region id."""

	cleaned = str(label).strip()
	if not cleaned:
		return None, None

	match = COUNTY_CODE_PATTERN.match(cleaned)
	if match is not None:
		entity_code = _canonicalize_entity_code(match.group("entity_code"))
		entity_name = match.group("entity_name")
		lookup_match = region_lookup.get(_normalize_entity_name_key(entity_name))
		if lookup_match is not None:
			return lookup_match
		return entity_code, entity_name

	return region_lookup.get(_normalize_entity_name_key(cleaned), (None, None))


def _build_long_numeric_panel(
	frame: pd.DataFrame,
	year_column_map: dict[int, str],
	value_column_name: str,
	base_columns: list[str],
) -> pd.DataFrame:
	"""Expand one wide year map into a compact numeric long-form panel."""

	pieces: list[pd.DataFrame] = []
	for year, source_column in sorted(year_column_map.items()):
		if source_column not in frame.columns:
			continue
		piece = frame.loc[:, base_columns].copy()
		piece["year"] = int(year)
		piece[value_column_name] = pd.to_numeric(frame[source_column], errors="coerce")
		pieces.append(piece)

	if not pieces:
		return pd.DataFrame()

	result = pd.concat(pieces, ignore_index=True)
	return result.dropna(subset=[value_column_name]).reset_index(drop=True)


def _build_multi_value_long_panel(
	frame: pd.DataFrame,
	year_column_maps: dict[str, dict[int, str]],
	base_columns: list[str],
) -> pd.DataFrame:
	"""Expand several same-row yearly value maps into one long-form panel."""

	pieces: list[pd.DataFrame] = []
	all_years = sorted({year for year_column_map in year_column_maps.values() for year in year_column_map})
	for year in all_years:
		piece = frame.loc[:, base_columns].copy()
		piece["year"] = int(year)
		has_source_column = False
		for value_column_name, year_column_map in year_column_maps.items():
			source_column = year_column_map.get(year)
			if source_column is not None and source_column in frame.columns:
				piece[value_column_name] = pd.to_numeric(frame[source_column], errors="coerce")
				has_source_column = True
			else:
				piece[value_column_name] = np.nan
		if has_source_column:
			pieces.append(piece)

	if not pieces:
		return pd.DataFrame()

	result = pd.concat(pieces, ignore_index=True)
	value_columns = list(year_column_maps)
	return result.dropna(subset=value_columns, how="all").reset_index(drop=True)


def _prepare_segmented_region_panel(
	frame: pd.DataFrame,
	entity_label_column: str,
	region_lookup: dict[str, tuple[str, str]],
) -> pd.DataFrame:
	"""Resolve region ids from subtable headers that need forward-fill within one block."""

	if frame.empty or not region_lookup or entity_label_column not in frame.columns:
		return pd.DataFrame()

	panel = frame.copy()
	panel["_segment_entity_label"] = panel[entity_label_column].replace(r"^\s*$", np.nan, regex=True).ffill()
	panel[["entity_code", "entity_name"]] = panel["_segment_entity_label"].apply(
		lambda value: pd.Series(_resolve_region_label(value, region_lookup))
	)
	panel["entity_level"] = panel["entity_code"].fillna("").map(_classify_entity_level)
	return panel.loc[panel["entity_level"] == "region"].copy()


def build_td_candidate_panel(block_frame: pd.DataFrame) -> pd.DataFrame:
	"""Export the small TD slice that is internally coherent enough to audit downstream."""

	if block_frame.empty or "td__unnamed_1485_level_1" not in block_frame.columns:
		return pd.DataFrame()

	panel = block_frame.loc[block_frame["td__unnamed_1485_level_1"].isin(TD_MEASURE_LABEL_MAP)].copy()
	if panel.empty:
		return pd.DataFrame()

	panel[["entity_code", "entity_name"]] = panel["global_county"].apply(
		lambda value: pd.Series(_parse_county_label(value))
	)
	panel["entity_code"] = panel["entity_code"].map(_canonicalize_entity_code)
	panel["entity_level"] = panel["entity_code"].map(_classify_entity_level)
	panel = panel.loc[panel["entity_level"] == "region"].copy()
	if panel.empty:
		return pd.DataFrame()

	panel["measure_name"] = panel["td__unnamed_1485_level_1"].map(TD_MEASURE_LABEL_MAP)
	long_panel = _build_long_numeric_panel(
		frame=panel,
		year_column_map=TD_YEAR_COLUMN_MAP,
		value_column_name="value_numeric",
		base_columns=["entity_code", "entity_name", "measure_name"],
	)
	if long_panel.empty:
		return pd.DataFrame()

	aggregated = long_panel.groupby(["entity_code", "entity_name", "year", "measure_name"], as_index=False).agg(
		value_numeric=("value_numeric", "median"),
		candidate_count=("value_numeric", "count"),
	)
	wide_values = aggregated.pivot_table(
		index=["entity_code", "entity_name", "year"],
		columns="measure_name",
		values="value_numeric",
		aggfunc="first",
	)
	wide_counts = aggregated.pivot_table(
		index=["entity_code", "entity_name", "year"],
		columns="measure_name",
		values="candidate_count",
		aggfunc="first",
	)
	wide_values.columns = [str(column_name) for column_name in wide_values.columns]
	wide_counts.columns = [f"qa_{column_name}_candidate_count" for column_name in wide_counts.columns]
	result = wide_values.reset_index().merge(
		wide_counts.reset_index(),
		on=["entity_code", "entity_name", "year"],
		how="left",
	)
	return result.sort_values(["year", "entity_code"]).reset_index(drop=True)


def build_rydgdp_candidate_panel(
	block_frame: pd.DataFrame,
	region_lookup: dict[str, tuple[str, str]],
) -> pd.DataFrame:
	"""Build a region-level RYDGDP slice by reconstructing segmented entity headers."""

	if block_frame.empty or not region_lookup:
		return pd.DataFrame()

	numerator_panel = _prepare_segmented_region_panel(
		frame=block_frame,
		entity_label_column=RYDGDP_NUMERATOR_ENTITY_COLUMN,
		region_lookup=region_lookup,
	)
	if numerator_panel.empty or RYDGDP_NUMERATOR_DESCRIPTOR_COLUMN not in numerator_panel.columns:
		return pd.DataFrame()

	numerator_mask = numerator_panel[RYDGDP_NUMERATOR_DESCRIPTOR_COLUMN].fillna("").astype(str).str.strip().str.lower().eq(
		RYDGDP_DISPOSABLE_INCOME_LABEL
	)
	numerator_panel = numerator_panel.loc[numerator_mask].copy()
	if numerator_panel.empty:
		return pd.DataFrame()

	numerator_long = _build_multi_value_long_panel(
		frame=numerator_panel,
		year_column_maps={
			"rydgdp_disposable_income_msek_raw": RYDGDP_NUMERATOR_YEAR_COLUMN_MAP,
			"x_rydgdp_disposable_income_per_capita_raw": RYDGDP_NUMERATOR_PER_CAPITA_YEAR_COLUMN_MAP,
		},
		base_columns=["entity_code", "entity_name"],
	)
	if numerator_long.empty:
		return pd.DataFrame()

	numerator = numerator_long.groupby(["entity_code", "entity_name", "year"], as_index=False).agg(
		rydgdp_disposable_income_msek_raw=("rydgdp_disposable_income_msek_raw", "median"),
		x_rydgdp_disposable_income_per_capita_raw=("x_rydgdp_disposable_income_per_capita_raw", "median"),
		qa_rydgdp_numerator_candidate_count=("rydgdp_disposable_income_msek_raw", "count"),
		qa_rydgdp_numerator_per_capita_candidate_count=("x_rydgdp_disposable_income_per_capita_raw", "count"),
	)

	denominator_panel = _prepare_segmented_region_panel(
		frame=block_frame,
		entity_label_column=RYDGDP_DENOMINATOR_ENTITY_COLUMN,
		region_lookup=region_lookup,
	)
	if denominator_panel.empty:
		return pd.DataFrame()

	denominator_long = _build_multi_value_long_panel(
		frame=denominator_panel,
		year_column_maps={
			"rydgdp_grdp_msek_raw": RYDGDP_DENOMINATOR_YEAR_COLUMN_MAP,
			"x_rydgdp_grdp_per_capita_raw": RYDGDP_DENOMINATOR_PER_CAPITA_YEAR_COLUMN_MAP,
		},
		base_columns=["entity_code", "entity_name"],
	)
	if denominator_long.empty:
		return pd.DataFrame()

	denominator = denominator_long.groupby(["entity_code", "entity_name", "year"], as_index=False).agg(
		rydgdp_grdp_msek_raw=("rydgdp_grdp_msek_raw", "median"),
		x_rydgdp_grdp_per_capita_raw=("x_rydgdp_grdp_per_capita_raw", "median"),
		qa_rydgdp_denominator_candidate_count=("rydgdp_grdp_msek_raw", "count"),
		qa_rydgdp_denominator_per_capita_candidate_count=("x_rydgdp_grdp_per_capita_raw", "count"),
	)

	result = numerator.merge(denominator, on=["entity_code", "entity_name", "year"], how="inner")
	if result.empty:
		return pd.DataFrame()

	positive_denominator = result["rydgdp_grdp_msek_raw"].replace({0.0: np.nan})
	result["x_rydgdp_ratio_raw"] = result["rydgdp_disposable_income_msek_raw"] / positive_denominator
	return result.sort_values(["year", "entity_code"]).reset_index(drop=True)


def build_rpcyd_panel(
	block_frame: pd.DataFrame,
	block_inventory: pd.DataFrame,
	region_lookup: dict[str, tuple[str, str]],
) -> pd.DataFrame:
	"""Extract the stable RPCYD annual total-income-like region panel from the mixed block."""

	entity_label_column = "rpcyd__unnamed_143_level_1"
	if block_frame.empty or entity_label_column not in block_frame.columns or not region_lookup:
		return pd.DataFrame()

	panel = block_frame.copy()
	panel[["entity_code", "entity_name"]] = panel[entity_label_column].apply(
		lambda value: pd.Series(_resolve_region_label(value, region_lookup))
	)
	panel = panel.loc[panel["entity_code"].notna()].copy()
	if panel.empty:
		return pd.DataFrame()

	year_rows = block_inventory.loc[
		block_inventory["second_level"].astype(str).str.fullmatch(r"\d{4}"),
		["flattened_name", "period_year"],
	].dropna(subset=["period_year"]).drop_duplicates(subset=["period_year"]).sort_values(["period_year"])
	year_column_map: dict[int, str] = {}
	for year_row in year_rows.itertuples(index=False):
		source_column = str(year_row.flattened_name)
		if source_column not in panel.columns:
			continue
		column_position = panel.columns.get_loc(source_column)
		if column_position + 2 >= len(panel.columns):
			continue
		year_column_map[int(year_row.period_year)] = str(panel.columns[column_position + 2])

	if not year_column_map:
		return pd.DataFrame()

	long_panel = _build_long_numeric_panel(
		frame=panel,
		year_column_map=year_column_map,
		value_column_name="x_rpcyd_total_income_raw",
		base_columns=["entity_code", "entity_name", entity_label_column],
	)
	if long_panel.empty:
		return pd.DataFrame()

	result = long_panel.groupby(["entity_code", "entity_name", "year"], as_index=False).agg(
		x_rpcyd_total_income_raw=("x_rpcyd_total_income_raw", "median"),
		qa_rpcyd_candidate_count=("x_rpcyd_total_income_raw", "count"),
	)
	return result.sort_values(["year", "entity_code"]).reset_index(drop=True)


def build_informal_merged_long_panel(
	rcw_cleaned_panel: pd.DataFrame,
	rpcyd_panel: pd.DataFrame,
	rydgdp_candidate_panel: pd.DataFrame,
	td_candidate_panel: pd.DataFrame,
) -> pd.DataFrame:
	"""Merge the stable RCW region panel with additional formal-process candidates."""

	if rcw_cleaned_panel.empty:
		return pd.DataFrame()

	merged = rcw_cleaned_panel.copy()
	if not rpcyd_panel.empty:
		merged = merged.merge(
			rpcyd_panel,
			on=["entity_code", "entity_name", "year"],
			how="left",
		)
	if not rydgdp_candidate_panel.empty:
		merged = merged.merge(
			rydgdp_candidate_panel,
			on=["entity_code", "entity_name", "year"],
			how="left",
		)
	if not td_candidate_panel.empty:
		merged = merged.merge(
			td_candidate_panel,
			on=["entity_code", "entity_name", "year"],
			how="left",
		)

	formal_feature_columns = [
		column_name
		for column_name in [
			"x_rpcyd_total_income_raw",
			"rydgdp_disposable_income_msek_raw",
			"x_rydgdp_disposable_income_per_capita_raw",
			"rydgdp_grdp_msek_raw",
			"x_rydgdp_grdp_per_capita_raw",
			"x_rydgdp_ratio_raw",
			"x_td_direct_taxes_on_labour_raw",
			"x_td_indirect_taxes_on_labour_raw",
			"x_td_taxes_on_capital_raw",
			"x_td_total_tax_revenues_raw",
		]
		if column_name in merged.columns
	]
	if formal_feature_columns:
		merged["qa_formal_feature_count"] = merged[formal_feature_columns].notna().sum(axis=1).astype(int)
	else:
		merged["qa_formal_feature_count"] = 0

	return merged.sort_values(["year", "entity_code"]).reset_index(drop=True)


def _build_informal_acgate_base_panel(merged_long_panel: pd.DataFrame) -> pd.DataFrame:
	"""Project the merged panel into a common AC-GATE input table with legacy and curated columns."""

	if merged_long_panel.empty or "x_rpcyd_total_income_raw" not in merged_long_panel.columns:
		return pd.DataFrame()

	ready = merged_long_panel.loc[
		merged_long_panel["y_rcv_total_raw"].notna() & merged_long_panel["x_rpcyd_total_income_raw"].notna()
	].copy()
	if ready.empty:
		return pd.DataFrame()

	ready["y_t"] = pd.to_numeric(ready["y_rcv_total_raw"], errors="coerce")
	ready["x_t"] = pd.to_numeric(ready["x_rpcyd_total_income_raw"], errors="coerce")
	ready["proxy_atm_count"] = pd.to_numeric(ready["proxy_atm_count_raw"], errors="coerce")
	ready["proxy_terminal_count"] = pd.to_numeric(ready["proxy_terminal_count_raw"], errors="coerce")
	ready["proxy_credit_card_depth"] = pd.to_numeric(ready["x_credit_cards_raw"], errors="coerce")
	ready["static_noncash_value"] = pd.to_numeric(ready["static_noncash_transaction_value_raw"], errors="coerce")
	ready["static_withdrawal_value"] = pd.to_numeric(ready["proxy_withdrawal_value_raw"], errors="coerce")

	for source_column, target_column in CURATED_FORMAL_SEQUENCE_RENAME_MAP.items():
		if source_column in ready.columns:
			ready[target_column] = pd.to_numeric(ready[source_column], errors="coerce")

	if "x_td_direct_taxes_on_labour_raw" in ready.columns:
		ready["optional_x_td_direct_taxes_on_labour"] = pd.to_numeric(
			ready["x_td_direct_taxes_on_labour_raw"],
			errors="coerce",
		)
	if "x_td_indirect_taxes_on_labour_raw" in ready.columns:
		ready["optional_x_td_indirect_taxes_on_labour"] = pd.to_numeric(
			ready["x_td_indirect_taxes_on_labour_raw"],
			errors="coerce",
		)
	if "x_td_taxes_on_capital_raw" in ready.columns:
		ready["optional_x_td_taxes_on_capital"] = pd.to_numeric(
			ready["x_td_taxes_on_capital_raw"],
			errors="coerce",
		)
	if "x_td_total_tax_revenues_raw" in ready.columns:
		ready["optional_x_td_total_tax_revenues"] = pd.to_numeric(
			ready["x_td_total_tax_revenues_raw"],
			errors="coerce",
		)

	curated_sequence_columns = [
		target_column for target_column in CURATED_FORMAL_SEQUENCE_RENAME_MAP.values() if target_column in ready.columns
	]
	ready["qa_curated_formal_feature_count"] = (
		ready[curated_sequence_columns].notna().sum(axis=1).astype(int) if curated_sequence_columns else 0
	)
	return ready.sort_values(["year", "entity_code"]).reset_index(drop=True)


def build_informal_acgate_ready_panel(merged_long_panel: pd.DataFrame) -> pd.DataFrame:
	"""Project the merged panel into a multivariate AC-GATE-ready CSV with a legacy x_t alias."""

	ready = _build_informal_acgate_base_panel(merged_long_panel)
	if ready.empty:
		return pd.DataFrame()

	curated_sequence_columns = [
		target_column for target_column in CURATED_FORMAL_SEQUENCE_RENAME_MAP.values() if target_column in ready.columns
	]
	selected_columns = [
		"year",
		"entity_code",
		"entity_name",
		"y_t",
		"x_t",
	] + curated_sequence_columns + [
		"proxy_atm_count",
		"proxy_terminal_count",
		"proxy_credit_card_depth",
		"static_noncash_value",
		"static_withdrawal_value",
		"duplicate_row_count",
		"conflicted_measure_count",
		"qa_available_feature_count",
		"qa_formal_feature_count",
		"qa_curated_formal_feature_count",
	]
	optional_columns = [
		column_name
		for column_name in [
			"qa_rpcyd_candidate_count",
			"qa_rydgdp_numerator_candidate_count",
			"qa_rydgdp_numerator_per_capita_candidate_count",
			"qa_rydgdp_denominator_candidate_count",
			"qa_rydgdp_denominator_per_capita_candidate_count",
			"optional_x_td_direct_taxes_on_labour",
			"optional_x_td_indirect_taxes_on_labour",
			"optional_x_td_taxes_on_capital",
			"optional_x_td_total_tax_revenues",
		]
		if column_name in ready.columns
	]
	return ready.loc[:, selected_columns + optional_columns].sort_values(["year", "entity_code"]).reset_index(drop=True)


def build_informal_acgate_single_feature_panel(merged_long_panel: pd.DataFrame) -> pd.DataFrame:
	"""Project the merged panel into the original single-formal-feature AC-GATE slice."""

	ready = _build_informal_acgate_base_panel(merged_long_panel)
	if ready.empty:
		return pd.DataFrame()

	selected_columns = [
		"year",
		"entity_code",
		"entity_name",
		"y_t",
		"x_t",
		"proxy_atm_count",
		"proxy_terminal_count",
		"proxy_credit_card_depth",
		"static_noncash_value",
		"static_withdrawal_value",
		"duplicate_row_count",
		"conflicted_measure_count",
		"qa_available_feature_count",
		"qa_formal_feature_count",
	]
	optional_columns = [
		column_name
		for column_name in [
			"x_seq_rydgdp_ratio",
			"optional_x_td_direct_taxes_on_labour",
		]
		if column_name in ready.columns
	]
	return ready.loc[:, selected_columns + optional_columns].sort_values(["year", "entity_code"]).reset_index(drop=True)


def build_informal_acgate_multiseq_overlap_panel(merged_long_panel: pd.DataFrame) -> pd.DataFrame:
	"""Keep only rows where the curated multivariate formal sequence is fully observed."""

	ready = build_informal_acgate_ready_panel(merged_long_panel)
	if ready.empty:
		return pd.DataFrame()

	curated_sequence_columns = [
		target_column for target_column in CURATED_FORMAL_SEQUENCE_RENAME_MAP.values() if target_column in ready.columns
	]
	if not curated_sequence_columns:
		return pd.DataFrame()

	overlap_ready = ready.loc[ready[curated_sequence_columns].notna().all(axis=1)].copy()
	return overlap_ready.sort_values(["year", "entity_code"]).reset_index(drop=True)


def build_rcw_candidate_panel(block_frame: pd.DataFrame, min_duplicate_count: int = 1) -> pd.DataFrame:
	"""Aggregate the RCW block into a first region-year candidate panel."""

	panel_frame = block_frame.copy()
	panel_frame["global_year"] = pd.to_numeric(panel_frame["global_year"], errors="coerce")
	panel_frame["global_county"] = panel_frame["global_county"].astype(str).str.strip()
	panel_frame = panel_frame.dropna(subset=["global_year"])
	panel_frame = panel_frame.loc[panel_frame["global_county"] != ""].reset_index(drop=True)
	panel_frame["year"] = panel_frame["global_year"].astype(int)
	panel_frame[["entity_code", "entity_name"]] = panel_frame["global_county"].apply(
		lambda value: pd.Series(_parse_county_label(value))
	)

	measure_columns = [column_name for column_name in panel_frame.columns if column_name.startswith("rcw__")]
	measure_rename = {
		column_name: f"rcw_{_slugify_token(column_name.split('__', maxsplit=1)[1])}"
		for column_name in measure_columns
	}
	for column_name in measure_columns:
		panel_frame[column_name] = pd.to_numeric(panel_frame[column_name], errors="coerce")

	records: list[dict[str, Any]] = []
	group_columns = ["year", "entity_code", "entity_name"]
	for keys, group in panel_frame.groupby(group_columns, sort=True):
		duplicate_row_count = int(len(group))
		if duplicate_row_count < min_duplicate_count:
			continue
		record: dict[str, Any] = {
			"year": int(keys[0]),
			"entity_code": str(keys[1]),
			"entity_name": str(keys[2]),
			"duplicate_row_count": duplicate_row_count,
		}
		conflicted_measure_count = 0
		for source_column, target_column in measure_rename.items():
			values = group[source_column].dropna().to_numpy(dtype=float)
			unique_values = np.unique(values) if values.size else np.array([], dtype=float)
			if unique_values.size > 1:
				conflicted_measure_count += 1
			record[target_column] = float(unique_values[0]) if unique_values.size else np.nan
		record["conflicted_measure_count"] = conflicted_measure_count
		records.append(record)

	if not records:
		return pd.DataFrame()

	result = pd.DataFrame(records).sort_values(["year", "entity_code"]).reset_index(drop=True)
	return result


def build_rcw_cleaned_panel(rcw_candidate_panel: pd.DataFrame) -> pd.DataFrame:
	"""Build a county-level RCW panel with canonical AC-GATE-oriented raw columns."""

	if rcw_candidate_panel.empty:
		return pd.DataFrame()

	panel = rcw_candidate_panel.copy()
	panel["entity_level"] = panel["entity_code"].map(_classify_entity_level)
	panel = panel.loc[panel["entity_level"] == "region"].copy()
	if panel.empty:
		return pd.DataFrame()

	panel["y_rcv_total_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_total_dup2", "rcw_total", "rcw_total_1_dup2", "rcw_total_1"],
	)
	panel["x_cards_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_cards", "rcw_card_dup2", "rcw_card", "rcw_card_1"],
	)
	panel["x_bank_cards_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_bank_card", "rcw_bank_card_1", "rcw_debet_cards", "rcw_debet_cards_1"],
	)
	panel["x_credit_cards_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_credit_card", "rcw_credit_card_1", "rcw_credit_cards", "rcw_credit_cards_1"],
	)
	panel["x_credit_transfers_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_credit_transfers", "rcw_credit_transfers_1", "rcw_transfers", "rcw_gireringar"],
	)
	panel["x_electronic_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_electronic_dup2", "rcw_electronic", "rcw_electronic_1_dup2", "rcw_electronic_1"],
	)
	panel["x_direct_debit_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_direct_debet", "rcw_direct_debet_1", "rcw_autogiro", "rcw_autogiro_1"],
	)
	panel["x_paper_based_raw"] = _coalesce_numeric_columns(
		panel,
		[
			"rcw_paper_based_form",
			"rcw_paper_based_form_1",
			"rcw_check",
			"rcw_check_1",
			"rcw_cheques_including_bank_drafts",
			"rcw_cheques_including_bank_drafts_1",
		],
	)
	panel["proxy_atm_count_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_number_of_atms", "rcw_amount_of_atm_s"],
	)
	panel["proxy_terminal_count_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_number_of_payment_terminals", "rcw_amount_of_terminals", "rcw_payment_terminals"],
	)
	panel["proxy_withdrawal_transactions_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_number_of_transactions_withdrawals_millions"],
	)
	panel["proxy_withdrawal_value_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_transaction_value_withdrawals_sek_billions"],
	)
	panel["static_noncash_transaction_count_raw"] = _coalesce_numeric_columns(
		panel,
		[
			"rcw_number_of_transactions_millions",
			"rcw_amount_of_transactions_millions_dup2",
			"rcw_amount_of_transactions_millions",
		],
	)
	panel["static_noncash_transaction_value_raw"] = _coalesce_numeric_columns(
		panel,
		["rcw_transaction_value_sek_billions", "rcw_transaction_value_billions_of_kronor"],
	)
	panel["qa_has_conflict"] = panel["conflicted_measure_count"].fillna(0).astype(int) > 0
	feature_columns = [
		"y_rcv_total_raw",
		"x_cards_raw",
		"x_bank_cards_raw",
		"x_credit_cards_raw",
		"x_credit_transfers_raw",
		"x_electronic_raw",
		"x_direct_debit_raw",
		"x_paper_based_raw",
		"proxy_atm_count_raw",
		"proxy_terminal_count_raw",
		"proxy_withdrawal_transactions_raw",
		"proxy_withdrawal_value_raw",
		"static_noncash_transaction_count_raw",
		"static_noncash_transaction_value_raw",
	]
	panel["qa_available_feature_count"] = panel[feature_columns].notna().sum(axis=1).astype(int)

	base_columns = [
		"year",
		"entity_code",
		"entity_name",
		"entity_level",
		"duplicate_row_count",
		"conflicted_measure_count",
		"qa_has_conflict",
		"qa_available_feature_count",
	]
	selected_columns = base_columns + feature_columns
	result = panel.loc[panel["y_rcv_total_raw"].notna(), selected_columns].copy()
	return result.sort_values(["year", "entity_code"]).reset_index(drop=True)


def build_rcw_model_ready_panel(rcw_cleaned_panel: pd.DataFrame) -> pd.DataFrame:
	"""Reduce the cleaned RCW slice into a compact AC-GATE-oriented panel."""

	if rcw_cleaned_panel.empty:
		return pd.DataFrame()

	model_ready = rcw_cleaned_panel.copy()
	rename_map = {
		"y_rcv_total_raw": "y_t",
		"x_cards_raw": "x_cards",
		"x_credit_transfers_raw": "x_credit_transfers",
		"x_electronic_raw": "x_electronic",
		"x_direct_debit_raw": "x_direct_debit",
		"proxy_atm_count_raw": "proxy_atm_count",
		"proxy_terminal_count_raw": "proxy_terminal_count",
		"x_credit_cards_raw": "proxy_credit_card_depth",
		"static_noncash_transaction_value_raw": "static_noncash_value",
		"proxy_withdrawal_value_raw": "static_withdrawal_value",
	}
	base_columns = [
		"year",
		"entity_code",
		"entity_name",
		"duplicate_row_count",
		"conflicted_measure_count",
		"qa_available_feature_count",
	]
	selected_columns = base_columns + list(rename_map)
	model_ready = model_ready.loc[:, selected_columns].rename(columns=rename_map)
	return model_ready.sort_values(["year", "entity_code"]).reset_index(drop=True)


def save_informal_exports(
	workbook_path: str | Path = DEFAULT_WORKBOOK_PATH,
	codebook_path: str | Path = DEFAULT_CODEBOOK_PATH,
	output_dir: str | Path = DEFAULT_OUTPUT_DIR,
	sheet_name: str | None = None,
	rcw_min_duplicate_count: int = 1,
) -> InformalExportResult:
	"""Audit the workbook and export reusable CSV artifacts."""

	destination_dir = Path(output_dir).resolve()
	destination_dir.mkdir(parents=True, exist_ok=True)
	block_output_dir = destination_dir / BLOCK_OUTPUT_DIRNAME
	block_output_dir.mkdir(parents=True, exist_ok=True)
	acgate_output_dir = destination_dir / ACGATE_OUTPUT_DIRNAME
	acgate_output_dir.mkdir(parents=True, exist_ok=True)

	variable_registry = load_variable_registry(codebook_path)
	raw_frame, resolved_sheet_name = load_workbook_frame(workbook_path=workbook_path, sheet_name=sheet_name)
	flattened_names, column_inventory = flatten_multiindex_columns(raw_frame.columns, variable_registry)
	frame = raw_frame.copy()
	frame.columns = flattened_names
	block_frames = build_block_frame_map(frame, column_inventory)
	block_summary = build_block_summary(block_frames, column_inventory, variable_registry)

	codebook_registry_path = destination_dir / "informal_codebook_registry.csv"
	column_inventory_path = destination_dir / "informal_column_inventory.csv"
	block_summary_path = destination_dir / "informal_block_summary.csv"
	audit_json_path = destination_dir / "informal_workbook_audit.json"

	variable_registry.to_csv(codebook_registry_path, index=False)
	column_inventory.to_csv(column_inventory_path, index=False)
	block_summary.to_csv(block_summary_path, index=False)

	raw_block_paths: dict[str, Path] = {}
	year_long_block_paths: dict[str, Path] = {}
	for observed_block_code, block_frame in sorted(block_frames.items()):
		raw_path = block_output_dir / f"{observed_block_code.lower()}_block_raw.csv"
		block_frame.to_csv(raw_path, index=False)
		raw_block_paths[observed_block_code] = raw_path

		block_inventory = column_inventory.loc[column_inventory["observed_block_code"] == observed_block_code]
		year_long_frame = build_year_long_block(block_frame, block_inventory)
		if not year_long_frame.empty:
			year_long_path = block_output_dir / f"{observed_block_code.lower()}_block_year_long.csv"
			year_long_frame.to_csv(year_long_path, index=False)
			year_long_block_paths[observed_block_code] = year_long_path

	rcw_candidate_panel_path: Path | None = None
	rcw_cleaned_panel_path: Path | None = None
	rcw_model_ready_path: Path | None = None
	td_candidate_panel_path: Path | None = None
	rydgdp_candidate_panel_path: Path | None = None
	rpcyd_panel_path: Path | None = None
	merged_long_panel_path: Path | None = None
	acgate_ready_panel_path: Path | None = None
	acgate_single_feature_panel_path: Path | None = None
	acgate_multiseq_panel_path: Path | None = None
	acgate_multiseq_overlap_panel_path: Path | None = None
	acgate_manifest_path: Path | None = None
	rcw_cleaned_panel = pd.DataFrame()
	if "RCW" in block_frames:
		rcw_candidate_panel = build_rcw_candidate_panel(
			block_frames["RCW"],
			min_duplicate_count=rcw_min_duplicate_count,
		)
		if not rcw_candidate_panel.empty:
			rcw_candidate_panel_path = destination_dir / "informal_rcw_candidate_panel.csv"
			rcw_candidate_panel.to_csv(rcw_candidate_panel_path, index=False)

			rcw_cleaned_panel = build_rcw_cleaned_panel(rcw_candidate_panel)
			if not rcw_cleaned_panel.empty:
				rcw_cleaned_panel_path = destination_dir / "informal_rcw_cleaned_panel.csv"
				rcw_cleaned_panel.to_csv(rcw_cleaned_panel_path, index=False)

				rcw_model_ready_panel = build_rcw_model_ready_panel(rcw_cleaned_panel)
				if not rcw_model_ready_panel.empty:
					rcw_model_ready_path = destination_dir / "informal_rcw_model_ready_panel.csv"
					rcw_model_ready_panel.to_csv(rcw_model_ready_path, index=False)

	region_lookup = build_region_lookup(rcw_cleaned_panel)

	if "TD" in block_frames:
		td_candidate_panel = build_td_candidate_panel(block_frames["TD"])
		if not td_candidate_panel.empty:
			td_candidate_panel_path = destination_dir / "informal_td_candidate_panel.csv"
			td_candidate_panel.to_csv(td_candidate_panel_path, index=False)

	if "RYDGDP" in block_frames:
		rydgdp_candidate_panel = build_rydgdp_candidate_panel(block_frames["RYDGDP"], region_lookup)
		if not rydgdp_candidate_panel.empty:
			rydgdp_candidate_panel_path = destination_dir / "informal_rydgdp_candidate_panel.csv"
			rydgdp_candidate_panel.to_csv(rydgdp_candidate_panel_path, index=False)

	if "RPCYD" in block_frames:
		rpcyd_inventory = column_inventory.loc[column_inventory["observed_block_code"] == "RPCYD"]
		rpcyd_panel = build_rpcyd_panel(block_frames["RPCYD"], rpcyd_inventory, region_lookup)
		if not rpcyd_panel.empty:
			rpcyd_panel_path = destination_dir / "informal_rpcyd_panel.csv"
			rpcyd_panel.to_csv(rpcyd_panel_path, index=False)

	if not rcw_cleaned_panel.empty:
		merged_long_panel = build_informal_merged_long_panel(
			rcw_cleaned_panel=rcw_cleaned_panel,
			rpcyd_panel=pd.read_csv(rpcyd_panel_path) if rpcyd_panel_path is not None else pd.DataFrame(),
			rydgdp_candidate_panel=pd.read_csv(rydgdp_candidate_panel_path) if rydgdp_candidate_panel_path is not None else pd.DataFrame(),
			td_candidate_panel=pd.read_csv(td_candidate_panel_path) if td_candidate_panel_path is not None else pd.DataFrame(),
		)
		if not merged_long_panel.empty:
			merged_long_panel_path = destination_dir / "informal_formal_merged_long.csv"
			merged_long_panel.to_csv(merged_long_panel_path, index=False)

			acgate_single_feature_panel = build_informal_acgate_single_feature_panel(merged_long_panel)
			if not acgate_single_feature_panel.empty:
				acgate_single_feature_panel_path = acgate_output_dir / "informal_acgate_single_feature_ready.csv"
				acgate_single_feature_panel.to_csv(acgate_single_feature_panel_path, index=False)

			acgate_ready_panel = build_informal_acgate_ready_panel(merged_long_panel)
			if not acgate_ready_panel.empty:
				acgate_ready_panel_path = destination_dir / "informal_acgate_ready_panel.csv"
				acgate_ready_panel.to_csv(acgate_ready_panel_path, index=False)
				acgate_multiseq_panel_path = acgate_output_dir / "informal_acgate_multiseq_ready.csv"
				acgate_ready_panel.to_csv(acgate_multiseq_panel_path, index=False)

			acgate_multiseq_overlap_panel = build_informal_acgate_multiseq_overlap_panel(merged_long_panel)
			if not acgate_multiseq_overlap_panel.empty:
				acgate_multiseq_overlap_panel_path = acgate_output_dir / "informal_acgate_multiseq_overlap_ready.csv"
				acgate_multiseq_overlap_panel.to_csv(acgate_multiseq_overlap_panel_path, index=False)

			if acgate_multiseq_panel_path is not None or acgate_single_feature_panel_path is not None:
				acgate_manifest_path = acgate_output_dir / "informal_acgate_manifest.json"
				manifest_payload = {
					"acgate_output_dir": str(acgate_output_dir),
					"single_feature_ready_path": None if acgate_single_feature_panel_path is None else str(acgate_single_feature_panel_path),
					"multiseq_ready_path": None if acgate_multiseq_panel_path is None else str(acgate_multiseq_panel_path),
					"multiseq_overlap_ready_path": None if acgate_multiseq_overlap_panel_path is None else str(acgate_multiseq_overlap_panel_path),
					"legacy_primary_sequence_column": "x_t",
					"sequence_columns": [
						column_name
						for column_name in ["x_t", *CURATED_FORMAL_SEQUENCE_RENAME_MAP.values()]
						if acgate_ready_panel_path is not None and column_name in acgate_ready_panel.columns
					],
					"proxy_columns": [
						column_name
						for column_name in ["proxy_atm_count", "proxy_terminal_count", "proxy_credit_card_depth"]
						if acgate_ready_panel_path is not None and column_name in acgate_ready_panel.columns
					],
					"static_columns": [
						column_name
						for column_name in ["static_noncash_value", "static_withdrawal_value"]
						if acgate_ready_panel_path is not None and column_name in acgate_ready_panel.columns
					],
					"full_panel_rows": 0 if acgate_ready_panel_path is None else int(len(acgate_ready_panel)),
					"full_panel_entities": 0 if acgate_ready_panel_path is None else int(acgate_ready_panel["entity_code"].nunique()),
					"full_panel_year_span": None if acgate_ready_panel_path is None else [int(acgate_ready_panel["year"].min()), int(acgate_ready_panel["year"].max())],
					"overlap_panel_rows": 0 if acgate_multiseq_overlap_panel_path is None else int(len(acgate_multiseq_overlap_panel)),
					"overlap_panel_entities": 0 if acgate_multiseq_overlap_panel_path is None else int(acgate_multiseq_overlap_panel["entity_code"].nunique()),
					"overlap_panel_year_span": None if acgate_multiseq_overlap_panel_path is None else [int(acgate_multiseq_overlap_panel["year"].min()), int(acgate_multiseq_overlap_panel["year"].max())],
					"td_status": "TD remains an audit-only sparse candidate and is excluded from the curated multiseq bundle.",
					"knn_recommendation": "Do not impute workbook parsing gaps here. If you later need KNN, fit it year-wise or on the train/reference window inside the loader to avoid temporal leakage.",
				}
				with acgate_manifest_path.open("w", encoding="utf-8") as handle:
					json.dump(manifest_payload, handle, ensure_ascii=False, indent=2)

	known_codes = set(variable_registry["variable_code"].dropna().astype(str))
	observed_codes = sorted(code for code in block_frames if code)
	mapped_codes = sorted(
		set(column_inventory.loc[~column_inventory["is_global_key"], "mapped_variable_code"].dropna().astype(str))
	)
	audit_payload = {
		"workbook_path": str(Path(workbook_path).resolve()),
		"sheet_name": resolved_sheet_name,
		"row_count": int(len(frame)),
		"column_count": int(frame.shape[1]),
		"observed_block_codes": observed_codes,
		"mapped_variable_codes": mapped_codes,
		"codebook_codes_not_observed": sorted(known_codes.difference(mapped_codes)),
		"observed_codes_not_in_codebook": sorted(set(mapped_codes).difference(known_codes)),
		"raw_block_paths": {key: str(path) for key, path in raw_block_paths.items()},
		"year_long_block_paths": {key: str(path) for key, path in year_long_block_paths.items()},
		"rcw_candidate_panel_path": None if rcw_candidate_panel_path is None else str(rcw_candidate_panel_path),
		"rcw_cleaned_panel_path": None if rcw_cleaned_panel_path is None else str(rcw_cleaned_panel_path),
		"rcw_model_ready_path": None if rcw_model_ready_path is None else str(rcw_model_ready_path),
		"td_candidate_panel_path": None if td_candidate_panel_path is None else str(td_candidate_panel_path),
		"rydgdp_candidate_panel_path": None if rydgdp_candidate_panel_path is None else str(rydgdp_candidate_panel_path),
		"rpcyd_panel_path": None if rpcyd_panel_path is None else str(rpcyd_panel_path),
		"merged_long_panel_path": None if merged_long_panel_path is None else str(merged_long_panel_path),
		"acgate_ready_panel_path": None if acgate_ready_panel_path is None else str(acgate_ready_panel_path),
		"acgate_output_dir": str(acgate_output_dir),
		"acgate_single_feature_panel_path": None if acgate_single_feature_panel_path is None else str(acgate_single_feature_panel_path),
		"acgate_multiseq_panel_path": None if acgate_multiseq_panel_path is None else str(acgate_multiseq_panel_path),
		"acgate_multiseq_overlap_panel_path": None if acgate_multiseq_overlap_panel_path is None else str(acgate_multiseq_overlap_panel_path),
		"acgate_manifest_path": None if acgate_manifest_path is None else str(acgate_manifest_path),
	}
	with audit_json_path.open("w", encoding="utf-8") as handle:
		json.dump(audit_payload, handle, ensure_ascii=False, indent=2)

	return InformalExportResult(
		audit_json_path=audit_json_path,
		column_inventory_path=column_inventory_path,
		codebook_registry_path=codebook_registry_path,
		block_summary_path=block_summary_path,
		rcw_candidate_panel_path=rcw_candidate_panel_path,
		rcw_cleaned_panel_path=rcw_cleaned_panel_path,
		rcw_model_ready_path=rcw_model_ready_path,
		td_candidate_panel_path=td_candidate_panel_path,
		rydgdp_candidate_panel_path=rydgdp_candidate_panel_path,
		rpcyd_panel_path=rpcyd_panel_path,
		merged_long_panel_path=merged_long_panel_path,
		acgate_ready_panel_path=acgate_ready_panel_path,
		acgate_output_dir=acgate_output_dir,
		acgate_single_feature_panel_path=acgate_single_feature_panel_path,
		acgate_multiseq_panel_path=acgate_multiseq_panel_path,
		acgate_multiseq_overlap_panel_path=acgate_multiseq_overlap_panel_path,
		acgate_manifest_path=acgate_manifest_path,
		raw_block_paths=raw_block_paths,
		year_long_block_paths=year_long_block_paths,
	)


def main() -> None:
	"""Run the workbook audit/export flow and print a compact summary."""

	args = parse_args()
	result = save_informal_exports(
		workbook_path=args.workbook_path,
		codebook_path=args.codebook_path,
		output_dir=args.output_dir,
		sheet_name=args.sheet_name,
		rcw_min_duplicate_count=args.rcw_min_duplicate_count,
	)

	block_count = len(result.raw_block_paths)
	year_long_count = len(result.year_long_block_paths)
	print(f"Saved informal workbook audit to: {result.audit_json_path}")
	print(f"Saved column inventory to: {result.column_inventory_path}")
	print(f"Saved block summary to: {result.block_summary_path}")
	print(f"Exported {block_count} raw block CSV files and {year_long_count} year-long block CSV files.")
	if result.rcw_candidate_panel_path is not None:
		print(f"Saved RCW candidate panel to: {result.rcw_candidate_panel_path}")
	else:
		print("RCW candidate panel was not generated.")
	if result.rcw_cleaned_panel_path is not None:
		print(f"Saved RCW cleaned panel to: {result.rcw_cleaned_panel_path}")
	if result.rcw_model_ready_path is not None:
		print(f"Saved RCW model-ready panel to: {result.rcw_model_ready_path}")
	if result.td_candidate_panel_path is not None:
		print(f"Saved TD candidate panel to: {result.td_candidate_panel_path}")
	if result.rydgdp_candidate_panel_path is not None:
		print(f"Saved RYDGDP candidate panel to: {result.rydgdp_candidate_panel_path}")
	if result.rpcyd_panel_path is not None:
		print(f"Saved RPCYD panel to: {result.rpcyd_panel_path}")
	if result.merged_long_panel_path is not None:
		print(f"Saved merged informal long panel to: {result.merged_long_panel_path}")
	if result.acgate_ready_panel_path is not None:
		print(f"Saved AC-GATE-ready panel to: {result.acgate_ready_panel_path}")
	if result.acgate_single_feature_panel_path is not None:
		print(f"Saved AC-GATE single-feature panel to: {result.acgate_single_feature_panel_path}")
	if result.acgate_multiseq_panel_path is not None:
		print(f"Saved AC-GATE multiseq panel to: {result.acgate_multiseq_panel_path}")
	if result.acgate_multiseq_overlap_panel_path is not None:
		print(f"Saved AC-GATE multiseq overlap panel to: {result.acgate_multiseq_overlap_panel_path}")
	if result.acgate_manifest_path is not None:
		print(f"Saved AC-GATE manifest to: {result.acgate_manifest_path}")


BLOCK_OUTPUT_DIRNAME = "blocks"


if __name__ == "__main__":
	main()


__all__ = [
	"DEFAULT_BLOCK_OUTPUT_DIR",
	"DEFAULT_CODEBOOK_PATH",
	"DEFAULT_OUTPUT_DIR",
	"DEFAULT_WORKBOOK_PATH",
	"InformalExportResult",
	"build_block_frame_map",
	"build_block_summary",
	"build_informal_acgate_ready_panel",
	"build_informal_acgate_multiseq_overlap_panel",
	"build_informal_acgate_single_feature_panel",
	"build_informal_merged_long_panel",
	"build_region_lookup",
	"build_rcw_cleaned_panel",
	"build_rcw_candidate_panel",
	"build_rcw_model_ready_panel",
	"build_rpcyd_panel",
	"build_rydgdp_candidate_panel",
	"build_td_candidate_panel",
	"build_year_long_block",
	"flatten_multiindex_columns",
	"load_variable_registry",
	"load_workbook_frame",
	"save_informal_exports",
]