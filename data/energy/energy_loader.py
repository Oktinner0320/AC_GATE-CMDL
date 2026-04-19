"""Energy-domain loader for the minimal renewables-to-CO2 validation contract.

该 loader 读取下载阶段生成的标准化 merged CSV，完成：
1. 平衡面板筛选与缺失修复；
2. 训练窗口统计下的 X/Y 标准化；
3. WGI proxy 与静态特征聚合；
4. 最终张量化与时间切分。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from config.cmdl_config import CMDLConfig
from data.energy.download import DEFAULT_ENERGY_OUTPUT_PATH


DEFAULT_YEAR_START = 1996
DEFAULT_YEAR_END = 2023
DEFAULT_ENERGY_FEATURE_BUNDLE = "minimal"
SUPPORTED_ENERGY_FEATURE_BUNDLES = ("minimal",)
DEFAULT_TREATMENT_COLUMN = "renewables_share_energy"
DEFAULT_TARGET_COLUMN = "co2_per_unit_energy"
EPSILON = 1e-8


@dataclass(slots=True)
class EnergyPanel:
	"""Dense energy panel tensors aligned with the CMDL model contract."""

	X_it: torch.Tensor
	p_i: torch.Tensor
	s_i: torch.Tensor
	Y_it: torch.Tensor
	entity_ids: torch.Tensor
	time_index: torch.Tensor
	entity_codes: list[str]
	entity_names: list[str]
	metadata: dict[str, Any]


def _resolve_csv_path(csv_path: str | Path | None) -> Path:
	"""Resolve the standardized merged CSV path for the energy domain."""

	resolved = DEFAULT_ENERGY_OUTPUT_PATH if csv_path is None else Path(csv_path)
	resolved = resolved.resolve()
	if not resolved.exists():
		raise FileNotFoundError(
			"Energy table file not found. Run data/energy/download.py first or pass an explicit csv_path."
		)
	return resolved


def _zscore(values: pd.Series) -> pd.Series:
	"""Standardize one 1D series, returning zeros for degenerate inputs."""

	array = pd.to_numeric(values, errors="coerce").astype(float)
	mean = float(array.mean())
	std = float(array.std(ddof=0))
	if not np.isfinite(std) or std < EPSILON:
		return pd.Series(np.zeros(len(array), dtype=np.float32), index=values.index)
	return pd.Series(((array - mean) / std).astype(np.float32), index=values.index)


def _zscore_with_reference(values: pd.Series, reference_mask: pd.Series | np.ndarray) -> pd.Series:
	"""Standardize a series using statistics computed on a reference subset."""

	array = pd.to_numeric(values, errors="coerce").astype(float)
	mask = np.asarray(reference_mask, dtype=bool)
	if mask.shape[0] != len(array):
		raise ValueError("reference_mask must align with the values length")

	reference_array = array[mask]
	reference_array = reference_array[np.isfinite(reference_array)]
	if reference_array.size == 0:
		raise ValueError("reference subset for z-score is empty or non-finite")

	mean = float(reference_array.mean())
	std = float(reference_array.std(ddof=0))
	if not np.isfinite(std) or std < EPSILON:
		return pd.Series(np.zeros(len(array), dtype=np.float32), index=values.index)

	return pd.Series(((array - mean) / std).astype(np.float32), index=values.index)


def _require_columns(frame: pd.DataFrame, required_columns: set[str]) -> None:
	"""Validate that the standardized energy frame contains all required columns."""

	missing_columns = sorted(required_columns.difference(frame.columns))
	if missing_columns:
		raise ValueError(f"Missing required energy columns: {missing_columns}")


def _validate_feature_bundle(feature_bundle: str) -> str:
	"""Validate and normalize the requested energy feature bundle."""

	normalized = str(feature_bundle).strip().lower()
	if normalized not in SUPPORTED_ENERGY_FEATURE_BUNDLES:
		raise ValueError(
			f"Unsupported energy feature_bundle: {feature_bundle}. "
			f"Expected one of {list(SUPPORTED_ENERGY_FEATURE_BUNDLES)}"
		)
	return normalized


def _feature_bundle_columns(feature_bundle: str) -> tuple[list[str], list[str], list[str]]:
	"""Return the standardized column names emitted by one energy bundle."""

	_validate_feature_bundle(feature_bundle)
	return ["x_t"], [
		"proxy_government_effectiveness",
		"proxy_regulatory_quality",
		"proxy_rule_of_law",
	], ["static_log_population", "static_log_gdp_per_capita"]


def _validate_loader_shape_contract(
	cfg: CMDLConfig,
	seq_features: int,
	n_proxies: int,
	static_dim: int,
) -> None:
	"""Ensure the energy loader output matches the provided runtime config."""

	if cfg.seq_features != seq_features:
		raise ValueError(f"Energy loader emits {seq_features} sequential features, but cfg expects {cfg.seq_features}")
	if cfg.n_proxies != n_proxies:
		raise ValueError(f"Energy loader emits {n_proxies} proxies, but cfg expects {cfg.n_proxies}")
	if cfg.static_dim != static_dim:
		raise ValueError(f"Energy loader emits {static_dim} static features, but cfg expects {cfg.static_dim}")


def build_cleaned_energy_dataframe(
	csv_path: str | Path | None = None,
	treatment_column: str = DEFAULT_TREATMENT_COLUMN,
	target_column: str = DEFAULT_TARGET_COLUMN,
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
	max_missing_share: float = 0.15,
) -> pd.DataFrame:
	"""Build a cleaned long-form dataframe for the minimal energy contract."""

	if year_end <= year_start:
		raise ValueError(f"year_end must be greater than year_start, got {year_start}..{year_end}")
	if not 0.0 <= max_missing_share < 1.0:
		raise ValueError(f"max_missing_share must be in [0, 1), got {max_missing_share}")

	source_path = _resolve_csv_path(csv_path)
	dataframe = pd.read_csv(source_path)
	required_columns = {
		"entity_code",
		"entity_name",
		"year",
		treatment_column,
		target_column,
		"population",
		"gdp",
		"government_effectiveness",
		"regulatory_quality",
		"rule_of_law",
	}
	_require_columns(dataframe, required_columns)

	frame = dataframe.copy()
	frame["entity_code"] = frame["entity_code"].astype(str).str.strip().str.upper()
	frame["entity_name"] = frame["entity_name"].fillna(frame["entity_code"]).astype(str)
	frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
	numeric_columns = [
		treatment_column,
		target_column,
		"population",
		"gdp",
		"government_effectiveness",
		"regulatory_quality",
		"rule_of_law",
	]
	for column_name in numeric_columns:
		frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")

	frame = frame.dropna(subset=["entity_code", "year"])
	frame["year"] = frame["year"].astype(int)
	frame = frame[(frame["year"] >= year_start) & (frame["year"] <= year_end)].copy()
	frame = frame[frame["entity_code"].str.fullmatch(r"[A-Z]{3}", na=False)].copy()
	frame.loc[frame["population"] <= 0.0, "population"] = np.nan
	frame.loc[frame["gdp"] <= 0.0, "gdp"] = np.nan

	if frame.empty:
		raise ValueError("No energy rows remain after the initial year and code filters")

	expected_years = list(range(year_start, year_end + 1))
	required_value_columns = list(numeric_columns)
	missing_flag_columns = [f"{column_name}_was_missing" for column_name in required_value_columns]
	retained_entities: list[pd.DataFrame] = []

	for entity_code, group in frame.groupby("entity_code", sort=True):
		group = group.sort_values("year").groupby("year", as_index=False).last()
		entity_name = str(group["entity_name"].dropna().iloc[0]) if not group["entity_name"].dropna().empty else str(entity_code)
		entity_frame = group.set_index("year").reindex(expected_years)
		entity_frame["entity_code"] = str(entity_code)
		entity_frame["entity_name"] = entity_name

		missing_share = float(entity_frame[required_value_columns].isna().mean().mean())
		if missing_share > max_missing_share:
			continue

		for column_name in required_value_columns:
			entity_frame[f"{column_name}_was_missing"] = entity_frame[column_name].isna().astype(np.int8)
		entity_frame["row_was_missing"] = entity_frame[missing_flag_columns].max(axis=1).astype(np.int8)
		entity_frame["entity_missing_share"] = float(missing_share)

		entity_frame[required_value_columns] = entity_frame[required_value_columns].interpolate(
			method="linear",
			limit_direction="both",
		)
		if entity_frame[required_value_columns].isna().any().any():
			continue
		if not np.isfinite(entity_frame[required_value_columns].to_numpy(dtype=float)).all():
			continue

		entity_frame["gdp_per_capita_raw"] = entity_frame["gdp"] / entity_frame["population"]
		entity_frame["log_population_raw"] = np.log(entity_frame["population"].clip(lower=EPSILON))
		entity_frame["log_gdp_per_capita_raw"] = np.log(entity_frame["gdp_per_capita_raw"].clip(lower=EPSILON))
		if not np.isfinite(
			entity_frame[["gdp_per_capita_raw", "log_population_raw", "log_gdp_per_capita_raw"]].to_numpy(dtype=float)
		).all():
			continue

		retained_entities.append(entity_frame.reset_index().rename(columns={"index": "year"}))

	if not retained_entities:
		raise ValueError(
			"No energy entities remain after missing-value filtering. "
			"Relax max_missing_share or use a different year range."
		)

	cleaned_frame = pd.concat(retained_entities, ignore_index=True)
	cleaned_frame = cleaned_frame.sort_values(["entity_code", "year"]).reset_index(drop=True)
	return cleaned_frame.loc[
		:,
		[
			"entity_code",
			"entity_name",
			"year",
			treatment_column,
			target_column,
			"population",
			"gdp",
			"government_effectiveness",
			"regulatory_quality",
			"rule_of_law",
			"gdp_per_capita_raw",
			"log_population_raw",
			"log_gdp_per_capita_raw",
			*[column_name for column_name in missing_flag_columns if column_name in cleaned_frame.columns],
			"row_was_missing",
			"entity_missing_share",
		],
	]


def build_energy_dataframe(
	csv_path: str | Path | None = None,
	treatment_column: str = DEFAULT_TREATMENT_COLUMN,
	target_column: str = DEFAULT_TARGET_COLUMN,
	feature_bundle: str = DEFAULT_ENERGY_FEATURE_BUNDLE,
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
	stats_end_year: int | None = None,
	max_missing_share: float = 0.15,
) -> pd.DataFrame:
	"""Build the model-ready long-form dataframe for the energy domain."""

	if stats_end_year is None:
		stats_end_year = year_end
	if not year_start <= stats_end_year <= year_end:
		raise ValueError(f"stats_end_year must be within [{year_start}, {year_end}], got {stats_end_year}")
	feature_bundle = _validate_feature_bundle(feature_bundle)

	cleaned_frame = build_cleaned_energy_dataframe(
		csv_path=csv_path,
		treatment_column=treatment_column,
		target_column=target_column,
		year_start=year_start,
		year_end=year_end,
		max_missing_share=max_missing_share,
	)

	retained_entities: list[pd.DataFrame] = []
	for _, group in cleaned_frame.groupby("entity_code", sort=True):
		entity_frame = group.sort_values("year").copy()
		stats_frame = entity_frame.loc[entity_frame["year"] <= stats_end_year]
		if stats_frame.empty:
			continue
		reference_mask = entity_frame["year"].to_numpy() <= stats_end_year

		entity_frame["proxy_government_effectiveness_raw"] = float(stats_frame["government_effectiveness"].mean())
		entity_frame["proxy_regulatory_quality_raw"] = float(stats_frame["regulatory_quality"].mean())
		entity_frame["proxy_rule_of_law_raw"] = float(stats_frame["rule_of_law"].mean())
		entity_frame["static_log_population_raw"] = float(stats_frame["log_population_raw"].mean())
		entity_frame["static_log_gdp_per_capita_raw"] = float(stats_frame["log_gdp_per_capita_raw"].mean())
		entity_frame["x_t"] = _zscore_with_reference(entity_frame[treatment_column], reference_mask)
		entity_frame["y_t"] = _zscore_with_reference(entity_frame[target_column], reference_mask)
		retained_entities.append(entity_frame)

	if not retained_entities:
		raise ValueError(
			"No energy entities remain after missing-value filtering. "
			"Relax max_missing_share or use a different year range."
		)

	prepared_frame = pd.concat(retained_entities, ignore_index=True)
	static_frame = prepared_frame.groupby("entity_code", as_index=False).agg(
		{
			"entity_name": "first",
			"proxy_government_effectiveness_raw": "first",
			"proxy_regulatory_quality_raw": "first",
			"proxy_rule_of_law_raw": "first",
			"static_log_population_raw": "first",
			"static_log_gdp_per_capita_raw": "first",
		}
	)
	static_frame["proxy_government_effectiveness"] = _zscore(static_frame["proxy_government_effectiveness_raw"])
	static_frame["proxy_regulatory_quality"] = _zscore(static_frame["proxy_regulatory_quality_raw"])
	static_frame["proxy_rule_of_law"] = _zscore(static_frame["proxy_rule_of_law_raw"])
	static_frame["static_log_population"] = _zscore(static_frame["static_log_population_raw"])
	static_frame["static_log_gdp_per_capita"] = _zscore(static_frame["static_log_gdp_per_capita_raw"])

	prepared_frame = prepared_frame.merge(
		static_frame.loc[
			:,
			[
				"entity_code",
				"proxy_government_effectiveness",
				"proxy_regulatory_quality",
				"proxy_rule_of_law",
				"static_log_population",
				"static_log_gdp_per_capita",
			],
		],
		on="entity_code",
		how="left",
	)
	prepared_frame = prepared_frame.sort_values(["entity_code", "year"]).reset_index(drop=True)

	sequence_columns, proxy_columns, static_columns = _feature_bundle_columns(feature_bundle)
	return prepared_frame.loc[
		:,
		[
			"entity_code",
			"entity_name",
			"year",
			*sequence_columns,
			"y_t",
			*proxy_columns,
			*static_columns,
		],
	]


def load_energy_panel(
	csv_path: str | Path | None = None,
	cfg: CMDLConfig | None = None,
	treatment_column: str = DEFAULT_TREATMENT_COLUMN,
	target_column: str = DEFAULT_TARGET_COLUMN,
	feature_bundle: str = DEFAULT_ENERGY_FEATURE_BUNDLE,
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
	stats_end_year: int | None = None,
	max_missing_share: float = 0.15,
) -> EnergyPanel:
	"""Load the energy domain into dense CMDL tensors."""

	runtime_cfg = CMDLConfig.from_domain("energy") if cfg is None else cfg
	feature_bundle = _validate_feature_bundle(feature_bundle)
	sequence_columns, proxy_columns, static_columns = _feature_bundle_columns(feature_bundle)
	dataframe = build_energy_dataframe(
		csv_path=csv_path,
		treatment_column=treatment_column,
		target_column=target_column,
		feature_bundle=feature_bundle,
		year_start=year_start,
		year_end=year_end,
		stats_end_year=stats_end_year,
		max_missing_share=max_missing_share,
	)
	if stats_end_year is None:
		stats_end_year = year_end

	entity_codes = sorted(dataframe["entity_code"].unique().tolist())
	years = sorted(dataframe["year"].unique().tolist())
	n_entities = len(entity_codes)
	seq_length = len(years)

	if seq_length <= runtime_cfg.max_lag:
		raise ValueError(f"Energy panel length {seq_length} must be greater than max_lag {runtime_cfg.max_lag}")
	if cfg is not None:
		_validate_loader_shape_contract(runtime_cfg, len(sequence_columns), len(proxy_columns), len(static_columns))

	entity_names: list[str] = []
	X_it = np.zeros((n_entities, seq_length, len(sequence_columns)), dtype=np.float32)
	Y_it = np.zeros((n_entities, seq_length), dtype=np.float32)
	p_i = np.zeros((n_entities, len(proxy_columns)), dtype=np.float32)
	s_i = np.zeros((n_entities, len(static_columns)), dtype=np.float32)

	for entity_index, entity_code in enumerate(entity_codes):
		entity_frame = dataframe.loc[dataframe["entity_code"] == entity_code].sort_values("year")
		if len(entity_frame) != seq_length:
			raise ValueError(f"Entity {entity_code} does not have a balanced energy panel")

		entity_names.append(str(entity_frame["entity_name"].iloc[0]))
		for feature_index, column_name in enumerate(sequence_columns):
			X_it[entity_index, :, feature_index] = entity_frame[column_name].to_numpy(dtype=np.float32)
		Y_it[entity_index, :] = entity_frame["y_t"].to_numpy(dtype=np.float32)
		for proxy_index, column_name in enumerate(proxy_columns):
			p_i[entity_index, proxy_index] = float(entity_frame[column_name].iloc[0])
		for static_index, column_name in enumerate(static_columns):
			s_i[entity_index, static_index] = float(entity_frame[column_name].iloc[0])

	source_path = _resolve_csv_path(csv_path)
	return EnergyPanel(
		X_it=torch.tensor(X_it, dtype=torch.float32),
		p_i=torch.tensor(p_i, dtype=torch.float32),
		s_i=torch.tensor(s_i, dtype=torch.float32),
		Y_it=torch.tensor(Y_it, dtype=torch.float32),
		entity_ids=torch.arange(n_entities, dtype=torch.long),
		time_index=torch.arange(seq_length, dtype=torch.long),
		entity_codes=entity_codes,
		entity_names=entity_names,
		metadata={
			"domain": "energy",
			"source_path": str(source_path),
			"treatment_column": treatment_column,
			"target_column": target_column,
			"feature_bundle": feature_bundle,
			"seq_feature_columns": list(sequence_columns),
			"proxy_columns": list(proxy_columns),
			"static_columns": list(static_columns),
			"years": years,
			"year_start": int(years[0]),
			"year_end": int(years[-1]),
			"stats_end_year": int(stats_end_year),
			"n_entities": n_entities,
			"seq_length": seq_length,
			"max_missing_share": float(max_missing_share),
		},
	)


def slice_energy_panel(panel: EnergyPanel, start_year: int, end_year: int) -> EnergyPanel:
	"""Slice a contiguous year window from one dense energy panel."""

	if end_year < start_year:
		raise ValueError(f"end_year must be >= start_year, got {start_year}..{end_year}")

	years = panel.metadata["years"]
	if start_year not in years or end_year not in years:
		raise ValueError(f"Requested slice {start_year}..{end_year} is outside the energy panel range")

	start_index = years.index(start_year)
	end_index = years.index(end_year)
	positions = list(range(start_index, end_index + 1))
	sliced_years = years[start_index : end_index + 1]

	metadata = dict(panel.metadata)
	metadata["years"] = sliced_years
	metadata["year_start"] = int(sliced_years[0])
	metadata["year_end"] = int(sliced_years[-1])
	metadata["seq_length"] = len(sliced_years)

	return EnergyPanel(
		X_it=panel.X_it[:, positions, :].clone(),
		p_i=panel.p_i.clone(),
		s_i=panel.s_i.clone(),
		Y_it=panel.Y_it[:, positions].clone(),
		entity_ids=panel.entity_ids.clone(),
		time_index=torch.arange(len(positions), dtype=torch.long),
		entity_codes=list(panel.entity_codes),
		entity_names=list(panel.entity_names),
		metadata=metadata,
	)


def build_temporal_splits(
	panel: EnergyPanel,
	max_lag: int,
	train_end_year: int,
	val_end_year: int,
) -> tuple[EnergyPanel, EnergyPanel, EnergyPanel]:
	"""Create train/validation/test panels with lag-context overlap."""

	years = panel.metadata["years"]
	full_start_year = int(years[0])
	full_end_year = int(years[-1])
	val_start_year = train_end_year + 1
	test_start_year = val_end_year + 1

	if not full_start_year < train_end_year < val_end_year < full_end_year:
		raise ValueError(
			"Expected full_start_year < train_end_year < val_end_year < full_end_year, "
			f"got {full_start_year} < {train_end_year} < {val_end_year} < {full_end_year}"
		)
	if val_start_year - max_lag < full_start_year or test_start_year - max_lag < full_start_year:
		raise ValueError("Not enough historical context before validation/test windows for the requested max_lag")

	train_panel = slice_energy_panel(panel, full_start_year, train_end_year)
	val_panel = slice_energy_panel(panel, val_start_year - max_lag, val_end_year)
	test_panel = slice_energy_panel(panel, test_start_year - max_lag, full_end_year)

	for split_name, split_panel in [("train", train_panel), ("val", val_panel), ("test", test_panel)]:
		if split_panel.X_it.shape[1] <= max_lag:
			raise ValueError(f"{split_name} split length must exceed max_lag; got {split_panel.X_it.shape[1]}")

	return train_panel, val_panel, test_panel


def get_prediction_years(panel: EnergyPanel, max_lag: int) -> list[int]:
	"""Return the actual calendar years aligned with post-warm-up predictions."""

	years = panel.metadata["years"]
	if len(years) <= max_lag:
		raise ValueError(f"Panel length {len(years)} must exceed max_lag {max_lag}")
	return [int(year) for year in years[max_lag:]]


__all__ = [
	"DEFAULT_ENERGY_FEATURE_BUNDLE",
	"DEFAULT_TARGET_COLUMN",
	"DEFAULT_TREATMENT_COLUMN",
	"DEFAULT_YEAR_END",
	"DEFAULT_YEAR_START",
	"EnergyPanel",
	"SUPPORTED_ENERGY_FEATURE_BUNDLES",
	"build_cleaned_energy_dataframe",
	"build_energy_dataframe",
	"build_temporal_splits",
	"get_prediction_years",
	"load_energy_panel",
	"slice_energy_panel",
]
