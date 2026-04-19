"""Economics-domain loader for Penn World Table based CMDL experiments.

该 loader 不假定与其他真实数据域共享 preprocessing。它只处理 PWT 单表：
原始列筛选、资本深化构造、缺失修复、静态特征与 proxy 抽取、以及最终张量化。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from config.cmdl_config import CMDLConfig
from data.economics.download import DEFAULT_PWT_OUTPUT_PATH, load_pwt_source


DEFAULT_YEAR_START = 1980
DEFAULT_YEAR_END = 2023
EPSILON = 1e-8
DEFAULT_ECONOMICS_FEATURE_BUNDLE = "minimal"
SUPPORTED_ECONOMICS_FEATURE_BUNDLES = ("minimal", "growth_aware")
OPTIONAL_ECONOMICS_EXPORT_COLUMNS = ("ctfp", "rtfpna", "emp", "avh", "labsh", "delta", "rnna", "rkna")


@dataclass(slots=True)
class EconomicsPanel:
	"""Dense economics panel tensors aligned with the CMDL model contract.

	economics 域最终张量容器；字段对齐 CMDL 前向所需输入，但不携带 synthetic
	专属的 ground-truth latent annotations。
	"""

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
	"""Resolve the source CSV path for the economics domain.

	解析 economics 域使用的本地 CSV 路径。
	"""

	resolved = DEFAULT_PWT_OUTPUT_PATH if csv_path is None else Path(csv_path)
	resolved = resolved.resolve()
	if not resolved.exists():
		raise FileNotFoundError(
			"Economics table file not found. Run data/economics/download.py first or pass an explicit csv_path."
		)
	return resolved


def _zscore(values: pd.Series) -> pd.Series:
	"""Standardize one 1D series, returning zeros for degenerate inputs.

	对一维序列做 z-score 标准化；若方差退化则回退为全零。
	"""

	array = pd.to_numeric(values, errors="coerce").astype(float)
	mean = float(array.mean())
	std = float(array.std(ddof=0))
	if not np.isfinite(std) or std < EPSILON:
		return pd.Series(np.zeros(len(array), dtype=np.float32), index=values.index)
	normalized = (array - mean) / std
	return pd.Series(normalized.astype(np.float32), index=values.index)


def _zscore_with_reference(values: pd.Series, reference_mask: pd.Series | np.ndarray) -> pd.Series:
	"""Standardize a series using statistics computed on a reference subset.

	用参考子区间上的均值和方差对整段序列做标准化，避免把未来窗口信息泄漏回训练期。
	"""

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

	normalized = (array - mean) / std
	return pd.Series(normalized.astype(np.float32), index=values.index)


def _require_columns(frame: pd.DataFrame, required_columns: set[str]) -> None:
	"""Validate that the PWT frame contains all required columns.

	校验 PWT 原始表是否包含所需列。
	"""

	missing_columns = sorted(required_columns.difference(frame.columns))
	if missing_columns:
		raise ValueError(f"Missing required economics columns: {missing_columns}")


def _validate_feature_bundle(feature_bundle: str) -> str:
	"""Validate and normalize the requested economics feature bundle."""

	normalized = str(feature_bundle).strip().lower()
	if normalized not in SUPPORTED_ECONOMICS_FEATURE_BUNDLES:
		raise ValueError(
			f"Unsupported economics feature_bundle: {feature_bundle}. "
			f"Expected one of {list(SUPPORTED_ECONOMICS_FEATURE_BUNDLES)}"
		)
	return normalized


def _feature_bundle_columns(feature_bundle: str) -> tuple[list[str], list[str], list[str]]:
	"""Return the standardized column names emitted by one economics bundle."""

	normalized = _validate_feature_bundle(feature_bundle)
	if normalized == "minimal":
		return ["x_t"], ["proxy_hc"], ["static_log_rgdpna", "static_log_ck"]
	if normalized == "growth_aware":
		return ["x_cap_deepening", "x_log_rgdpna_growth", "x_log_ck_growth"], ["proxy_hc_level", "proxy_hc_trend"], [
			"static_log_rgdpna",
			"static_log_ck",
		]
	raise AssertionError(f"Unhandled feature_bundle: {feature_bundle}")


def _validate_loader_shape_contract(
	cfg: CMDLConfig,
	seq_features: int,
	n_proxies: int,
	static_dim: int,
) -> None:
	"""Ensure the economics loader output matches the provided runtime config."""

	if cfg.seq_features != seq_features:
		raise ValueError(
			f"Economics loader emits {seq_features} sequential features, but cfg expects {cfg.seq_features}"
		)
	if cfg.n_proxies != n_proxies:
		raise ValueError(f"Economics loader emits {n_proxies} proxies, but cfg expects {cfg.n_proxies}")
	if cfg.static_dim != static_dim:
		raise ValueError(f"Economics loader emits {static_dim} static features, but cfg expects {cfg.static_dim}")


def _linear_trend(values: pd.Series) -> float:
	"""Estimate a simple per-step linear trend on a 1D series."""

	array = pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
	mask = np.isfinite(array)
	if mask.sum() <= 1:
		return 0.0
	x = np.arange(array.shape[0], dtype=np.float64)[mask]
	y = array[mask].astype(np.float64)
	x_centered = x - x.mean()
	denominator = float(np.dot(x_centered, x_centered))
	if denominator < EPSILON:
		return 0.0
	y_centered = y - y.mean()
	return float(np.dot(x_centered, y_centered) / denominator)


def _looks_like_cleaned_economics_frame(frame: pd.DataFrame, target_column: str) -> bool:
	"""Infer whether a table is the cleaned long-form economics export.

	判断输入表是否已经是清洗后的 economics long-form 导出。
	"""

	required_columns = {
		"entity_code",
		"entity_name",
		"year",
		"hc",
		"ck",
		"rgdpna",
		"cap_deepening_raw",
		"log_rgdpna_raw",
		"log_ck_raw",
	}
	return required_columns.issubset(frame.columns)


def _build_cleaned_from_exported_table(
	dataframe: pd.DataFrame,
	target_column: str,
	year_start: int,
	year_end: int,
) -> pd.DataFrame:
	"""Validate and normalize an already cleaned long-form economics table.

	校验并规范化已经清洗好的 economics long-form 表，使其可直接复用到训练 loader。
	"""

	required_columns = {
		"entity_code",
		"entity_name",
		"year",
		"hc",
		"ck",
		"rgdpna",
		target_column,
		"cap_deepening_raw",
		"log_rgdpna_raw",
		"log_ck_raw",
	}
	_require_columns(dataframe, required_columns)

	frame = dataframe.copy()
	optional_export_columns = [
		column_name
		for column_name in OPTIONAL_ECONOMICS_EXPORT_COLUMNS
		if column_name in frame.columns and column_name not in required_columns
	]
	frame["entity_code"] = frame["entity_code"].astype(str).str.strip()
	frame["entity_name"] = frame["entity_name"].fillna(frame["entity_code"]).astype(str)
	frame["year"] = pd.to_numeric(frame["year"], errors="coerce")

	numeric_columns = [
		"hc",
		"ck",
		"rgdpna",
		target_column,
		"cap_deepening_raw",
		"log_rgdpna_raw",
		"log_ck_raw",
	]
	for column_name in optional_export_columns:
		frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
	for column_name in numeric_columns:
		frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")

	frame = frame.dropna(subset=["entity_code", "year"])
	frame["year"] = frame["year"].astype(int)
	frame = frame[(frame["year"] >= year_start) & (frame["year"] <= year_end)].copy()
	if frame.empty:
		raise ValueError("No economics rows remain after filtering the cleaned long-form table")

	expected_years = list(range(year_start, year_end + 1))
	retained_entities: list[pd.DataFrame] = []
	optional_defaults: dict[str, Any] = {
		"hc_was_missing": 0,
		"ck_was_missing": 0,
		"rgdpna_was_missing": 0,
		f"{target_column}_was_missing": 0,
		"row_was_missing": 0,
		"entity_missing_share": 0.0,
	}

	for entity_code, group in frame.groupby("entity_code", sort=True):
		entity_frame = group.sort_values("year").groupby("year", as_index=False).last()
		if entity_frame["year"].tolist() != expected_years:
			raise ValueError(
				"Cleaned economics input must remain a balanced panel for the requested year range; "
				f"entity {entity_code} does not cover {year_start}..{year_end}"
			)
		if entity_frame[numeric_columns].isna().any().any():
			raise ValueError(f"Cleaned economics input still contains missing numeric values for entity {entity_code}")
		if not np.isfinite(entity_frame[numeric_columns].to_numpy(dtype=float)).all():
			raise ValueError(f"Cleaned economics input contains non-finite numeric values for entity {entity_code}")

		entity_frame["dlog_rgdpna_raw"] = entity_frame["log_rgdpna_raw"].diff().fillna(0.0)
		entity_frame["dlog_ck_raw"] = entity_frame["log_ck_raw"].diff().fillna(0.0)

		for column_name, default_value in optional_defaults.items():
			if column_name not in entity_frame.columns:
				entity_frame[column_name] = default_value
		retained_entities.append(entity_frame)

	cleaned_frame = pd.concat(retained_entities, ignore_index=True)
	cleaned_frame = cleaned_frame.sort_values(["entity_code", "year"]).reset_index(drop=True)

	output_columns = [
		"entity_code",
		"entity_name",
		"year",
		"hc",
		"ck",
		"rgdpna",
	]
	for column_name in ["ctfp", "rtfpna"]:
		if column_name in cleaned_frame.columns and column_name not in output_columns:
			output_columns.append(column_name)
	for column_name in optional_export_columns:
		if column_name not in output_columns:
			output_columns.append(column_name)
	output_columns.extend([
		"cap_deepening_raw",
		"log_rgdpna_raw",
		"log_ck_raw",
		"dlog_rgdpna_raw",
		"dlog_ck_raw",
		"hc_was_missing",
		"ck_was_missing",
		"rgdpna_was_missing",
		f"{target_column}_was_missing",
		"row_was_missing",
		"entity_missing_share",
	])
	return cleaned_frame.loc[:, [column_name for column_name in output_columns if column_name in cleaned_frame.columns]]


def build_cleaned_economics_dataframe(
	csv_path: str | Path | None = None,
	target_column: str = "ctfp",
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
	max_missing_share: float = 0.15,
) -> pd.DataFrame:
	"""Build a cleaned long-form economics table before model-specific transforms.

	先生成可落盘检查的 economics 清洗长表，再由下游步骤追加训练窗口相关的
	aggregation 与标准化，避免把“清洗”和“建模预处理”混为一体。
	"""

	if year_end <= year_start:
		raise ValueError(f"year_end must be greater than year_start, got {year_start}..{year_end}")
	if not 0.0 <= max_missing_share < 1.0:
		raise ValueError(f"max_missing_share must be in [0, 1), got {max_missing_share}")

	source_path = _resolve_csv_path(csv_path)
	dataframe = load_pwt_source(str(source_path))
	if _looks_like_cleaned_economics_frame(dataframe, target_column):
		return _build_cleaned_from_exported_table(
			dataframe=dataframe,
			target_column=target_column,
			year_start=year_start,
			year_end=year_end,
		)

	required_columns = {"countrycode", "year", "hc", "ck", "rgdpna", target_column}
	_require_columns(dataframe, required_columns)

	optional_export_columns = [
		column_name
		for column_name in OPTIONAL_ECONOMICS_EXPORT_COLUMNS
		if column_name in dataframe.columns and column_name not in {"countrycode", "year", "hc", "ck", "rgdpna"}
	]
	selected_columns = ["countrycode", "year", "hc", "ck", "rgdpna", *optional_export_columns]
	if "country" in dataframe.columns:
		selected_columns.insert(1, "country")

	long_frame = dataframe.loc[:, selected_columns].copy()
	if "country" not in long_frame.columns:
		long_frame["country"] = long_frame["countrycode"].astype(str)

	long_frame["countrycode"] = long_frame["countrycode"].astype(str).str.strip()
	long_frame["country"] = long_frame["country"].fillna(long_frame["countrycode"]).astype(str)
	long_frame["year"] = pd.to_numeric(long_frame["year"], errors="coerce")

	required_value_columns = ["hc", "ck", "rgdpna", target_column]
	numeric_columns = [column_name for column_name in optional_export_columns if column_name not in {"countrycode", "year"}]
	for column_name in sorted(set(required_value_columns + numeric_columns)):
		long_frame[column_name] = pd.to_numeric(long_frame[column_name], errors="coerce")

	long_frame = long_frame.dropna(subset=["countrycode", "year"])
	long_frame["year"] = long_frame["year"].astype(int)
	long_frame = long_frame[(long_frame["year"] >= year_start) & (long_frame["year"] <= year_end)].copy()
	long_frame = long_frame[(long_frame["ck"] > 0.0) & (long_frame["rgdpna"] > 0.0)].copy()

	if long_frame.empty:
		raise ValueError("No economics rows remain after the initial year and positivity filters")

	expected_years = list(range(year_start, year_end + 1))
	retained_entities: list[pd.DataFrame] = []
	missing_flag_columns = [f"{column_name}_was_missing" for column_name in required_value_columns]

	for entity_code, group in long_frame.groupby("countrycode", sort=True):
		group = group.sort_values("year").groupby("year", as_index=False).last()
		entity_name = (
			str(group["country"].dropna().iloc[0]) if not group["country"].dropna().empty else str(entity_code)
		)
		entity_frame = group.set_index("year").reindex(expected_years)
		entity_frame["countrycode"] = str(entity_code)
		entity_frame["country"] = entity_name

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
		optional_numeric_columns = [
			column_name
			for column_name in numeric_columns
			if column_name not in required_value_columns and column_name in entity_frame.columns
		]
		if optional_numeric_columns:
			entity_frame[optional_numeric_columns] = entity_frame[optional_numeric_columns].interpolate(
				method="linear",
				limit_direction="both",
			)

		entity_frame["cap_deepening_raw"] = entity_frame["ck"] / entity_frame["rgdpna"]
		entity_frame["log_rgdpna_raw"] = np.log(entity_frame["rgdpna"].clip(lower=EPSILON))
		entity_frame["log_ck_raw"] = np.log(entity_frame["ck"].clip(lower=EPSILON))
		entity_frame["dlog_rgdpna_raw"] = entity_frame["log_rgdpna_raw"].diff().fillna(0.0)
		entity_frame["dlog_ck_raw"] = entity_frame["log_ck_raw"].diff().fillna(0.0)
		derived_columns = ["cap_deepening_raw", "log_rgdpna_raw", "log_ck_raw", "dlog_rgdpna_raw", "dlog_ck_raw"]
		if not np.isfinite(entity_frame[derived_columns].to_numpy(dtype=float)).all():
			continue

		retained_entities.append(entity_frame.reset_index().rename(columns={"index": "year"}))

	if not retained_entities:
		raise ValueError(
			"No economics entities remain after missing-value filtering. "
			"Relax max_missing_share or use a different year range."
		)

	cleaned_frame = pd.concat(retained_entities, ignore_index=True)
	cleaned_frame = cleaned_frame.rename(columns={"countrycode": "entity_code", "country": "entity_name"})
	cleaned_frame = cleaned_frame.sort_values(["entity_code", "year"]).reset_index(drop=True)

	output_columns = [
		"entity_code",
		"entity_name",
		"year",
		"hc",
		"ck",
		"rgdpna",
	]
	for column_name in ["ctfp", "rtfpna"]:
		if column_name in cleaned_frame.columns and column_name not in output_columns:
			output_columns.append(column_name)
	for column_name in [
		column_name
		for column_name in optional_export_columns
		if column_name not in {"ctfp", "rtfpna"} and column_name in cleaned_frame.columns
	]:
		output_columns.append(column_name)
	output_columns.extend([
		"cap_deepening_raw",
		"log_rgdpna_raw",
		"log_ck_raw",
		"dlog_rgdpna_raw",
		"dlog_ck_raw",
		"hc_was_missing",
		"ck_was_missing",
		"rgdpna_was_missing",
		f"{target_column}_was_missing",
		"row_was_missing",
		"entity_missing_share",
	])
	return cleaned_frame.loc[:, [column_name for column_name in output_columns if column_name in cleaned_frame.columns]]


def build_economics_dataframe(
	csv_path: str | Path | None = None,
	target_column: str = "ctfp",
	feature_bundle: str = DEFAULT_ECONOMICS_FEATURE_BUNDLE,
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
	stats_end_year: int | None = None,
	max_missing_share: float = 0.15,
) -> pd.DataFrame:
	"""Build a cleaned long-form dataframe for the economics domain.

	为 economics 域构建清洗后的长表。该函数只服务 PWT，不尝试跨域抽象。
	其中 proxy/static 聚合和 X/Y 标准化默认只使用 stats_end_year 之前的统计量。
	"""

	if stats_end_year is None:
		stats_end_year = year_end
	if not year_start <= stats_end_year <= year_end:
		raise ValueError(
			f"stats_end_year must be within [{year_start}, {year_end}], got {stats_end_year}"
		)
	feature_bundle = _validate_feature_bundle(feature_bundle)

	cleaned_frame = build_cleaned_economics_dataframe(
		csv_path=csv_path,
		target_column=target_column,
		year_start=year_start,
		year_end=year_end,
		max_missing_share=max_missing_share,
	)

	retained_entities: list[pd.DataFrame] = []
	for entity_code, group in cleaned_frame.groupby("entity_code", sort=True):
		entity_frame = group.sort_values("year").copy()
		stats_frame = entity_frame.loc[entity_frame["year"] <= stats_end_year]
		if stats_frame.empty:
			continue
		reference_mask = entity_frame["year"].to_numpy() <= stats_end_year

		entity_frame["static_log_rgdpna_raw"] = float(stats_frame["log_rgdpna_raw"].mean())
		entity_frame["static_log_ck_raw"] = float(stats_frame["log_ck_raw"].mean())
		if feature_bundle == "minimal":
			entity_frame["proxy_hc_raw"] = float(stats_frame["hc"].mean())
			entity_frame["x_t"] = _zscore_with_reference(entity_frame["cap_deepening_raw"], reference_mask)
		elif feature_bundle == "growth_aware":
			entity_frame["proxy_hc_level_raw"] = float(stats_frame["hc"].mean())
			entity_frame["proxy_hc_trend_raw"] = _linear_trend(stats_frame["hc"])
			entity_frame["x_cap_deepening"] = _zscore_with_reference(entity_frame["cap_deepening_raw"], reference_mask)
			entity_frame["x_log_rgdpna_growth"] = _zscore_with_reference(entity_frame["dlog_rgdpna_raw"], reference_mask)
			entity_frame["x_log_ck_growth"] = _zscore_with_reference(entity_frame["dlog_ck_raw"], reference_mask)
		else:
			raise AssertionError(f"Unhandled feature_bundle: {feature_bundle}")
		entity_frame["y_t"] = _zscore_with_reference(entity_frame[target_column], reference_mask)
		retained_entities.append(entity_frame)

	if not retained_entities:
		raise ValueError(
			"No economics entities remain after missing-value filtering. "
			"Relax max_missing_share or use a different year range."
		)

	cleaned_frame = pd.concat(retained_entities, ignore_index=True)
	if feature_bundle == "minimal":
		proxy_raw_columns = [("proxy_hc_raw", "proxy_hc")]
		sequence_columns = ["x_t"]
	elif feature_bundle == "growth_aware":
		proxy_raw_columns = [("proxy_hc_level_raw", "proxy_hc_level"), ("proxy_hc_trend_raw", "proxy_hc_trend")]
		sequence_columns = ["x_cap_deepening", "x_log_rgdpna_growth", "x_log_ck_growth"]
	else:
		raise AssertionError(f"Unhandled feature_bundle: {feature_bundle}")

	static_frame = cleaned_frame.groupby("entity_code", as_index=False).agg(
		{
			"entity_name": "first",
			**{raw_column: "first" for raw_column, _ in proxy_raw_columns},
			"static_log_rgdpna_raw": "first",
			"static_log_ck_raw": "first",
		}
	)
	for raw_column, normalized_column in proxy_raw_columns:
		static_frame[normalized_column] = _zscore(static_frame[raw_column])
	static_frame["static_log_rgdpna"] = _zscore(static_frame["static_log_rgdpna_raw"])
	static_frame["static_log_ck"] = _zscore(static_frame["static_log_ck_raw"])

	merge_columns = [
		"entity_code",
		*[normalized_column for _, normalized_column in proxy_raw_columns],
		"static_log_rgdpna",
		"static_log_ck",
	]
	cleaned_frame = cleaned_frame.merge(
		static_frame[merge_columns],
		on="entity_code",
		how="left",
	)
	cleaned_frame = cleaned_frame.sort_values(["entity_code", "year"]).reset_index(drop=True)

	return cleaned_frame.loc[
		:,
		[
			"entity_code",
			"entity_name",
			"year",
			*sequence_columns,
			"y_t",
			*[normalized_column for _, normalized_column in proxy_raw_columns],
			"static_log_rgdpna",
			"static_log_ck",
		],
	]


def load_economics_panel(
	csv_path: str | Path | None = None,
	cfg: CMDLConfig | None = None,
	target_column: str = "ctfp",
	feature_bundle: str = DEFAULT_ECONOMICS_FEATURE_BUNDLE,
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
	stats_end_year: int | None = None,
	max_missing_share: float = 0.15,
) -> EconomicsPanel:
	"""Load the economics domain into dense CMDL tensors.

	将 economics 域转换为可直接送入 CMDL 模型的稠密张量。
	"""

	runtime_cfg = CMDLConfig.from_domain("economics") if cfg is None else cfg
	feature_bundle = _validate_feature_bundle(feature_bundle)
	sequence_columns, proxy_columns, static_columns = _feature_bundle_columns(feature_bundle)

	dataframe = build_economics_dataframe(
		csv_path=csv_path,
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
		raise ValueError(
			f"Economics panel length {seq_length} must be greater than max_lag {runtime_cfg.max_lag}"
		)
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
			raise ValueError(f"Entity {entity_code} does not have a balanced economics panel")

		entity_names.append(str(entity_frame["entity_name"].iloc[0]))
		for feature_index, column_name in enumerate(sequence_columns):
			X_it[entity_index, :, feature_index] = entity_frame[column_name].to_numpy(dtype=np.float32)
		Y_it[entity_index, :] = entity_frame["y_t"].to_numpy(dtype=np.float32)
		for proxy_index, column_name in enumerate(proxy_columns):
			p_i[entity_index, proxy_index] = float(entity_frame[column_name].iloc[0])
		for static_index, column_name in enumerate(static_columns):
			s_i[entity_index, static_index] = float(entity_frame[column_name].iloc[0])

	source_path = _resolve_csv_path(csv_path)
	return EconomicsPanel(
		X_it=torch.tensor(X_it, dtype=torch.float32),
		p_i=torch.tensor(p_i, dtype=torch.float32),
		s_i=torch.tensor(s_i, dtype=torch.float32),
		Y_it=torch.tensor(Y_it, dtype=torch.float32),
		entity_ids=torch.arange(n_entities, dtype=torch.long),
		time_index=torch.arange(seq_length, dtype=torch.long),
		entity_codes=entity_codes,
		entity_names=entity_names,
		metadata={
			"domain": "economics",
			"source_path": str(source_path),
			"target_column": target_column,
			"feature_bundle": feature_bundle,
			"treatment_column": "cap_deepening",
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


def slice_economics_panel(panel: EconomicsPanel, start_year: int, end_year: int) -> EconomicsPanel:
	"""Slice a contiguous year window from one dense economics panel.

	从 economics 面板中裁切一个连续年份窗口。
	"""

	if end_year < start_year:
		raise ValueError(f"end_year must be >= start_year, got {start_year}..{end_year}")

	years = panel.metadata["years"]
	if start_year not in years or end_year not in years:
		raise ValueError(f"Requested slice {start_year}..{end_year} is outside the economics panel range")

	start_index = years.index(start_year)
	end_index = years.index(end_year)
	positions = list(range(start_index, end_index + 1))
	sliced_years = years[start_index : end_index + 1]

	metadata = dict(panel.metadata)
	metadata["years"] = sliced_years
	metadata["year_start"] = int(sliced_years[0])
	metadata["year_end"] = int(sliced_years[-1])
	metadata["seq_length"] = len(sliced_years)

	return EconomicsPanel(
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
	panel: EconomicsPanel,
	max_lag: int,
	train_end_year: int,
	val_end_year: int,
) -> tuple[EconomicsPanel, EconomicsPanel, EconomicsPanel]:
	"""Create train/validation/test panels with lag-context overlap.

	构造按时间切分的 train/val/test 面板；验证和测试面板会自动附带前置 lag
	context 年份，使 warm-up 后的预测刚好落在各自的评估区间上。
	"""

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

	train_panel = slice_economics_panel(panel, full_start_year, train_end_year)
	val_panel = slice_economics_panel(panel, val_start_year - max_lag, val_end_year)
	test_panel = slice_economics_panel(panel, test_start_year - max_lag, full_end_year)

	for split_name, split_panel in [("train", train_panel), ("val", val_panel), ("test", test_panel)]:
		if split_panel.X_it.shape[1] <= max_lag:
			raise ValueError(f"{split_name} split length must exceed max_lag; got {split_panel.X_it.shape[1]}")

	return train_panel, val_panel, test_panel


def get_prediction_years(panel: EconomicsPanel, max_lag: int) -> list[int]:
	"""Return the actual calendar years aligned with post-warm-up predictions.

	返回 warm-up 截断后与模型预测对齐的真实年份。
	"""

	years = panel.metadata["years"]
	if len(years) <= max_lag:
		raise ValueError(f"Panel length {len(years)} must exceed max_lag {max_lag}")
	return [int(year) for year in years[max_lag:]]


__all__ = [
	"DEFAULT_ECONOMICS_FEATURE_BUNDLE",
	"DEFAULT_YEAR_END",
	"DEFAULT_YEAR_START",
	"EconomicsPanel",
	"SUPPORTED_ECONOMICS_FEATURE_BUNDLES",
	"build_economics_dataframe",
	"build_cleaned_economics_dataframe",
	"build_temporal_splits",
	"get_prediction_years",
	"load_economics_panel",
	"slice_economics_panel",
]
