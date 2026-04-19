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


def _validate_loader_shape_contract(cfg: CMDLConfig) -> None:
	"""Ensure the economics loader matches the current domain preset contract.

	保证 economics loader 当前构造出的特征维度与配置契约一致。
	"""

	if cfg.seq_features != 1:
		raise ValueError(f"Economics loader currently emits exactly 1 sequential feature, got {cfg.seq_features}")
	if cfg.n_proxies != 1:
		raise ValueError(f"Economics loader currently emits exactly 1 proxy feature, got {cfg.n_proxies}")
	if cfg.static_dim != 2:
		raise ValueError(f"Economics loader currently emits exactly 2 static features, got {cfg.static_dim}")


def build_economics_dataframe(
	csv_path: str | Path | None = None,
	target_column: str = "ctfp",
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
	stats_end_year: int | None = None,
	max_missing_share: float = 0.15,
) -> pd.DataFrame:
	"""Build a cleaned long-form dataframe for the economics domain.

	为 economics 域构建清洗后的长表。该函数只服务 PWT，不尝试跨域抽象。
	其中 proxy/static 聚合和 X/Y 标准化默认只使用 stats_end_year 之前的统计量。
	"""

	if year_end <= year_start:
		raise ValueError(f"year_end must be greater than year_start, got {year_start}..{year_end}")
	if not 0.0 <= max_missing_share < 1.0:
		raise ValueError(f"max_missing_share must be in [0, 1), got {max_missing_share}")
	if stats_end_year is None:
		stats_end_year = year_end
	if not year_start <= stats_end_year <= year_end:
		raise ValueError(
			f"stats_end_year must be within [{year_start}, {year_end}], got {stats_end_year}"
		)

	source_path = _resolve_csv_path(csv_path)
	dataframe = load_pwt_source(str(source_path))

	required_columns = {"countrycode", "year", "hc", "ck", "rgdpna", target_column}
	_require_columns(dataframe, required_columns)

	selected_columns = ["countrycode", "year", "hc", "ck", "rgdpna", target_column]
	if "country" in dataframe.columns:
		selected_columns.insert(1, "country")

	long_frame = dataframe.loc[:, selected_columns].copy()
	if "country" not in long_frame.columns:
		long_frame["country"] = long_frame["countrycode"].astype(str)

	long_frame["countrycode"] = long_frame["countrycode"].astype(str).str.strip()
	long_frame["country"] = long_frame["country"].fillna(long_frame["countrycode"]).astype(str)
	long_frame["year"] = pd.to_numeric(long_frame["year"], errors="coerce")

	numeric_columns = ["hc", "ck", "rgdpna", target_column]
	for column_name in numeric_columns:
		long_frame[column_name] = pd.to_numeric(long_frame[column_name], errors="coerce")

	long_frame = long_frame.dropna(subset=["countrycode", "year"])
	long_frame["year"] = long_frame["year"].astype(int)
	long_frame = long_frame[(long_frame["year"] >= year_start) & (long_frame["year"] <= year_end)].copy()
	long_frame = long_frame[(long_frame["ck"] > 0.0) & (long_frame["rgdpna"] > 0.0)].copy()

	if long_frame.empty:
		raise ValueError("No economics rows remain after the initial year and positivity filters")

	long_frame["cap_deepening_raw"] = long_frame["ck"] / long_frame["rgdpna"]
	long_frame.loc[~np.isfinite(long_frame["cap_deepening_raw"]), "cap_deepening_raw"] = np.nan
	long_frame["log_rgdpna_raw"] = np.log(long_frame["rgdpna"].clip(lower=EPSILON))
	long_frame["log_ck_raw"] = np.log(long_frame["ck"].clip(lower=EPSILON))

	expected_years = list(range(year_start, year_end + 1))
	fill_columns = [target_column, "hc", "cap_deepening_raw", "log_rgdpna_raw", "log_ck_raw"]
	retained_entities: list[pd.DataFrame] = []

	for entity_code, group in long_frame.groupby("countrycode", sort=True):
		group = group.sort_values("year").groupby("year", as_index=False).last()
		entity_name = (
			str(group["country"].dropna().iloc[0]) if not group["country"].dropna().empty else str(entity_code)
		)
		entity_frame = group.set_index("year").reindex(expected_years)
		entity_frame["countrycode"] = str(entity_code)
		entity_frame["country"] = entity_name

		missing_share = float(entity_frame[fill_columns].isna().mean().mean())
		if missing_share > max_missing_share:
			continue

		entity_frame[fill_columns] = entity_frame[fill_columns].interpolate(
			method="linear",
			limit_direction="both",
		)
		if entity_frame[fill_columns].isna().any().any():
			continue

		stats_frame = entity_frame.loc[entity_frame.index <= stats_end_year, fill_columns]
		if stats_frame.empty or stats_frame.isna().any().any():
			continue
		reference_mask = entity_frame.index.to_numpy() <= stats_end_year

		entity_frame["proxy_hc_raw"] = float(stats_frame["hc"].mean())
		entity_frame["static_log_rgdpna_raw"] = float(stats_frame["log_rgdpna_raw"].mean())
		entity_frame["static_log_ck_raw"] = float(stats_frame["log_ck_raw"].mean())
		entity_frame["x_t"] = _zscore_with_reference(entity_frame["cap_deepening_raw"], reference_mask)
		entity_frame["y_t"] = _zscore_with_reference(entity_frame[target_column], reference_mask)
		retained_entities.append(entity_frame.reset_index().rename(columns={"index": "year"}))

	if not retained_entities:
		raise ValueError(
			"No economics entities remain after missing-value filtering. "
			"Relax max_missing_share or use a different year range."
		)

	cleaned_frame = pd.concat(retained_entities, ignore_index=True)
	static_frame = cleaned_frame.groupby("countrycode", as_index=False).agg(
		entity_name=("country", "first"),
		proxy_hc_raw=("proxy_hc_raw", "first"),
		static_log_rgdpna_raw=("static_log_rgdpna_raw", "first"),
		static_log_ck_raw=("static_log_ck_raw", "first"),
	)
	static_frame["proxy_hc"] = _zscore(static_frame["proxy_hc_raw"])
	static_frame["static_log_rgdpna"] = _zscore(static_frame["static_log_rgdpna_raw"])
	static_frame["static_log_ck"] = _zscore(static_frame["static_log_ck_raw"])

	cleaned_frame = cleaned_frame.merge(
		static_frame[
			["countrycode", "proxy_hc", "static_log_rgdpna", "static_log_ck"]
		],
		on="countrycode",
		how="left",
	)
	cleaned_frame = cleaned_frame.rename(columns={"countrycode": "entity_code", "country": "entity_name"})
	cleaned_frame = cleaned_frame.sort_values(["entity_code", "year"]).reset_index(drop=True)

	return cleaned_frame[
		[
			"entity_code",
			"entity_name",
			"year",
			"x_t",
			"y_t",
			"proxy_hc",
			"static_log_rgdpna",
			"static_log_ck",
		]
	]


def load_economics_panel(
	csv_path: str | Path | None = None,
	cfg: CMDLConfig | None = None,
	target_column: str = "ctfp",
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
	stats_end_year: int | None = None,
	max_missing_share: float = 0.15,
) -> EconomicsPanel:
	"""Load the economics domain into dense CMDL tensors.

	将 economics 域转换为可直接送入 CMDL 模型的稠密张量。
	"""

	runtime_cfg = CMDLConfig.from_domain("economics") if cfg is None else cfg
	_validate_loader_shape_contract(runtime_cfg)

	dataframe = build_economics_dataframe(
		csv_path=csv_path,
		target_column=target_column,
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

	entity_names: list[str] = []
	X_it = np.zeros((n_entities, seq_length, 1), dtype=np.float32)
	Y_it = np.zeros((n_entities, seq_length), dtype=np.float32)
	p_i = np.zeros((n_entities, 1), dtype=np.float32)
	s_i = np.zeros((n_entities, 2), dtype=np.float32)

	for entity_index, entity_code in enumerate(entity_codes):
		entity_frame = dataframe.loc[dataframe["entity_code"] == entity_code].sort_values("year")
		if len(entity_frame) != seq_length:
			raise ValueError(f"Entity {entity_code} does not have a balanced economics panel")

		entity_names.append(str(entity_frame["entity_name"].iloc[0]))
		X_it[entity_index, :, 0] = entity_frame["x_t"].to_numpy(dtype=np.float32)
		Y_it[entity_index, :] = entity_frame["y_t"].to_numpy(dtype=np.float32)
		p_i[entity_index, 0] = float(entity_frame["proxy_hc"].iloc[0])
		s_i[entity_index, 0] = float(entity_frame["static_log_rgdpna"].iloc[0])
		s_i[entity_index, 1] = float(entity_frame["static_log_ck"].iloc[0])

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
			"treatment_column": "cap_deepening",
			"proxy_columns": ["hc"],
			"static_columns": ["log_rgdpna_base", "log_ck_base"],
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
	"DEFAULT_YEAR_END",
	"DEFAULT_YEAR_START",
	"EconomicsPanel",
	"build_economics_dataframe",
	"build_temporal_splits",
	"get_prediction_years",
	"load_economics_panel",
	"slice_economics_panel",
]
