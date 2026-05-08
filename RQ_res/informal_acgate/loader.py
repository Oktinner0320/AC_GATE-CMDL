"""Informal panel loader isolated under RQ_res.

The loader consumes the curated CSVs produced by data/Informal/prepare.py and
emits tensors that match the existing CMDL/AC-GATE model contract without
modifying the shared project loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


RQ_RES_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = WORKSPACE_ROOT / "data" / "Informal" / "processed" / "acgate_inputs"

TARGET_COLUMN = "y_t"
LEGACY_SEQUENCE_COLUMN = "x_t"
MULTISEQ_COLUMNS = [
    "x_seq_rpcyd_total_income",
    "x_seq_rydgdp_disposable_income",
    "x_seq_rydgdp_disposable_income_per_capita",
    "x_seq_rydgdp_grdp",
    "x_seq_rydgdp_grdp_per_capita",
    "x_seq_rydgdp_ratio",
]
PROXY_COLUMNS = ["proxy_atm_count", "proxy_terminal_count", "proxy_credit_card_depth"]
STATIC_COLUMNS = ["static_noncash_value", "static_withdrawal_value"]
DERIVED_PROXY_COLUMNS = [
    "proxy_formal_income_level",
    "proxy_formalization_ratio",
    "proxy_formal_capacity_signal",
]
DERIVED_STATIC_COLUMNS = ["static_formal_income_trend", "static_formal_income_volatility"]
INCOME_PROXY_COLUMNS = [
    "proxy_income_level",
    "proxy_income_recent_level",
    "proxy_income_growth_signal",
]
INCOME_STATIC_COLUMNS = ["static_income_trend", "static_income_volatility"]

SUPPORTED_FEATURE_BUNDLES = {
    "single_overlap": {
        "file_name": "informal_acgate_single_feature_ready.csv",
        "sequence_columns": [LEGACY_SEQUENCE_COLUMN],
        "year_start": 2019,
        "year_end": 2023,
    },
    "single_fullspan": {
        "file_name": "informal_acgate_single_feature_ready.csv",
        "sequence_columns": [LEGACY_SEQUENCE_COLUMN],
        "year_start": 2006,
        "year_end": 2023,
    },
    "multiseq_overlap": {
        "file_name": "informal_acgate_multiseq_overlap_ready.csv",
        "sequence_columns": list(MULTISEQ_COLUMNS),
        "year_start": 2019,
        "year_end": 2023,
    },
    "multiseq_overlap_with_alias": {
        "file_name": "informal_acgate_multiseq_overlap_ready.csv",
        "sequence_columns": [LEGACY_SEQUENCE_COLUMN, *MULTISEQ_COLUMNS],
        "year_start": 2019,
        "year_end": 2023,
    },
    "multiseq_fullspan": {
        "file_name": "informal_acgate_multiseq_ready.csv",
        "sequence_columns": list(MULTISEQ_COLUMNS),
        "year_start": 2006,
        "year_end": 2023,
    },
    "single_overlap_region_proxy": {
        "file_name": "informal_acgate_single_feature_ready.csv",
        "sequence_columns": [LEGACY_SEQUENCE_COLUMN],
        "year_start": 2019,
        "year_end": 2023,
        "proxy_mode": "formal_region_summary",
        "proxy_columns": list(DERIVED_PROXY_COLUMNS),
        "static_columns": list(DERIVED_STATIC_COLUMNS),
        "derived_source_columns": [LEGACY_SEQUENCE_COLUMN, "x_seq_rydgdp_ratio"],
    },
    "single_fullspan_region_proxy": {
        "file_name": "informal_acgate_single_feature_ready.csv",
        "sequence_columns": [LEGACY_SEQUENCE_COLUMN],
        "year_start": 2006,
        "year_end": 2023,
        "proxy_mode": "income_region_summary",
        "proxy_columns": list(INCOME_PROXY_COLUMNS),
        "static_columns": list(INCOME_STATIC_COLUMNS),
        "derived_source_columns": [LEGACY_SEQUENCE_COLUMN],
    },
    "multiseq_overlap_region_proxy": {
        "file_name": "informal_acgate_multiseq_overlap_ready.csv",
        "sequence_columns": list(MULTISEQ_COLUMNS),
        "year_start": 2019,
        "year_end": 2023,
        "proxy_mode": "formal_region_summary",
        "proxy_columns": list(DERIVED_PROXY_COLUMNS),
        "static_columns": list(DERIVED_STATIC_COLUMNS),
        "derived_source_columns": [LEGACY_SEQUENCE_COLUMN, *MULTISEQ_COLUMNS],
    },
}

MISSING_POLICIES = {"error", "drop_any_missing_year"}
EPSILON = 1e-8


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


@dataclass(slots=True)
class InformalPanel:
    """Dense Informal panel tensors aligned with the CMDL model contract."""

    X_it: torch.Tensor
    p_i: torch.Tensor
    s_i: torch.Tensor
    Y_it: torch.Tensor
    entity_ids: torch.Tensor
    time_index: torch.Tensor
    entity_codes: list[str]
    entity_names: list[str]
    metadata: dict[str, Any]


def _resolve_bundle(feature_bundle: str) -> dict[str, Any]:
    normalized = str(feature_bundle).strip().lower()
    if normalized not in SUPPORTED_FEATURE_BUNDLES:
        raise ValueError(
            f"Unsupported Informal feature_bundle: {feature_bundle}. "
            f"Expected one of {sorted(SUPPORTED_FEATURE_BUNDLES)}"
        )
    bundle = dict(SUPPORTED_FEATURE_BUNDLES[normalized])
    bundle["name"] = normalized
    return bundle


def _resolve_csv_path(csv_path: str | Path | None, feature_bundle: str) -> Path:
    bundle = _resolve_bundle(feature_bundle)
    path = DEFAULT_INPUT_DIR / str(bundle["file_name"]) if csv_path is None else Path(csv_path)
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Informal input CSV not found: {resolved}")
    return resolved


def _require_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required Informal columns: {missing_columns}")


def _zscore_with_reference(values: pd.Series, reference_mask: pd.Series | np.ndarray) -> pd.Series:
    array = pd.to_numeric(values, errors="coerce").astype(float)
    mask = np.asarray(reference_mask, dtype=bool)
    if mask.shape[0] != len(array):
        raise ValueError("reference_mask must align with values")

    reference = array[mask]
    reference = reference[np.isfinite(reference)]
    if reference.size == 0:
        raise ValueError("reference subset for z-score is empty or non-finite")

    mean = float(reference.mean())
    std = float(reference.std(ddof=0))
    if not np.isfinite(std) or std < EPSILON:
        return pd.Series(np.zeros(len(array), dtype=np.float32), index=values.index)
    normalized = (array - mean) / std
    return pd.Series(normalized.astype(np.float32), index=values.index)


def _cross_section_zscore(values: pd.Series) -> pd.Series:
    array = pd.to_numeric(values, errors="coerce").astype(float)
    mean = float(array.mean())
    std = float(array.std(ddof=0))
    if not np.isfinite(std) or std < EPSILON:
        return pd.Series(np.zeros(len(array), dtype=np.float32), index=values.index)
    normalized = (array - mean) / std
    return pd.Series(normalized.astype(np.float32), index=values.index)


def _linear_trend(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
    mask = np.isfinite(array)
    if mask.sum() <= 1:
        return 0.0
    index = np.arange(array.shape[0], dtype=np.float64)[mask]
    target = array[mask].astype(np.float64)
    index_centered = index - index.mean()
    denominator = float(np.dot(index_centered, index_centered))
    if denominator < EPSILON:
        return 0.0
    target_centered = target - target.mean()
    return float(np.dot(index_centered, target_centered) / denominator)


def _safe_log1p(values: pd.Series) -> pd.Series:
    array = pd.to_numeric(values, errors="coerce").astype(float)
    return np.log1p(array.clip(lower=0.0))


def _resolve_first_available(frame: pd.DataFrame, candidates: list[str]) -> str:
    for column_name in candidates:
        if column_name in frame.columns:
            return column_name
    raise ValueError(f"None of the candidate columns are available: {candidates}")


def _is_contiguous_annual(years: list[int]) -> bool:
    if not years:
        return False
    return years == list(range(int(years[0]), int(years[-1]) + 1))


def _build_proxy_metadata(proxy_columns: list[str]) -> dict[str, Any]:
    anchor_proxy_index = 0
    return {
        "anchor_proxy_name": proxy_columns[anchor_proxy_index],
        "anchor_proxy_index": anchor_proxy_index,
        "anchor_expected_sign": -1.0,
        "auxiliary_proxy_names": [name for index, name in enumerate(proxy_columns) if index != anchor_proxy_index],
        "proxy_expected_signs": [-1.0 for _ in proxy_columns],
        "proxy_aggregate_name": "proxy_mean",
    }


def _derive_formal_region_proxy_static(
    frame: pd.DataFrame,
    stats_end_year: int,
) -> pd.DataFrame:
    """Build region-varying proxy/static summaries from train-window formal signals."""

    reference_frame = frame.loc[frame["year"] <= stats_end_year].copy()
    if reference_frame.empty:
        raise ValueError("Cannot derive formal-region proxies from an empty reference window")

    income_column = _resolve_first_available(reference_frame, ["x_seq_rpcyd_total_income", LEGACY_SEQUENCE_COLUMN])
    ratio_column = _resolve_first_available(reference_frame, ["x_seq_rydgdp_ratio"])
    capacity_column = _resolve_first_available(
        reference_frame,
        [
            "x_seq_rydgdp_grdp_per_capita",
            "x_seq_rydgdp_disposable_income_per_capita",
            LEGACY_SEQUENCE_COLUMN,
        ],
    )

    rows: list[dict[str, Any]] = []
    for entity_code, group in reference_frame.groupby("entity_code", sort=True):
        entity_frame = group.sort_values("year")
        income_log = _safe_log1p(entity_frame[income_column])
        capacity_log = _safe_log1p(entity_frame[capacity_column])
        income_diff = income_log.diff().dropna()
        rows.append(
            {
                "entity_code": str(entity_code),
                "proxy_formal_income_level": float(income_log.mean()),
                "proxy_formalization_ratio": float(pd.to_numeric(entity_frame[ratio_column], errors="coerce").mean()),
                "proxy_formal_capacity_signal": float(capacity_log.mean()),
                "static_formal_income_trend": _linear_trend(income_log),
                "static_formal_income_volatility": float(income_diff.std(ddof=0)) if len(income_diff) > 0 else 0.0,
            }
        )

    derived = pd.DataFrame(rows)
    if derived[[*DERIVED_PROXY_COLUMNS, *DERIVED_STATIC_COLUMNS]].isna().any().any():
        raise ValueError("Derived formal-region proxy/static columns contain NaN values")
    return derived


def _derive_income_region_proxy_static(
    frame: pd.DataFrame,
    stats_end_year: int,
) -> pd.DataFrame:
    """Build full-span region-varying proxy/static summaries from train-window income."""

    reference_frame = frame.loc[frame["year"] <= stats_end_year].copy()
    if reference_frame.empty:
        raise ValueError("Cannot derive income-region proxies from an empty reference window")

    income_column = _resolve_first_available(reference_frame, ["x_seq_rpcyd_total_income", LEGACY_SEQUENCE_COLUMN])
    rows: list[dict[str, Any]] = []
    for entity_code, group in reference_frame.groupby("entity_code", sort=True):
        entity_frame = group.sort_values("year")
        income_log = _safe_log1p(entity_frame[income_column])
        income_diff = income_log.diff().dropna()
        recent_window = income_log.tail(min(3, len(income_log)))
        growth_signal = float(income_log.iloc[-1] - income_log.iloc[0]) if len(income_log) > 1 else 0.0
        rows.append(
            {
                "entity_code": str(entity_code),
                "proxy_income_level": float(income_log.mean()),
                "proxy_income_recent_level": float(recent_window.mean()),
                "proxy_income_growth_signal": growth_signal,
                "static_income_trend": _linear_trend(income_log),
                "static_income_volatility": float(income_diff.std(ddof=0)) if len(income_diff) > 0 else 0.0,
            }
        )

    derived = pd.DataFrame(rows)
    if derived[[*INCOME_PROXY_COLUMNS, *INCOME_STATIC_COLUMNS]].isna().any().any():
        raise ValueError("Derived income-region proxy/static columns contain NaN values")
    return derived


def _build_audit_payload(
    raw_frame: pd.DataFrame,
    cleaned_frame: pd.DataFrame,
    sequence_columns: list[str],
    proxy_columns: list[str],
    static_columns: list[str],
    required_value_columns: list[str],
    dropped_years: list[int],
) -> dict[str, Any]:
    entity_summary = raw_frame.groupby("entity_code", sort=True)
    proxy_static_raw = cleaned_frame.groupby("entity_code", sort=True)[[*proxy_columns, *static_columns]].first()
    proxy_static_std = proxy_static_raw.std(axis=0, ddof=0).to_dict()
    zero_variance_columns = [
        column for column, value in proxy_static_std.items() if not np.isfinite(float(value)) or float(value) < EPSILON
    ]

    return {
        "required_missing_counts_before_policy": raw_frame[required_value_columns].isna().sum().astype(int).to_dict(),
        "dropped_years_due_to_missing_policy": [int(year) for year in dropped_years],
        "entity_count": int(cleaned_frame["entity_code"].nunique()),
        "year_count": int(cleaned_frame["year"].nunique()),
        "years": [int(year) for year in sorted(cleaned_frame["year"].unique().tolist())],
        "sequence_columns": list(sequence_columns),
        "proxy_columns": list(proxy_columns),
        "static_columns": list(static_columns),
        "proxy_static_entity_mean_std": {key: float(value) for key, value in proxy_static_std.items()},
        "zero_variance_proxy_static_columns": zero_variance_columns,
        "target_entity_mean_unique_count": int(raw_frame.groupby("entity_code")[TARGET_COLUMN].mean().nunique()),
        "warning_flags": {
            "proxy_static_cross_section_degenerate": bool(zero_variance_columns),
            "target_entity_mean_degenerate": bool(raw_frame.groupby("entity_code")[TARGET_COLUMN].mean().nunique() <= 1),
        },
    }


def build_informal_dataframe(
    csv_path: str | Path | None = None,
    feature_bundle: str = "multiseq_overlap",
    year_start: int | None = None,
    year_end: int | None = None,
    stats_end_year: int | None = None,
    missing_policy: str = "error",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a normalized long-form Informal dataframe and metadata payload."""

    if missing_policy not in MISSING_POLICIES:
        raise ValueError(f"missing_policy must be one of {sorted(MISSING_POLICIES)}, got {missing_policy}")

    bundle = _resolve_bundle(feature_bundle)
    sequence_columns = list(bundle["sequence_columns"])
    proxy_mode = str(bundle.get("proxy_mode", "original"))
    proxy_columns = list(bundle.get("proxy_columns", PROXY_COLUMNS))
    static_columns = list(bundle.get("static_columns", STATIC_COLUMNS))
    source_path = _resolve_csv_path(csv_path, feature_bundle)

    resolved_year_start = int(bundle["year_start"] if year_start is None else year_start)
    resolved_year_end = int(bundle["year_end"] if year_end is None else year_end)
    if resolved_year_end <= resolved_year_start:
        raise ValueError("year_end must be greater than year_start")
    if stats_end_year is None:
        stats_end_year = resolved_year_end
    stats_end_year = int(stats_end_year)
    if not resolved_year_start <= stats_end_year <= resolved_year_end:
        raise ValueError(
            f"stats_end_year must be within [{resolved_year_start}, {resolved_year_end}], got {stats_end_year}"
        )

    if proxy_mode == "original":
        source_proxy_columns = list(proxy_columns)
        source_static_columns = list(static_columns)
        derived_source_columns: list[str] = []
    elif proxy_mode in {"formal_region_summary", "income_region_summary"}:
        source_proxy_columns = []
        source_static_columns = []
        derived_source_columns = list(bundle.get("derived_source_columns", []))
    else:
        raise ValueError(f"Unsupported Informal proxy_mode: {proxy_mode}")

    required_columns = _unique_preserve_order(
        [
            "entity_code",
            "entity_name",
            "year",
            TARGET_COLUMN,
            *sequence_columns,
            *source_proxy_columns,
            *source_static_columns,
            *derived_source_columns,
        ]
    )
    dataframe = pd.read_csv(source_path)
    _require_columns(dataframe, required_columns)

    frame = dataframe.loc[:, required_columns].copy()
    frame["entity_code"] = frame["entity_code"].astype(str).str.strip()
    frame["entity_name"] = frame["entity_name"].fillna(frame["entity_code"]).astype(str)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.dropna(subset=["entity_code", "year"])
    frame["year"] = frame["year"].astype(int)
    frame = frame[(frame["year"] >= resolved_year_start) & (frame["year"] <= resolved_year_end)].copy()
    if frame.empty:
        raise ValueError("No Informal rows remain after year filtering")

    value_columns = _unique_preserve_order(
        [TARGET_COLUMN, *sequence_columns, *source_proxy_columns, *source_static_columns, *derived_source_columns]
    )
    for column_name in value_columns:
        frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")

    raw_filtered_frame = frame.copy()
    dropped_years: list[int] = []
    if missing_policy == "drop_any_missing_year":
        year_has_missing = frame.groupby("year")[value_columns].apply(lambda part: bool(part.isna().any().any()))
        dropped_years = [int(year) for year, has_missing in year_has_missing.items() if has_missing]
        if dropped_years:
            frame = frame.loc[~frame["year"].isin(dropped_years)].copy()
            if frame.empty:
                raise ValueError("All Informal years were dropped by missing_policy=drop_any_missing_year")
            stats_end_year = min(stats_end_year, int(frame["year"].max()))

    missing_counts = frame[value_columns].isna().sum()
    if int(missing_counts.sum()) > 0:
        raise ValueError(
            "Informal panel contains missing values in required columns after the selected missing policy: "
            f"{missing_counts[missing_counts > 0].astype(int).to_dict()}"
        )

    years = sorted(int(year) for year in frame["year"].unique().tolist())
    if not _is_contiguous_annual(years):
        raise ValueError(f"Informal years must be contiguous annual values, got {years}")
    if stats_end_year not in years:
        supported_stats_years = [year for year in years if year <= stats_end_year]
        if not supported_stats_years:
            raise ValueError("stats_end_year leaves no supported reference years")
        stats_end_year = int(max(supported_stats_years))

    normalized_entities: list[pd.DataFrame] = []
    for entity_code, group in frame.groupby("entity_code", sort=True):
        entity_frame = group.sort_values("year").groupby("year", as_index=False).last()
        entity_years = [int(year) for year in entity_frame["year"].tolist()]
        if entity_years != years:
            raise ValueError(f"Entity {entity_code} does not have a balanced Informal panel over {years}")

        reference_mask = entity_frame["year"].to_numpy(dtype=int) <= stats_end_year
        output = entity_frame[["entity_code", "entity_name", "year"]].copy()
        output[TARGET_COLUMN] = _zscore_with_reference(entity_frame[TARGET_COLUMN], reference_mask)
        for column_name in sequence_columns:
            output[column_name] = _zscore_with_reference(entity_frame[column_name], reference_mask)
        normalized_entities.append(output)

    normalized_frame = pd.concat(normalized_entities, ignore_index=True)

    reference_frame = frame.loc[frame["year"] <= stats_end_year]
    if proxy_mode == "original":
        proxy_static_raw = reference_frame.groupby("entity_code", as_index=False).agg(
            {"entity_name": "first", **{column: "mean" for column in [*proxy_columns, *static_columns]}}
        )
    elif proxy_mode == "formal_region_summary":
        proxy_static_raw = _derive_formal_region_proxy_static(frame, int(stats_end_year))
    elif proxy_mode == "income_region_summary":
        proxy_static_raw = _derive_income_region_proxy_static(frame, int(stats_end_year))
    else:
        raise ValueError(f"Unsupported Informal proxy_mode: {proxy_mode}")
    for column_name in [*proxy_columns, *static_columns]:
        proxy_static_raw[column_name] = _cross_section_zscore(proxy_static_raw[column_name])

    normalized_frame = normalized_frame.merge(
        proxy_static_raw[["entity_code", *proxy_columns, *static_columns]],
        on="entity_code",
        how="left",
    )
    normalized_frame = normalized_frame.sort_values(["entity_code", "year"]).reset_index(drop=True)

    audit = _build_audit_payload(
        raw_frame=raw_filtered_frame,
        cleaned_frame=normalized_frame,
        sequence_columns=sequence_columns,
        proxy_columns=proxy_columns,
        static_columns=static_columns,
        required_value_columns=value_columns,
        dropped_years=dropped_years,
    )

    metadata = {
        "domain": "informal",
        "source_path": str(source_path),
        "feature_bundle": str(bundle["name"]),
        "target_column": TARGET_COLUMN,
        "seq_feature_columns": sequence_columns,
        "proxy_columns": proxy_columns,
        **_build_proxy_metadata(proxy_columns),
        "static_columns": static_columns,
        "years": years,
        "year_start": int(years[0]),
        "year_end": int(years[-1]),
        "stats_end_year": int(stats_end_year),
        "missing_policy": missing_policy,
        "proxy_mode": proxy_mode,
        "audit": audit,
    }
    metadata["audit"]["proxy_mode"] = proxy_mode
    if proxy_mode == "formal_region_summary":
        metadata["audit"]["proxy_construction"] = {
            "fit_window": [int(years[0]), int(stats_end_year)],
            "uses_target_column": False,
            "description": (
                "Region-varying AC/proxy signals are derived from train-window formal sequence summaries: "
                "formal income level, formalization ratio, and formal capacity signal."
            ),
        }
    elif proxy_mode == "income_region_summary":
        metadata["audit"]["proxy_construction"] = {
            "fit_window": [int(years[0]), int(stats_end_year)],
            "uses_target_column": False,
            "description": (
                "Region-varying AC/proxy signals are derived from train-window income/RPCYD summaries: "
                "income level, recent income level, and income growth signal."
            ),
        }
    return normalized_frame, metadata


def _validate_loader_shape_contract(cfg: Any, seq_features: int, n_proxies: int, static_dim: int) -> None:
    if cfg.seq_features != seq_features:
        raise ValueError(f"Informal loader emits {seq_features} sequence features, but cfg expects {cfg.seq_features}")
    if cfg.n_proxies != n_proxies:
        raise ValueError(f"Informal loader emits {n_proxies} proxies, but cfg expects {cfg.n_proxies}")
    if cfg.static_dim != static_dim:
        raise ValueError(f"Informal loader emits {static_dim} static features, but cfg expects {cfg.static_dim}")


def load_informal_panel(
    csv_path: str | Path | None = None,
    cfg: Any | None = None,
    feature_bundle: str = "multiseq_overlap",
    year_start: int | None = None,
    year_end: int | None = None,
    stats_end_year: int | None = None,
    missing_policy: str = "error",
) -> InformalPanel:
    """Load Informal data into dense CMDL tensors."""

    dataframe, metadata = build_informal_dataframe(
        csv_path=csv_path,
        feature_bundle=feature_bundle,
        year_start=year_start,
        year_end=year_end,
        stats_end_year=stats_end_year,
        missing_policy=missing_policy,
    )
    sequence_columns = list(metadata["seq_feature_columns"])
    proxy_columns = list(metadata["proxy_columns"])
    static_columns = list(metadata["static_columns"])
    if cfg is not None:
        _validate_loader_shape_contract(cfg, len(sequence_columns), len(proxy_columns), len(static_columns))

    entity_codes = sorted(dataframe["entity_code"].unique().tolist())
    years = sorted(int(year) for year in dataframe["year"].unique().tolist())
    n_entities = len(entity_codes)
    seq_length = len(years)
    X_it = np.zeros((n_entities, seq_length, len(sequence_columns)), dtype=np.float32)
    Y_it = np.zeros((n_entities, seq_length), dtype=np.float32)
    p_i = np.zeros((n_entities, len(proxy_columns)), dtype=np.float32)
    s_i = np.zeros((n_entities, len(static_columns)), dtype=np.float32)
    entity_names: list[str] = []

    for entity_index, entity_code in enumerate(entity_codes):
        entity_frame = dataframe.loc[dataframe["entity_code"] == entity_code].sort_values("year")
        if len(entity_frame) != seq_length:
            raise ValueError(f"Entity {entity_code} does not have a balanced Informal panel")
        entity_names.append(str(entity_frame["entity_name"].iloc[0]))
        for feature_index, column_name in enumerate(sequence_columns):
            X_it[entity_index, :, feature_index] = entity_frame[column_name].to_numpy(dtype=np.float32)
        Y_it[entity_index, :] = entity_frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
        for proxy_index, column_name in enumerate(proxy_columns):
            p_i[entity_index, proxy_index] = float(entity_frame[column_name].iloc[0])
        for static_index, column_name in enumerate(static_columns):
            s_i[entity_index, static_index] = float(entity_frame[column_name].iloc[0])

    metadata = dict(metadata)
    metadata["n_entities"] = n_entities
    metadata["seq_length"] = seq_length
    return InformalPanel(
        X_it=torch.tensor(X_it, dtype=torch.float32),
        p_i=torch.tensor(p_i, dtype=torch.float32),
        s_i=torch.tensor(s_i, dtype=torch.float32),
        Y_it=torch.tensor(Y_it, dtype=torch.float32),
        entity_ids=torch.arange(n_entities, dtype=torch.long),
        time_index=torch.arange(seq_length, dtype=torch.long),
        entity_codes=entity_codes,
        entity_names=entity_names,
        metadata=metadata,
    )


def slice_informal_panel(panel: InformalPanel, start_year: int, end_year: int) -> InformalPanel:
    """Slice a contiguous year window from one Informal panel."""

    if end_year < start_year:
        raise ValueError(f"end_year must be >= start_year, got {start_year}..{end_year}")
    years = [int(year) for year in panel.metadata["years"]]
    if start_year not in years or end_year not in years:
        raise ValueError(f"Requested slice {start_year}..{end_year} is outside the Informal panel range")

    start_index = years.index(start_year)
    end_index = years.index(end_year)
    positions = list(range(start_index, end_index + 1))
    sliced_years = years[start_index : end_index + 1]

    metadata = dict(panel.metadata)
    metadata["years"] = sliced_years
    metadata["year_start"] = int(sliced_years[0])
    metadata["year_end"] = int(sliced_years[-1])
    metadata["seq_length"] = len(sliced_years)

    return InformalPanel(
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
    panel: InformalPanel,
    max_lag: int,
    train_end_year: int,
    val_end_year: int,
) -> tuple[InformalPanel, InformalPanel, InformalPanel]:
    """Create train/validation/test panels with lag-context overlap."""

    years = [int(year) for year in panel.metadata["years"]]
    full_start_year = int(years[0])
    full_end_year = int(years[-1])
    val_start_year = int(train_end_year) + 1
    test_start_year = int(val_end_year) + 1
    if not full_start_year < train_end_year < val_end_year < full_end_year:
        raise ValueError(
            "Expected full_start_year < train_end_year < val_end_year < full_end_year, "
            f"got {full_start_year} < {train_end_year} < {val_end_year} < {full_end_year}"
        )
    if val_start_year - max_lag < full_start_year or test_start_year - max_lag < full_start_year:
        raise ValueError("Not enough historical context before validation/test windows for the requested max_lag")

    train_panel = slice_informal_panel(panel, full_start_year, int(train_end_year))
    val_panel = slice_informal_panel(panel, val_start_year - max_lag, int(val_end_year))
    test_panel = slice_informal_panel(panel, test_start_year - max_lag, full_end_year)
    for split_name, split_panel in [("train", train_panel), ("val", val_panel), ("test", test_panel)]:
        if split_panel.X_it.shape[1] <= max_lag:
            raise ValueError(f"{split_name} split length must exceed max_lag; got {split_panel.X_it.shape[1]}")
    return train_panel, val_panel, test_panel


def get_prediction_years(panel: InformalPanel, max_lag: int) -> list[int]:
    """Return calendar years aligned with post-warm-up predictions."""

    years = [int(year) for year in panel.metadata["years"]]
    if len(years) <= max_lag:
        raise ValueError(f"Panel length {len(years)} must exceed max_lag {max_lag}")
    return years[max_lag:]


def move_panel_to_device(panel: InformalPanel, device: torch.device) -> InformalPanel:
    """Move every tensor field of an Informal panel onto one device."""

    return InformalPanel(
        X_it=panel.X_it.to(device),
        p_i=panel.p_i.to(device),
        s_i=panel.s_i.to(device),
        Y_it=panel.Y_it.to(device),
        entity_ids=panel.entity_ids.to(device),
        time_index=panel.time_index.to(device),
        entity_codes=list(panel.entity_codes),
        entity_names=list(panel.entity_names),
        metadata=dict(panel.metadata),
    )


__all__ = [
    "DEFAULT_INPUT_DIR",
    "InformalPanel",
    "MISSING_POLICIES",
    "PROXY_COLUMNS",
    "STATIC_COLUMNS",
    "DERIVED_PROXY_COLUMNS",
    "DERIVED_STATIC_COLUMNS",
    "INCOME_PROXY_COLUMNS",
    "INCOME_STATIC_COLUMNS",
    "SUPPORTED_FEATURE_BUNDLES",
    "build_informal_dataframe",
    "build_temporal_splits",
    "get_prediction_years",
    "load_informal_panel",
    "move_panel_to_device",
    "slice_informal_panel",
]
