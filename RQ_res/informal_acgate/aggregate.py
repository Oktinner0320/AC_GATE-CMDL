"""Aggregate isolated Informal experiment summaries into a compact CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
RQ_RES_ROOT = PACKAGE_ROOT.parent
if str(RQ_RES_ROOT) not in sys.path:
    sys.path.insert(0, str(RQ_RES_ROOT))


DEFAULT_OUTPUT_DIR = RQ_RES_ROOT / "outputs" / "informal_acgate"
SELECTED_TEST_METRICS = [
    "mse",
    "mae",
    "r2",
    "r2_delta_vs_train_mean",
    "r2_delta_vs_entity_mean",
    "r2_delta_vs_persistence",
    "r2_delta_vs_panel_ols",
    "r2_delta_vs_grouped_ardl",
    "kstar_mean",
    "kstar_std",
    "kstar_proxy_spearman_adjusted_rho",
    "kstar_proxy_mean_spearman_adjusted_rho",
    "lag_gate_sensitivity_range",
    "omega_entropy_mean",
    "omega_top1_share",
    "proxy_recon_r2",
    "proxy_anchor_recon_r2",
    "baseline_grouped_ardl_r2",
    "baseline_grouped_ardl_effective_lag_mean",
]


def _read_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _get_nested(payload: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _scenario_label(feature_bundle: str) -> str:
    if "fullspan" in feature_bundle:
        return "fullspan_region_proxy" if "region_proxy" in feature_bundle else "fullspan_original_proxy"
    if "overlap" in feature_bundle:
        return "overlap_region_proxy" if "region_proxy" in feature_bundle else "overlap_original_proxy"
    return "informal_unspecified"


def _effective_prediction_count(years: list[Any], max_lag: Any, n_entities: Any) -> int | None:
    try:
        lag_count = int(max_lag)
        entity_count = int(n_entities)
    except (TypeError, ValueError):
        return None
    return int(max(0, len(years) - lag_count) * entity_count)


def summary_to_row(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    data = summary.get("data", {})
    test_metrics = _get_nested(summary, ["metrics", "test"], {}) or {}
    audit = data.get("audit", {}) or {}
    warnings = audit.get("warning_flags", {}) or {}
    proxy_construction = audit.get("proxy_construction", {}) or {}
    feature_bundle = str(data.get("feature_bundle", ""))
    max_lag = _get_nested(summary, ["config", "max_lag"])
    n_entities = data.get("n_entities")
    train_years = data.get("train_years", [])
    val_years = data.get("val_years", [])
    test_years = data.get("test_years", [])
    row: dict[str, Any] = {
        "experiment": summary.get("experiment"),
        "scenario": _scenario_label(feature_bundle),
        "model": summary.get("model"),
        "ablation": summary.get("ablation"),
        "feature_bundle": feature_bundle,
        "seed": _get_nested(summary, ["config", "seed"]),
        "max_lag": max_lag,
        "seq_features": _get_nested(summary, ["config", "seq_features"]),
        "best_epoch": summary.get("best_epoch"),
        "year_start": data.get("year_start"),
        "year_end": data.get("year_end"),
        "stats_end_year": data.get("stats_end_year"),
        "train_end_year": data.get("train_end_year"),
        "val_end_year": data.get("val_end_year"),
        "missing_policy": data.get("missing_policy"),
        "train_years": ",".join(str(year) for year in train_years),
        "val_years": ",".join(str(year) for year in val_years),
        "test_years": ",".join(str(year) for year in test_years),
        "train_effective_samples": _effective_prediction_count(train_years, max_lag, n_entities),
        "val_effective_samples": _effective_prediction_count(val_years, max_lag, n_entities),
        "test_effective_samples": _effective_prediction_count(test_years, max_lag, n_entities),
        "proxy_mode": audit.get("proxy_mode"),
        "proxy_construction_fit_window": json.dumps(proxy_construction.get("fit_window"), ensure_ascii=True),
        "proxy_construction_uses_target_column": proxy_construction.get("uses_target_column"),
        "proxy_construction_description": proxy_construction.get("description"),
        "proxy_static_cross_section_degenerate": warnings.get("proxy_static_cross_section_degenerate"),
        "target_entity_mean_degenerate": warnings.get("target_entity_mean_degenerate"),
        "zero_variance_proxy_static_columns": ",".join(audit.get("zero_variance_proxy_static_columns", [])),
        "summary_path": str(summary_path),
    }
    for metric_name in SELECTED_TEST_METRICS:
        row[f"test_{metric_name}"] = test_metrics.get(metric_name)
    return row


def aggregate_summaries(output_dir: str | Path = DEFAULT_OUTPUT_DIR, output_csv: str | Path | None = None) -> pd.DataFrame:
    root = Path(output_dir).resolve()
    summaries = sorted(root.glob("**/summary.json"))
    rows = [summary_to_row(_read_summary(path), path) for path in summaries]
    dataframe = pd.DataFrame(rows)
    if output_csv is None:
        output_csv = root / "comparison.csv"
    output_path = Path(output_csv).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate isolated Informal experiment summaries.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = aggregate_summaries(args.output_dir, args.output_csv)
    print(f"Aggregated {len(dataframe)} Informal runs.")


if __name__ == "__main__":
    main()
