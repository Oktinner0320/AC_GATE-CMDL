"""Aggregate one-shot Informal RQ improvement matrix outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
RQ_RES_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = RQ_RES_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(RQ_RES_ROOT) not in sys.path:
    sys.path.insert(0, str(RQ_RES_ROOT))

from informal_acgate.aggregate import summary_to_row
from informal_acgate.experiment_matrix import DEFAULT_REPORT_ROOT


SUMMARY_METRICS = [
    "test_mse",
    "test_mae",
    "test_r2",
    "test_kstar_std",
    "test_kstar_proxy_spearman_adjusted_rho",
    "test_kstar_proxy_mean_spearman_adjusted_rho",
    "test_lag_gate_sensitivity_range",
    "test_omega_entropy_mean",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)


def infer_report_dir(matrix_dir: str | Path) -> Path:
    matrix_path = Path(matrix_dir).resolve()
    return DEFAULT_REPORT_ROOT / matrix_path.name


def _run_group(row: pd.Series) -> str:
    model = str(row.get("model", ""))
    ablation = str(row.get("ablation", ""))
    if model == "plain_lstm":
        return "Plain LSTM"
    if ablation == "no_ac_encoder":
        return "No AC Encoder"
    if ablation == "uniform_lag":
        return "Uniform Lag"
    if ablation == "no_recon_regularization":
        return "No Recon"
    return "CMDL"


def _matrix_metadata(summary_path: Path) -> dict[str, Any]:
    metadata_path = summary_path.parent / "matrix_task.json"
    if not metadata_path.exists():
        return {}
    return read_json(metadata_path)


def _status_rows(matrix_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(matrix_dir.glob("**/matrix_task_status.json")):
        payload = read_json(path)
        if payload.get("status") == "failed":
            rows.append(payload)
    return rows


def _enrich_row(row: dict[str, Any], summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    metadata = _matrix_metadata(summary_path)
    variant = metadata.get("variant", {}) if metadata else {}
    run_spec = metadata.get("run_spec", {}) if metadata else {}
    config = summary.get("config", {}) or {}
    training_controls = summary.get("diagnostics", {}).get("training_controls", {}) or {}
    data_audit = summary.get("data", {}).get("audit", {}) or {}
    proxy_perturbation = data_audit.get("proxy_perturbation") or training_controls.get("proxy_perturbation") or {}
    if isinstance(proxy_perturbation, dict):
        proxy_perturbation_mode = proxy_perturbation.get("mode", "none")
    else:
        proxy_perturbation_mode = str(proxy_perturbation or "none")

    row.update(
        {
            "matrix_name": metadata.get("matrix_name"),
            "variant_id": variant.get("variant_id", summary_path.parent.parent.name),
            "track": variant.get("track", "external_or_legacy"),
            "matrix_scenario": variant.get("scenario"),
            "interpretation": variant.get("interpretation"),
            "run_spec": run_spec.get("name"),
            "matrix_output_dir": metadata.get("output_dir", str(summary_path.parent.parent)),
            "run_dir": str(summary_path.parent),
            "proxy_perturbation": proxy_perturbation_mode,
            "d_model": config.get("d_model"),
            "dropout": config.get("dropout"),
            "temperature": config.get("temperature"),
            "lambda_omega_entropy": training_controls.get("lambda_omega_entropy", config.get("lambda_omega_entropy")),
            "omega_entropy_min": training_controls.get("omega_entropy_min", config.get("omega_entropy_min")),
            "omega_entropy_max": training_controls.get("omega_entropy_max", config.get("omega_entropy_max")),
            "evidence_scope": data_audit.get("evidence_scope"),
            "sequence_transform": json.dumps(data_audit.get("sequence_transform", {}), ensure_ascii=True),
        }
    )
    row["run_group"] = _run_group(pd.Series(row))
    return row


def _aggregate_rows(matrix_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(matrix_dir.glob("**/summary.json")):
        summary = read_json(summary_path)
        row = summary_to_row(summary, summary_path)
        rows.append(_enrich_row(row, summary, summary_path))
    return pd.DataFrame(rows)


def _summary_table(dataframe: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(columns=[*group_columns, "run_count", "seed_count"])
    aggregations: dict[str, list[str]] = {metric: ["mean", "std", "min", "max"] for metric in SUMMARY_METRICS if metric in dataframe.columns}
    grouped = dataframe.groupby(group_columns, dropna=False).agg(aggregations)
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    counts = (
        dataframe.groupby(group_columns, dropna=False)
        .agg(run_count=("experiment", "count"), seed_count=("seed", pd.Series.nunique))
        .reset_index()
    )
    output = grouped.merge(counts, on=group_columns, how="left")
    for column in output.columns:
        if output[column].dtype.kind in {"f", "c"}:
            output[column] = output[column].replace([np.inf, -np.inf], np.nan)
    return output


def aggregate_matrix(matrix_dir: str | Path, report_dir: str | Path | None = None) -> dict[str, Path]:
    matrix_path = Path(matrix_dir).resolve()
    if report_dir is None:
        report_path = infer_report_dir(matrix_path)
    else:
        report_path = Path(report_dir).resolve()
    report_path.mkdir(parents=True, exist_ok=True)

    all_runs = _aggregate_rows(matrix_path)
    all_runs_path = report_path / "all_runs.csv"
    all_runs.to_csv(all_runs_path, index=False)

    variant_summary = _summary_table(
        all_runs,
        ["track", "variant_id", "run_group", "feature_bundle", "max_lag", "proxy_perturbation"],
    )
    variant_summary_path = report_path / "variant_summary.csv"
    variant_summary.to_csv(variant_summary_path, index=False)

    track_summary = _summary_table(all_runs, ["track", "run_group"])
    track_summary_path = report_path / "track_summary.csv"
    track_summary.to_csv(track_summary_path, index=False)

    mechanism_frame = all_runs.loc[all_runs.get("run_group", pd.Series(dtype=str)).eq("CMDL")].copy()
    mechanism_summary = _summary_table(mechanism_frame, ["track", "variant_id", "proxy_perturbation"])
    mechanism_summary_path = report_path / "mechanism_summary.csv"
    mechanism_summary.to_csv(mechanism_summary_path, index=False)

    failed_runs = pd.DataFrame(
        _status_rows(matrix_path),
        columns=["status", "updated_at", "variant_id", "track", "run_spec", "seed", "experiment_name", "run_dir", "error", "traceback"],
    )
    failed_runs_path = report_path / "failed_runs.csv"
    failed_runs.to_csv(failed_runs_path, index=False)

    paths = {
        "all_runs_csv": all_runs_path,
        "variant_summary_csv": variant_summary_path,
        "track_summary_csv": track_summary_path,
        "mechanism_summary_csv": mechanism_summary_path,
        "failed_runs_csv": failed_runs_path,
    }
    write_json(report_path / "aggregate_manifest.json", {key: str(value) for key, value in paths.items()})
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Informal RQ one-shot improvement matrix outputs.")
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--report-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = aggregate_matrix(args.matrix_dir, args.report_dir)
    print(f"Aggregated matrix outputs into {Path(paths['all_runs_csv']).parent}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()