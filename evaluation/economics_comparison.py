"""Utilities for consolidating economics CMDL, baseline, and ablation outputs.

统一读取 economics CMDL、plain LSTM baseline 与 ablation 的 summary.json，
并生成可直接在 notebook 中展示的对比表。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


_ABLATION_LABELS = {
    "no_ac_encoder": "No AC Encoder",
    "uniform_lag": "Uniform Lag",
    "no_recon_regularization": "No Recon Regularization",
}

_SPLITS = ("train", "val", "test")
_IDENTITY_COLUMNS = [
    "family",
    "display_name",
    "experiment",
    "model",
    "variant",
    "lag_method",
    "tracking_backend",
    "device",
    "domain",
    "scenario",
    "seed",
    "target_column",
    "source_path",
    "stats_end_year",
    "year_start",
    "year_end",
    "train_end_year",
    "val_end_year",
    "train_year_start",
    "train_year_end",
    "val_year_start",
    "val_year_end",
    "test_year_start",
    "test_year_end",
    "n_entities",
    "full_seq_length",
    "max_lag",
    "d_model",
    "seq_features",
    "n_proxies",
    "static_dim",
    "lambda_r",
    "effective_lambda_r",
    "temperature",
    "lag_bias_strength",
    "lstm_layers",
    "dropout",
    "noise_std",
    "best_epoch",
    "best_val_task_loss",
    "run_dir",
    "output_root",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_paths(output_root: Path | str | None) -> list[Path]:
    if output_root is None:
        return []

    root = Path(output_root)
    if root.is_file() and root.name == "summary.json":
        return [root.resolve()]

    if not root.exists():
        return []

    direct_summary = root / "summary.json"
    if direct_summary.is_file():
        return [direct_summary.resolve()]

    return sorted(path.resolve() for path in root.glob("*/summary.json") if path.is_file())


def _display_name(family: str, experiment: str, variant: str | None) -> str:
    if family == "plain_lstm":
        return "Plain LSTM"
    if family == "ablation":
        return _ABLATION_LABELS.get(variant or "", variant or experiment)
    return "CMDL"


def _year_bounds(years: Any) -> tuple[int | None, int | None]:
    if not isinstance(years, list) or not years:
        return None, None
    return int(years[0]), int(years[-1])


def _family_model_name(payload: dict[str, Any], family: str) -> str:
    if payload.get("model"):
        return str(payload["model"])
    if family == "plain_lstm":
        return "plain_lstm"
    if family == "ablation":
        return "cmdl_ablation"
    return "cmdl"


def _lag_method(payload: dict[str, Any], family: str) -> str:
    if family == "plain_lstm":
        return str(payload.get("posthoc_lag_method", "lag_occlusion"))
    return "learned_omega"


def _normalize_split_metrics(split: str, metrics: dict[str, Any], family: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        f"{split}_total_loss": metrics.get("total_loss"),
        f"{split}_task_loss": metrics.get("task_loss"),
        f"{split}_recon_loss": metrics.get("recon_loss"),
        f"{split}_mse": metrics.get("mse"),
        f"{split}_mae": metrics.get("mae"),
        f"{split}_r2": metrics.get("r2"),
    }

    if family == "plain_lstm":
        row.update(
            {
                f"{split}_proxy_signal_r2": None,
                f"{split}_effective_kstar_proxy_spearman_rho": metrics.get("posthoc_kstar_proxy_spearman_rho"),
                f"{split}_effective_kstar_proxy_spearman_p": metrics.get("posthoc_kstar_proxy_spearman_p"),
                f"{split}_effective_kstar_mean": metrics.get("posthoc_kstar_mean"),
                f"{split}_effective_kstar_std": metrics.get("posthoc_kstar_std"),
                f"{split}_effective_lag_entropy_mean": metrics.get("lag_profile_entropy_mean"),
            }
        )
        return row

    row.update(
        {
            f"{split}_proxy_signal_r2": metrics.get("proxy_recon_r2"),
            f"{split}_effective_kstar_proxy_spearman_rho": metrics.get("kstar_proxy_spearman_rho"),
            f"{split}_effective_kstar_proxy_spearman_p": metrics.get("kstar_proxy_spearman_p"),
            f"{split}_effective_kstar_mean": metrics.get("kstar_mean"),
            f"{split}_effective_kstar_std": metrics.get("kstar_std"),
            f"{split}_effective_lag_entropy_mean": metrics.get("omega_entropy_mean"),
        }
    )
    return row


def _normalize_summary(summary_path: Path, output_root: Path, family: str) -> dict[str, Any]:
    payload = _read_json(summary_path)
    config = dict(payload.get("config", {}))
    data = dict(payload.get("data", {}))
    train_year_start, train_year_end = _year_bounds(data.get("train_years"))
    val_year_start, val_year_end = _year_bounds(data.get("val_years"))
    test_year_start, test_year_end = _year_bounds(data.get("test_years"))
    variant = payload.get("variant")
    experiment = str(payload.get("experiment", summary_path.parent.name))

    row: dict[str, Any] = {
        "family": family,
        "display_name": _display_name(family, experiment, variant),
        "experiment": experiment,
        "model": _family_model_name(payload, family),
        "variant": variant,
        "lag_method": _lag_method(payload, family),
        "tracking_backend": payload.get("tracking_backend"),
        "device": payload.get("device"),
        "domain": config.get("domain"),
        "scenario": payload.get("scenario", config.get("scenario")),
        "seed": config.get("seed"),
        "target_column": data.get("target_column"),
        "source_path": data.get("source_path"),
        "stats_end_year": data.get("stats_end_year"),
        "year_start": data.get("year_start"),
        "year_end": data.get("year_end"),
        "train_end_year": data.get("train_end_year"),
        "val_end_year": data.get("val_end_year"),
        "train_year_start": train_year_start,
        "train_year_end": train_year_end,
        "val_year_start": val_year_start,
        "val_year_end": val_year_end,
        "test_year_start": test_year_start,
        "test_year_end": test_year_end,
        "n_entities": data.get("n_entities", config.get("n_entities")),
        "full_seq_length": data.get("full_seq_length", config.get("seq_length")),
        "max_lag": config.get("max_lag"),
        "d_model": config.get("d_model"),
        "seq_features": config.get("seq_features"),
        "n_proxies": config.get("n_proxies"),
        "static_dim": config.get("static_dim"),
        "lambda_r": config.get("lambda_r"),
        "effective_lambda_r": payload.get("effective_lambda_r", config.get("lambda_r")),
        "temperature": config.get("temperature"),
        "lag_bias_strength": config.get("lag_bias_strength"),
        "lstm_layers": config.get("lstm_layers"),
        "dropout": config.get("dropout"),
        "noise_std": config.get("noise_std"),
        "best_epoch": payload.get("best_epoch"),
        "best_val_task_loss": payload.get("best_val_task_loss"),
        "run_dir": str(summary_path.parent),
        "output_root": str(output_root),
    }

    metrics_by_split = payload.get("metrics", {})
    for split in _SPLITS:
        split_metrics = dict(metrics_by_split.get(split, {}))
        for key, value in split_metrics.items():
            row[f"{split}_{key}"] = value
        row.update(_normalize_split_metrics(split, split_metrics, family))

    return row


def _load_summary_runs(output_root: Path | str | None, family: str) -> pd.DataFrame:
    summary_paths = _summary_paths(output_root)
    if not summary_paths:
        return pd.DataFrame(columns=_IDENTITY_COLUMNS)

    if output_root is None:
        return pd.DataFrame(columns=_IDENTITY_COLUMNS)

    root = Path(output_root).resolve()
    rows = [_normalize_summary(path, root, family) for path in summary_paths]
    frame = pd.DataFrame(rows)
    for column in _IDENTITY_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)

    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = None
    return result.loc[:, columns]


def build_economics_comparison(
    cmdl_root: Path | str | None = None,
    baseline_root: Path | str | None = None,
    ablation_root: Path | str | None = None,
) -> pd.DataFrame:
    """Combine economics CMDL, baseline, and ablation runs into one table.

    将 economics CMDL、baseline 与 ablation 的结果合并为统一格式的总表。
    """

    frames = [
        _load_summary_runs(cmdl_root, family="cmdl"),
        _load_summary_runs(baseline_root, family="plain_lstm"),
        _load_summary_runs(ablation_root, family="ablation"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=_IDENTITY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for frame in frames:
        rows.extend(frame.to_dict(orient="records"))
    combined = pd.DataFrame(rows)
    sort_columns = [
        column
        for column in ["target_column", "family", "display_name", "seed", "experiment"]
        if column in combined.columns
    ]
    combined = combined.sort_values(sort_columns, na_position="last").reset_index(drop=True)

    for column in _IDENTITY_COLUMNS:
        if column not in combined.columns:
            combined[column] = None
    return combined


def build_task_table(comparison_frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Build the forecast-comparison table used in the economics notebook.

    生成 economics notebook 使用的任务预测对比表。
    """

    columns = [
        "display_name",
        "family",
        "variant",
        "seed",
        "experiment",
        "target_column",
        "best_epoch",
        "best_val_task_loss",
        f"{split}_r2",
        f"{split}_mae",
        f"{split}_mse",
    ]
    task_table = _ensure_columns(comparison_frame, columns)
    if task_table.empty:
        return task_table

    return task_table.sort_values(
        ["target_column", f"{split}_r2", f"{split}_mae", "display_name", "seed"],
        ascending=[True, False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_interpretability_table(comparison_frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Build the lag/proxy diagnostics table used in the economics notebook.

    生成 economics notebook 使用的 lag/proxy 诊断对比表。
    """

    columns = [
        "display_name",
        "family",
        "variant",
        "seed",
        "experiment",
        "target_column",
        "lag_method",
        f"{split}_effective_kstar_proxy_spearman_rho",
        f"{split}_effective_kstar_mean",
        f"{split}_effective_kstar_std",
        f"{split}_effective_lag_entropy_mean",
        f"{split}_proxy_signal_r2",
    ]
    interpretability = _ensure_columns(comparison_frame, columns)
    if interpretability.empty:
        return interpretability

    return interpretability.sort_values(
        [
            "target_column",
            f"{split}_effective_kstar_proxy_spearman_rho",
            f"{split}_proxy_signal_r2",
            "display_name",
            "seed",
        ],
        ascending=[True, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


__all__ = [
    "build_economics_comparison",
    "build_interpretability_table",
    "build_task_table",
]