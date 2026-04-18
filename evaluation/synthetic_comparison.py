"""Utilities for consolidating synthetic CMDL, baseline, and ablation outputs.

统一读取 synthetic CMDL、plain LSTM baseline 与 ablation 的 summary.json，
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

_BASE_COLUMNS = [
    "family",
    "display_name",
    "experiment",
    "scenario",
    "tracking_backend",
    "best_epoch",
    "best_val_task_loss",
    "task_loss",
    "recon_loss",
    "effective_kstar_mae",
    "effective_kstar_spearman_rho",
    "effective_lag_entropy_mean",
    "effective_lag_peak_accuracy",
    "proxy_signal_r2",
    "z_signal_spearman_rho",
    "model",
    "variant",
    "run_dir",
    "output_root",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary_paths(output_root: Path) -> list[Path]:
    if not output_root.exists():
        return []
    return sorted(path for path in output_root.glob("*/summary.json") if path.is_file())


def _display_name(family: str, experiment: str, model: str | None, variant: str | None) -> str:
    if family == "plain_lstm":
        return "Plain LSTM"
    if family == "ablation":
        return _ABLATION_LABELS.get(variant or "", variant or experiment)
    return "CMDL"


def _normalize_summary(summary_path: Path, output_root: Path, family: str) -> dict[str, Any]:
    payload = _read_json(summary_path)
    metrics = dict(payload.get("metrics", {}))
    experiment = payload.get("experiment", summary_path.parent.name)
    model = payload.get("model")
    variant = payload.get("variant")

    row: dict[str, Any] = {
        "family": family,
        "display_name": _display_name(family, experiment, model, variant),
        "experiment": experiment,
        "scenario": payload.get("scenario"),
        "tracking_backend": payload.get("tracking_backend"),
        "best_epoch": payload.get("best_epoch"),
        "best_val_task_loss": payload.get("best_val_task_loss"),
        "task_loss": metrics.get("task_loss"),
        "recon_loss": metrics.get("recon_loss"),
        "model": model or ("cmdl" if family == "cmdl" else None),
        "variant": variant,
        "run_dir": str(summary_path.parent),
        "output_root": str(output_root),
        "proxy_signal_r2": metrics.get("proxy_recon_r2"),
        "z_signal_spearman_rho": metrics.get("z_spearman_rho"),
    }

    if family == "plain_lstm":
        row["effective_kstar_mae"] = metrics.get("posthoc_kstar_mae")
        row["effective_kstar_spearman_rho"] = metrics.get("posthoc_kstar_spearman_rho")
        row["effective_lag_entropy_mean"] = metrics.get("posthoc_profile_entropy_mean")
        row["effective_lag_peak_accuracy"] = metrics.get("posthoc_profile_peak_accuracy")
    else:
        row["effective_kstar_mae"] = metrics.get("kstar_mae")
        row["effective_kstar_spearman_rho"] = metrics.get("kstar_spearman_rho")
        row["effective_lag_entropy_mean"] = metrics.get("omega_entropy_mean")
        row["effective_lag_peak_accuracy"] = metrics.get("omega_peak_accuracy")

    row.update(metrics)
    return row


def load_summary_runs(output_root: Path | str | None, family: str) -> pd.DataFrame:
    """Load all per-run summary.json files from one synthetic output root.

    从单个 synthetic 输出根目录中读取所有按 run 存放的 summary.json。
    """

    if output_root is None:
        return pd.DataFrame(columns=_BASE_COLUMNS)

    root = Path(output_root)
    rows = [_normalize_summary(path, root, family) for path in _summary_paths(root)]
    if not rows:
        return pd.DataFrame(columns=_BASE_COLUMNS)

    frame = pd.DataFrame(rows)
    for column in _BASE_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame


def build_synthetic_comparison(
    cmdl_root: Path | str | None = None,
    baseline_root: Path | str | None = None,
    ablation_root: Path | str | None = None,
) -> pd.DataFrame:
    """Combine CMDL, baseline, and ablation runs into one normalized table.

    将 CMDL、baseline 与 ablation 的结果合并为统一格式的总表。
    """

    frames = [
        load_summary_runs(cmdl_root, family="cmdl"),
        load_summary_runs(baseline_root, family="plain_lstm"),
        load_summary_runs(ablation_root, family="ablation"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=_BASE_COLUMNS)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    sort_columns = [column for column in ["scenario", "family", "display_name", "experiment"] if column in combined.columns]
    combined = combined.sort_values(sort_columns, na_position="last").reset_index(drop=True)

    for column in _BASE_COLUMNS:
        if column not in combined.columns:
            combined[column] = None
    return combined


def build_recovery_table(comparison_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the lag-recovery comparison view used in notebooks.

    返回 notebook 中展示的滞后恢复对比表。
    """

    columns = [
        "display_name",
        "family",
        "scenario",
        "experiment",
        "task_loss",
        "effective_kstar_mae",
        "effective_kstar_spearman_rho",
        "effective_lag_entropy_mean",
        "effective_lag_peak_accuracy",
        "best_epoch",
    ]
    if comparison_frame.empty:
        return pd.DataFrame(columns=columns)

    recovery = comparison_frame[comparison_frame["effective_kstar_mae"].notna()].copy()
    if recovery.empty:
        return pd.DataFrame(columns=columns)

    recovery = recovery.sort_values(
        ["scenario", "effective_kstar_mae", "display_name"],
        na_position="last",
    )
    return recovery.loc[:, columns].reset_index(drop=True)


def build_identification_table(comparison_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the proxy/z identification comparison view used in notebooks.

    返回 notebook 中展示的 proxy 与 z 识别性对比表。
    """

    columns = [
        "display_name",
        "family",
        "scenario",
        "experiment",
        "task_loss",
        "proxy_signal_r2",
        "z_signal_spearman_rho",
        "best_epoch",
    ]
    if comparison_frame.empty:
        return pd.DataFrame(columns=columns)

    identification = comparison_frame[
        comparison_frame["proxy_signal_r2"].notna() | comparison_frame["z_signal_spearman_rho"].notna()
    ].copy()
    if identification.empty:
        return pd.DataFrame(columns=columns)

    identification = identification.sort_values(
        ["scenario", "proxy_signal_r2", "z_signal_spearman_rho", "display_name"],
        ascending=[True, False, False, True],
        na_position="last",
    )
    return identification.loc[:, columns].reset_index(drop=True)


__all__ = [
    "build_identification_table",
    "build_recovery_table",
    "build_synthetic_comparison",
    "load_summary_runs",
]