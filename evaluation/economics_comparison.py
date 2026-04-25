"""Utilities for consolidating economics CMDL, baseline, and ablation outputs.

统一读取 economics CMDL、plain LSTM baseline 与 ablation 的 summary.json，
并生成可直接在 notebook 中展示的对比表。
"""

from __future__ import annotations

import json
import math
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
    "feature_bundle",
    "seq_feature_names",
    "proxy_feature_names",
    "anchor_proxy_name",
    "anchor_proxy_index",
    "anchor_expected_sign",
    "auxiliary_proxy_names",
    "proxy_aggregate_name",
    "static_feature_names",
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
    "omega_transform",
    "lambda_omega_entropy",
    "omega_entropy_min",
    "omega_entropy_max",
    "lambda_z_anchor",
    "z_anchor_target_sign",
    "lag_bias_strength",
    "lstm_layers",
    "dropout",
    "noise_std",
    "best_epoch",
    "best_val_task_loss",
    "matched_init_to_full_cmdl",
    "causal_ablation_validity",
    "grad_clip_mode",
    "recon_loss_mode",
    "anchor_recon_weight",
    "reconstruction_detach",
    "omega_entropy_control",
    "z_anchor_control",
    "proxy_refit_status",
    "proxy_refit_applied",
    "proxy_metric_interpretable",
    "proxy_refit_reason",
    "proxy_design_rank",
    "proxy_design_columns",
    "proxy_latent_std",
    "proxy_target_std",
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
    if family == "grouped_ardl":
        return "Grouped ARDL"
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
    if family == "grouped_ardl":
        return "grouped_ardl"
    if family == "plain_lstm":
        return "plain_lstm"
    if family == "ablation":
        return "cmdl_ablation"
    return "cmdl"


def _lag_method(payload: dict[str, Any], family: str) -> str:
    if family == "grouped_ardl":
        return "grouped_distributed_lag_ols"
    if family == "plain_lstm":
        return str(payload.get("posthoc_lag_method", "lag_occlusion"))
    return "learned_omega"


def _finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    return value


def _bool_flag(value: Any) -> bool | None:
    normalized = _finite_or_none(value)
    if normalized is None:
        return None
    if isinstance(normalized, bool):
        return normalized
    return bool(float(normalized) >= 0.5)


def _normalize_split_metrics(
    split: str,
    metrics: dict[str, Any],
    family: str,
    proxy_metric_interpretable: bool | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        f"{split}_total_loss": _finite_or_none(metrics.get("total_loss")),
        f"{split}_task_loss": _finite_or_none(metrics.get("task_loss")),
        f"{split}_recon_loss": _finite_or_none(metrics.get("recon_loss")),
        f"{split}_mse": _finite_or_none(metrics.get("mse")),
        f"{split}_mae": _finite_or_none(metrics.get("mae")),
        f"{split}_r2": _finite_or_none(metrics.get("r2")),
    }

    if family == "plain_lstm":
        kstar_metric_valid = _bool_flag(
            metrics.get("posthoc_kstar_proxy_metric_valid", metrics.get("kstar_proxy_metric_valid"))
        )
        kstar_rho = _finite_or_none(metrics.get("posthoc_kstar_proxy_spearman_rho"))
        kstar_p = _finite_or_none(metrics.get("posthoc_kstar_proxy_spearman_p"))
        kstar_adjusted = _finite_or_none(metrics.get("posthoc_kstar_proxy_spearman_adjusted_rho"))
        if kstar_metric_valid is False:
            kstar_rho = None
            kstar_p = None
            kstar_adjusted = None
        row.update(
            {
                f"{split}_proxy_signal_r2": None,
                f"{split}_proxy_signal_metric_valid": None,
                f"{split}_effective_kstar_proxy_spearman_rho": kstar_rho,
                f"{split}_effective_kstar_proxy_spearman_p": kstar_p,
                f"{split}_effective_kstar_proxy_spearman_adjusted_rho": kstar_adjusted,
                f"{split}_effective_kstar_proxy_mean_spearman_rho": _finite_or_none(
                    metrics.get("posthoc_kstar_proxy_mean_spearman_rho")
                ),
                f"{split}_effective_kstar_proxy_mean_spearman_adjusted_rho": _finite_or_none(
                    metrics.get("posthoc_kstar_proxy_mean_spearman_adjusted_rho")
                ),
                f"{split}_effective_kstar_mean": _finite_or_none(metrics.get("posthoc_kstar_mean")),
                f"{split}_effective_kstar_std": _finite_or_none(metrics.get("posthoc_kstar_std")),
                f"{split}_effective_lag_entropy_mean": _finite_or_none(metrics.get("lag_profile_entropy_mean")),
                f"{split}_effective_lag_entropy_std": _finite_or_none(metrics.get("omega_entropy_std")),
                f"{split}_effective_lag_top1_share": _finite_or_none(metrics.get("omega_top1_share")),
                f"{split}_effective_kstar_proxy_metric_valid": kstar_metric_valid,
            }
        )
        return row

    if family == "grouped_ardl":
        row.update(
            {
                f"{split}_proxy_signal_r2": None,
                f"{split}_proxy_signal_metric_valid": None,
                f"{split}_effective_kstar_proxy_spearman_rho": None,
                f"{split}_effective_kstar_proxy_spearman_p": None,
                f"{split}_effective_kstar_proxy_spearman_adjusted_rho": None,
                f"{split}_effective_kstar_proxy_mean_spearman_rho": None,
                f"{split}_effective_kstar_proxy_mean_spearman_adjusted_rho": None,
                f"{split}_effective_kstar_mean": _finite_or_none(metrics.get("effective_lag_mean")),
                f"{split}_effective_kstar_std": None,
                f"{split}_effective_lag_entropy_mean": None,
                f"{split}_effective_lag_entropy_std": None,
                f"{split}_effective_lag_top1_share": None,
                f"{split}_effective_kstar_proxy_metric_valid": None,
                f"{split}_grouped_ardl_best_lag_mean": _finite_or_none(metrics.get("best_lag_mean")),
                f"{split}_grouped_ardl_effective_lag_mean": _finite_or_none(metrics.get("effective_lag_mean")),
                f"{split}_grouped_ardl_group_count": _finite_or_none(metrics.get("group_count")),
            }
        )
        return row

    proxy_metric_valid = _bool_flag(metrics.get("proxy_metric_valid"))
    if proxy_metric_interpretable is False:
        proxy_metric_valid = False
    kstar_metric_valid = _bool_flag(metrics.get("kstar_proxy_metric_valid"))
    proxy_signal = _finite_or_none(metrics.get("proxy_recon_r2"))
    kstar_rho = _finite_or_none(metrics.get("kstar_proxy_spearman_rho"))
    kstar_p = _finite_or_none(metrics.get("kstar_proxy_spearman_p"))
    kstar_adjusted = _finite_or_none(metrics.get("kstar_proxy_spearman_adjusted_rho"))
    if proxy_metric_valid is False:
        proxy_signal = None
    if kstar_metric_valid is False:
        kstar_rho = None
        kstar_p = None
        kstar_adjusted = None

    row.update(
        {
            f"{split}_proxy_signal_r2": proxy_signal,
            f"{split}_proxy_signal_metric_valid": proxy_metric_valid,
            f"{split}_effective_kstar_proxy_spearman_rho": kstar_rho,
            f"{split}_effective_kstar_proxy_spearman_p": kstar_p,
            f"{split}_effective_kstar_proxy_spearman_adjusted_rho": kstar_adjusted,
            f"{split}_effective_kstar_proxy_mean_spearman_rho": _finite_or_none(
                metrics.get("kstar_proxy_mean_spearman_rho")
            ),
            f"{split}_effective_kstar_proxy_mean_spearman_adjusted_rho": _finite_or_none(
                metrics.get("kstar_proxy_mean_spearman_adjusted_rho")
            ),
            f"{split}_effective_kstar_mean": _finite_or_none(metrics.get("kstar_mean")),
            f"{split}_effective_kstar_std": _finite_or_none(metrics.get("kstar_std")),
            f"{split}_effective_lag_entropy_mean": _finite_or_none(metrics.get("omega_entropy_mean")),
            f"{split}_effective_lag_entropy_std": _finite_or_none(metrics.get("omega_entropy_std")),
            f"{split}_effective_lag_top1_share": _finite_or_none(metrics.get("omega_top1_share")),
            f"{split}_effective_kstar_proxy_metric_valid": kstar_metric_valid,
            f"{split}_z_std": _finite_or_none(metrics.get("z_std")),
            f"{split}_z_proxy_spearman_adjusted_rho": _finite_or_none(
                metrics.get("z_proxy_spearman_adjusted_rho")
            ),
            f"{split}_z_anchor_adjusted_rho": _finite_or_none(metrics.get("z_anchor_adjusted_rho")),
            f"{split}_omega_entropy_penalty": _finite_or_none(metrics.get("omega_entropy_penalty")),
            f"{split}_omega_entropy_band_violation_share": _finite_or_none(
                metrics.get("omega_entropy_band_violation_share")
            ),
            f"{split}_z_anchor_loss": _finite_or_none(metrics.get("z_anchor_loss")),
            f"{split}_lag_gate_sensitivity_slope": _finite_or_none(metrics.get("lag_gate_sensitivity_slope")),
            f"{split}_lag_gate_sensitivity_range": _finite_or_none(metrics.get("lag_gate_sensitivity_range")),
        }
    )
    return row


def _normalize_summary(summary_path: Path, output_root: Path, family: str) -> dict[str, Any]:
    payload = _read_json(summary_path)
    config = dict(payload.get("config", {}))
    data = dict(payload.get("data", {}))
    diagnostics = dict(payload.get("diagnostics", {}))
    ablation = dict(diagnostics.get("ablation") or {})
    proxy_refit = dict(diagnostics.get("proxy_refit") or {})
    training_controls = dict(diagnostics.get("training_controls") or {})
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
        "feature_bundle": data.get("feature_bundle"),
        "seq_feature_names": ",".join(data.get("seq_feature_columns", [])),
        "proxy_feature_names": ",".join(data.get("proxy_columns", [])),
        "anchor_proxy_name": data.get("anchor_proxy_name"),
        "anchor_proxy_index": data.get("anchor_proxy_index"),
        "anchor_expected_sign": data.get("anchor_expected_sign"),
        "auxiliary_proxy_names": ",".join(data.get("auxiliary_proxy_names", [])),
        "proxy_aggregate_name": data.get("proxy_aggregate_name"),
        "static_feature_names": ",".join(data.get("static_columns", [])),
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
        "omega_transform": config.get("omega_transform", training_controls.get("omega_transform")),
        "lambda_omega_entropy": config.get("lambda_omega_entropy", training_controls.get("lambda_omega_entropy")),
        "omega_entropy_min": config.get("omega_entropy_min", training_controls.get("omega_entropy_min")),
        "omega_entropy_max": config.get("omega_entropy_max", training_controls.get("omega_entropy_max")),
        "lambda_z_anchor": config.get("lambda_z_anchor", training_controls.get("lambda_z_anchor")),
        "z_anchor_target_sign": config.get("z_anchor_target_sign", training_controls.get("z_anchor_target_sign")),
        "lag_bias_strength": config.get("lag_bias_strength"),
        "lstm_layers": config.get("lstm_layers"),
        "dropout": config.get("dropout"),
        "noise_std": config.get("noise_std"),
        "best_epoch": payload.get("best_epoch"),
        "best_val_task_loss": payload.get("best_val_task_loss"),
        "matched_init_to_full_cmdl": ablation.get("matched_init_to_full_cmdl"),
        "causal_ablation_validity": ablation.get("causal_ablation_validity"),
        "grad_clip_mode": training_controls.get("grad_clip_mode"),
        "recon_loss_mode": training_controls.get("recon_loss_mode"),
        "anchor_recon_weight": training_controls.get("anchor_recon_weight"),
        "reconstruction_detach": training_controls.get("reconstruction_detach"),
        "omega_entropy_control": training_controls.get("lambda_omega_entropy", config.get("lambda_omega_entropy")),
        "z_anchor_control": training_controls.get("lambda_z_anchor", config.get("lambda_z_anchor")),
        "proxy_refit_status": proxy_refit.get("status"),
        "proxy_refit_applied": proxy_refit.get("applied"),
        "proxy_metric_interpretable": proxy_refit.get("metrics_interpretable"),
        "proxy_refit_reason": proxy_refit.get("reason"),
        "proxy_design_rank": proxy_refit.get("design_rank"),
        "proxy_design_columns": proxy_refit.get("design_columns"),
        "proxy_latent_std": proxy_refit.get("latent_std"),
        "proxy_target_std": proxy_refit.get("proxy_std"),
        "run_dir": str(summary_path.parent),
        "output_root": str(output_root),
    }

    metrics_by_split = payload.get("metrics", {})
    for split in _SPLITS:
        split_metrics = dict(metrics_by_split.get(split, {}))
        for key, value in split_metrics.items():
            row[f"{split}_{key}"] = value
        row.update(
            _normalize_split_metrics(
                split,
                split_metrics,
                family,
                proxy_metric_interpretable=proxy_refit.get("metrics_interpretable"),
            )
        )

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


def _first_numeric(frame: pd.DataFrame, display_name: str, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    rows = frame.loc[frame["display_name"] == display_name]
    if rows.empty:
        return None
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.iloc[0])


def _delta_answer(*deltas: float | None) -> str:
    observed = [value for value in deltas if value is not None]
    if not observed:
        return "unknown"
    positive_count = sum(value > 0.0 for value in observed)
    if positive_count == len(observed):
        return "yes"
    if positive_count > 0:
        return "partial"
    return "no"


def _named_proxy_columns(comparison_frame: pd.DataFrame, split: str, metric_prefix: str) -> list[str]:
    prefix = f"{split}_{metric_prefix}_proxy_"
    suffix = "_spearman_adjusted_rho"
    columns: list[str] = []
    for column in comparison_frame.columns:
        if not column.startswith(prefix) or not column.endswith(suffix):
            continue
        if column == f"{split}_{metric_prefix}_proxy_spearman_adjusted_rho":
            continue
        if column == f"{split}_{metric_prefix}_proxy_mean_spearman_adjusted_rho":
            continue
        proxy_name = column.removeprefix(prefix).removesuffix(suffix)
        if proxy_name == "mean" or proxy_name.isdigit():
            continue
        columns.append(column)
    return sorted(columns)


def build_economics_comparison(
    cmdl_root: Path | str | None = None,
    baseline_root: Path | str | None = None,
    ablation_root: Path | str | None = None,
    grouped_ardl_root: Path | str | None = None,
) -> pd.DataFrame:
    """Combine economics CMDL, baseline, and ablation runs into one table.

    将 economics CMDL、baseline 与 ablation 的结果合并为统一格式的总表。
    """

    frames = [
        _load_summary_runs(cmdl_root, family="cmdl"),
        _load_summary_runs(baseline_root, family="plain_lstm"),
        _load_summary_runs(ablation_root, family="ablation"),
        _load_summary_runs(grouped_ardl_root, family="grouped_ardl"),
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
        for column in ["target_column", "feature_bundle", "family", "display_name", "seed", "experiment"]
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
        "feature_bundle",
        "best_epoch",
        "best_val_task_loss",
        f"{split}_r2",
        f"{split}_baseline_persistence_r2",
        f"{split}_baseline_panel_ols_r2",
        f"{split}_baseline_grouped_ardl_r2",
        f"{split}_baseline_best_simple_r2",
        f"{split}_r2_delta_vs_persistence",
        f"{split}_r2_delta_vs_panel_ols",
        f"{split}_r2_delta_vs_grouped_ardl",
        f"{split}_mae",
        f"{split}_mse",
    ]
    task_table = _ensure_columns(comparison_frame, columns)
    if task_table.empty:
        return task_table

    r2_column = f"{split}_r2"
    grouped_r2_column = f"{split}_baseline_grouped_ardl_r2"
    grouped_delta_column = f"{split}_r2_delta_vs_grouped_ardl"
    grouped_rows = task_table[task_table["family"].eq("grouped_ardl")]
    grouped_key_columns = [column for column in ["target_column", "feature_bundle"] if column in task_table.columns]
    if not grouped_rows.empty and r2_column in task_table.columns:
        grouped_means = grouped_rows.groupby(grouped_key_columns, dropna=False)[r2_column].mean().to_dict()
        for row_index, row in task_table.iterrows():
            if row.get("family") == "grouped_ardl":
                continue
            key = tuple(row.get(column) for column in grouped_key_columns)
            if len(grouped_key_columns) == 1:
                key = key[0]
            grouped_r2 = grouped_means.get(key)
            row_r2 = row.get(r2_column)
            if grouped_r2 is None or pd.isna(grouped_r2):
                continue
            if grouped_r2_column in task_table.columns and pd.isna(row.get(grouped_r2_column)):
                task_table.at[row_index, grouped_r2_column] = grouped_r2
            if (
                grouped_delta_column in task_table.columns
                and pd.isna(row.get(grouped_delta_column))
                and row_r2 is not None
                and not pd.isna(row_r2)
            ):
                task_table.at[row_index, grouped_delta_column] = float(row_r2) - float(grouped_r2)

    return task_table.sort_values(
            ["target_column", "feature_bundle", f"{split}_r2", f"{split}_mae", "display_name", "seed"],
            ascending=[True, True, False, True, True, True],
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
        "feature_bundle",
        "lag_method",
        "anchor_proxy_name",
        "anchor_expected_sign",
        "proxy_refit_status",
        "proxy_metric_interpretable",
        f"{split}_effective_kstar_proxy_spearman_rho",
        f"{split}_effective_kstar_proxy_spearman_adjusted_rho",
        f"{split}_effective_kstar_proxy_mean_spearman_rho",
        f"{split}_effective_kstar_proxy_mean_spearman_adjusted_rho",
        f"{split}_effective_kstar_proxy_metric_valid",
        f"{split}_effective_kstar_mean",
        f"{split}_effective_kstar_std",
        f"{split}_effective_lag_entropy_mean",
        f"{split}_effective_lag_entropy_std",
        f"{split}_effective_lag_top1_share",
        f"{split}_z_std",
        f"{split}_z_proxy_spearman_adjusted_rho",
        f"{split}_z_anchor_adjusted_rho",
        f"{split}_omega_entropy_penalty",
        f"{split}_omega_entropy_band_violation_share",
        f"{split}_z_anchor_loss",
        f"{split}_lag_gate_sensitivity_slope",
        f"{split}_lag_gate_sensitivity_range",
        f"{split}_proxy_signal_r2",
        f"{split}_proxy_signal_metric_valid",
        "omega_transform",
        "lambda_omega_entropy",
        "omega_entropy_min",
        "omega_entropy_max",
        "lambda_z_anchor",
        "z_anchor_target_sign",
    ]
    interpretability = _ensure_columns(comparison_frame, columns)
    if interpretability.empty:
        return interpretability

    return interpretability.sort_values(
        [
            "target_column",
            "feature_bundle",
            f"{split}_effective_kstar_proxy_spearman_rho",
            f"{split}_proxy_signal_r2",
            "display_name",
            "seed",
        ],
            ascending=[True, True, False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_per_proxy_alignment_table(
    comparison_frame: pd.DataFrame,
    split: str = "test",
    display_names: list[str] | tuple[str, ...] | None = ("CMDL",),
) -> pd.DataFrame:
    """Build a long-form per-proxy adjusted-rho table for anchor audits.

    生成逐 proxy 的 adjusted rho 长表，避免 notebook 手写列名解析。
    """

    columns = [
        "display_name",
        "family",
        "variant",
        "seed",
        "experiment",
        "target_column",
        "feature_bundle",
        "anchor_proxy_name",
        "anchor_expected_sign",
        "proxy_name",
        "adjusted_rho",
    ]
    if comparison_frame.empty:
        return pd.DataFrame(columns=columns)

    named_columns = _named_proxy_columns(comparison_frame, split, "kstar")
    if not named_columns:
        return pd.DataFrame(columns=columns)

    allowed_names = set(display_names) if display_names is not None else None
    rows: list[dict[str, Any]] = []
    for _, source_row in comparison_frame.iterrows():
        display_name = source_row.get("display_name")
        if allowed_names is not None and display_name not in allowed_names:
            continue
        for column in named_columns:
            adjusted_rho = _finite_or_none(source_row.get(column))
            if adjusted_rho is None:
                continue
            proxy_name = column.removeprefix(f"{split}_kstar_proxy_").removesuffix(
                "_spearman_adjusted_rho"
            )
            rows.append(
                {
                    "display_name": display_name,
                    "family": source_row.get("family"),
                    "variant": source_row.get("variant"),
                    "seed": source_row.get("seed"),
                    "experiment": source_row.get("experiment"),
                    "target_column": source_row.get("target_column"),
                    "feature_bundle": source_row.get("feature_bundle"),
                    "anchor_proxy_name": source_row.get("anchor_proxy_name"),
                    "anchor_expected_sign": source_row.get("anchor_expected_sign"),
                    "proxy_name": proxy_name,
                    "adjusted_rho": adjusted_rho,
                }
            )

    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.sort_values(
        ["target_column", "feature_bundle", "display_name", "seed", "proxy_name"],
        na_position="last",
    ).reset_index(drop=True)


def build_mechanism_summary_table(comparison_frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Aggregate task and mechanism diagnostics across seeds.

    汇总多 seed 的预测校准与 AC-GATE 机制指标，用于论文表格草稿。
    """

    group_columns = [
        "display_name",
        "family",
        "variant",
        "target_column",
        "feature_bundle",
        "anchor_proxy_name",
        "anchor_expected_sign",
    ]
    value_columns = [
        f"{split}_r2",
        f"{split}_r2_delta_vs_persistence",
        f"{split}_r2_delta_vs_panel_ols",
        f"{split}_r2_delta_vs_grouped_ardl",
        f"{split}_effective_kstar_proxy_spearman_adjusted_rho",
        f"{split}_effective_kstar_proxy_mean_spearman_adjusted_rho",
        f"{split}_effective_kstar_std",
        f"{split}_effective_lag_entropy_mean",
        f"{split}_effective_lag_top1_share",
        f"{split}_z_std",
        f"{split}_z_proxy_spearman_adjusted_rho",
        f"{split}_z_anchor_adjusted_rho",
        f"{split}_omega_entropy_penalty",
        f"{split}_omega_entropy_band_violation_share",
        f"{split}_z_anchor_loss",
        f"{split}_lag_gate_sensitivity_range",
        f"{split}_proxy_signal_r2",
    ]
    columns = [*group_columns, "n_seeds"]
    for column in value_columns:
        columns.extend([f"{column}_mean", f"{column}_std"])

    if comparison_frame.empty:
        return pd.DataFrame(columns=columns)

    existing_group_columns = [column for column in group_columns if column in comparison_frame.columns]
    existing_value_columns = [column for column in value_columns if column in comparison_frame.columns]
    if not existing_group_columns or not existing_value_columns:
        return pd.DataFrame(columns=columns)

    working = comparison_frame.loc[:, [*existing_group_columns, "seed", *existing_value_columns]].copy()
    for column in existing_value_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    grouped = working.groupby(existing_group_columns, dropna=False)
    summary = grouped[existing_value_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    n_seeds = grouped["seed"].nunique(dropna=True).reset_index(name="n_seeds")
    summary = summary.merge(n_seeds, on=existing_group_columns, how="left")

    for column in columns:
        if column not in summary.columns:
            summary[column] = None
    return summary.loc[:, columns].sort_values(
        ["target_column", "feature_bundle", "display_name"],
        na_position="last",
    ).reset_index(drop=True)


def build_mechanism_result_log(comparison_frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Build a compact mechanism-first yes/partial/no result log.

    生成机制优先的结论日志，把 forecast 当作校准证据而不是唯一 scoreboard。
    """

    columns = ["layer", "question", "answer", "evidence"]
    if comparison_frame.empty:
        return pd.DataFrame(columns=columns)

    cmdl_r2 = _first_numeric(comparison_frame, "CMDL", f"{split}_r2")
    plain_r2 = _first_numeric(comparison_frame, "Plain LSTM", f"{split}_r2")
    cmdl_adjusted_rho = _first_numeric(
        comparison_frame,
        "CMDL",
        f"{split}_effective_kstar_proxy_spearman_adjusted_rho",
    )
    cmdl_lag_range = _first_numeric(comparison_frame, "CMDL", f"{split}_lag_gate_sensitivity_range")
    cmdl_z_std = _first_numeric(comparison_frame, "CMDL", f"{split}_z_std")
    cmdl_delta_persistence = _first_numeric(
        comparison_frame,
        "CMDL",
        f"{split}_r2_delta_vs_persistence",
    )
    cmdl_delta_panel_ols = _first_numeric(
        comparison_frame,
        "CMDL",
        f"{split}_r2_delta_vs_panel_ols",
    )
    cmdl_delta_grouped_ardl = _first_numeric(
        comparison_frame,
        "CMDL",
        f"{split}_r2_delta_vs_grouped_ardl",
    )
    if cmdl_delta_grouped_ardl is None and cmdl_r2 is not None:
        grouped_ardl_r2 = _first_numeric(comparison_frame, "Grouped ARDL", f"{split}_r2")
        if grouped_ardl_r2 is not None:
            cmdl_delta_grouped_ardl = cmdl_r2 - grouped_ardl_r2
    no_ac_kstar_std = _first_numeric(
        comparison_frame,
        "No AC Encoder",
        f"{split}_effective_kstar_std",
    )
    uniform_top1_share = _first_numeric(
        comparison_frame,
        "Uniform Lag",
        f"{split}_effective_lag_top1_share",
    )
    per_proxy_table = build_per_proxy_alignment_table(comparison_frame, split=split)
    per_proxy_min = None
    if not per_proxy_table.empty:
        per_proxy_values = pd.to_numeric(per_proxy_table["adjusted_rho"], errors="coerce").dropna()
        if not per_proxy_values.empty:
            per_proxy_min = float(per_proxy_values.min())

    rows = [
        {
            "layer": "forecast_calibration",
            "question": "Does CMDL beat the matched LSTM?",
            "answer": "yes" if cmdl_r2 is not None and plain_r2 is not None and cmdl_r2 > plain_r2 else "no",
            "evidence": f"CMDL {split}_r2={cmdl_r2}, Plain LSTM {split}_r2={plain_r2}.",
        },
        {
            "layer": "simple_baseline_calibration",
            "question": "Is CMDL above the simple calibrated baselines?",
            "answer": _delta_answer(cmdl_delta_persistence, cmdl_delta_panel_ols, cmdl_delta_grouped_ardl),
            "evidence": (
                f"delta_vs_persistence={cmdl_delta_persistence}, "
                f"delta_vs_panel_ols={cmdl_delta_panel_ols}, "
                f"delta_vs_grouped_ardl={cmdl_delta_grouped_ardl}."
            ),
        },
        {
            "layer": "ac_gate_mechanism",
            "question": "Does the anchor-adjusted lag-proxy direction support the expected sign?",
            "answer": "yes" if cmdl_adjusted_rho is not None and cmdl_adjusted_rho > 0.0 else "no",
            "evidence": f"CMDL adjusted rho={cmdl_adjusted_rho}; positive means the expected anchor sign is satisfied.",
        },
        {
            "layer": "ac_gate_per_proxy",
            "question": "Are all named proxy adjusted correlations aligned?",
            "answer": "yes" if per_proxy_min is not None and per_proxy_min > 0.0 else "no",
            "evidence": f"minimum named per-proxy adjusted rho={per_proxy_min}.",
        },
        {
            "layer": "ac_gate_heterogeneity",
            "question": "Is the learned lag gate non-degenerate?",
            "answer": "yes" if (cmdl_lag_range or 0.0) > 0.0 and (cmdl_z_std or 0.0) > 0.0 else "no",
            "evidence": f"lag_gate_sensitivity_range={cmdl_lag_range}, z_std={cmdl_z_std}.",
        },
        {
            "layer": "ablation_guard",
            "question": "Do degenerate controls expose the heterogeneity boundary?",
            "answer": "yes" if no_ac_kstar_std == 0.0 or uniform_top1_share == 1.0 else "no",
            "evidence": f"No AC kstar_std={no_ac_kstar_std}, Uniform Lag top1_share={uniform_top1_share}.",
        },
    ]
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "build_economics_comparison",
    "build_interpretability_table",
    "build_mechanism_result_log",
    "build_mechanism_summary_table",
    "build_per_proxy_alignment_table",
    "build_task_table",
]