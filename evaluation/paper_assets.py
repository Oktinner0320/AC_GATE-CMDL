"""Build paper-facing tables and figures from existing CMDL experiment outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation.economics_comparison import (
    build_economics_comparison,
    build_mechanism_result_log as build_economics_result_log,
    build_significance_tables as build_economics_significance_tables,
)
from evaluation.energy_comparison import (
    build_energy_comparison,
    build_mechanism_result_log as build_energy_result_log,
    build_significance_tables as build_energy_significance_tables,
)
from evaluation.synthetic_comparison import build_significance_tables as build_synthetic_significance_tables
from evaluation.synthetic_comparison import build_synthetic_comparison
from evaluation.stratified_kstar import build_economics_stratifiers, build_energy_stratifiers
from evaluation.verdict_matrix import build_verdict_matrix
from visualization.paper_figures import (
    plot_case_study_scatter,
    plot_seed_distribution,
    plot_stratified_seed_distribution,
    plot_workflow_overview,
)


STRATIFIER_DISPLAY_LABELS = {
    "hc_mean_train": "Human capital (training-window mean)",
    "log_gdp_per_worker_train": "GDP per worker (training-window mean, log)",
    "log_capital_per_worker_train": "Capital per worker (training-window mean, log)",
    "rule_of_law_train": "Rule of law (training-window mean)",
    "government_effectiveness_train": "Government effectiveness (training-window mean)",
    "log_gdp_per_capita_train": "GDP per capita (training-window mean, log)",
}

DOMAIN_DISPLAY_LABELS = {
    "economics": "Economics",
    "energy": "Energy",
}

CASE_STUDY_POINT_COLORS = {
    "economics": "#66ccff",
    "energy": "#cc55ee",
}


@dataclass(frozen=True)
class PaperPaths:
    workspace_root: Path
    plan_name: str = "complete_20seed_20260426"

    @property
    def synthetic_root(self) -> Path:
        return self.workspace_root / "outputs" / "notebook_synthetic" / self.plan_name

    @property
    def economics_root(self) -> Path:
        return self.workspace_root / "outputs" / "notebook_economics" / self.plan_name

    @property
    def energy_root(self) -> Path:
        return self.workspace_root / "outputs" / "notebook_energy" / self.plan_name

    @property
    def proxy_shuffle_root(self) -> Path:
        return self.workspace_root / "outputs" / "negative_controls" / "proxy_shuffle_20seed_20260511"


@dataclass
class PaperArtifactBundle:
    tables: dict[str, pd.DataFrame]
    figure_paths: dict[str, Path]
    manifest_path: Path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_positive_share(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float((numeric > 0.0).mean())


def _safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def _safe_std(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.std(ddof=1))


def _bootstrap_mean_ci(values: pd.Series, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if numeric.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, numeric.size, size=(n_boot, numeric.size))
    sample_means = numeric[sample_index].mean(axis=1)
    return (
        float(sample_means.mean()),
        float(np.quantile(sample_means, 0.025)),
        float(np.quantile(sample_means, 0.975)),
        int(numeric.size),
    )


def _aggregate_synthetic_summary(comparison_frame: pd.DataFrame) -> pd.DataFrame:
    if comparison_frame.empty:
        return pd.DataFrame()

    group_columns = ["display_name", "family", "variant", "scenario"]
    value_columns = [
        "task_loss",
        "effective_kstar_mae",
        "effective_kstar_spearman_rho",
        "effective_lag_entropy_mean",
        "effective_lag_peak_accuracy",
        "proxy_signal_r2",
        "z_signal_spearman_rho",
    ]
    grouped = comparison_frame.groupby(group_columns, dropna=False)
    rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        row = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        for column in value_columns:
            numeric = pd.to_numeric(group.get(column), errors="coerce")
            row[f"{column}_mean"] = float(numeric.mean(skipna=True)) if not numeric.dropna().empty else float("nan")
            row[f"{column}_std"] = float(numeric.std(skipna=True)) if not numeric.dropna().empty else float("nan")
        row["n_seeds"] = int(pd.to_numeric(group.get("seed"), errors="coerce").dropna().nunique())
        row["kstar_positive_seed_share"] = _safe_positive_share(group["effective_kstar_spearman_rho"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario", "display_name"], na_position="last").reset_index(drop=True)


def _aggregate_realdata_compact(comparison_frame: pd.DataFrame) -> pd.DataFrame:
    if comparison_frame.empty:
        return pd.DataFrame()

    group_columns = ["display_name", "family", "variant", "target_column", "feature_bundle"]
    rows: list[dict[str, Any]] = []
    for keys, group in comparison_frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        row["n_seeds"] = int(pd.to_numeric(group.get("seed"), errors="coerce").dropna().nunique())
        row["test_r2_mean"] = _safe_mean(group["test_r2"])
        row["test_r2_std"] = _safe_std(group["test_r2"])
        row["anchor_adjusted_rho_mean"] = _safe_mean(group["test_effective_kstar_proxy_spearman_adjusted_rho"])
        row["anchor_positive_seed_share"] = _safe_positive_share(group["test_effective_kstar_proxy_spearman_adjusted_rho"])
        row["mean_proxy_adjusted_rho_mean"] = _safe_mean(group["test_effective_kstar_proxy_mean_spearman_adjusted_rho"])
        row["mean_proxy_positive_seed_share"] = _safe_positive_share(
            group["test_effective_kstar_proxy_mean_spearman_adjusted_rho"]
        )
        row["kstar_std_mean"] = _safe_mean(group["test_effective_kstar_std"])
        row["entropy_mean"] = _safe_mean(group["test_effective_lag_entropy_mean"])
        row["lag_gate_sensitivity_range_mean"] = _safe_mean(group["test_lag_gate_sensitivity_range"])
        row["proxy_signal_r2_mean"] = _safe_mean(group["test_proxy_signal_r2"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_column", "feature_bundle", "display_name"], na_position="last").reset_index(drop=True)


def _build_synthetic_bootstrap_table(comparison_frame: pd.DataFrame, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    if comparison_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    metrics = {
        "task_loss": "Task loss",
        "effective_kstar_mae": "k* MAE",
        "effective_kstar_spearman_rho": "k* Spearman rho",
    }
    for (scenario, display_name), group in comparison_frame.groupby(["scenario", "display_name"], dropna=False):
        for index, (metric, label) in enumerate(metrics.items()):
            mean_value, ci_low, ci_high, n_used = _bootstrap_mean_ci(group[metric], n_boot=n_boot, seed=seed + index)
            sample = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    "scenario": scenario,
                    "method": display_name,
                    "metric": metric,
                    "metric_label": label,
                    "sample_mean": float(sample.mean()) if not sample.empty else float("nan"),
                    "sample_std": float(sample.std(ddof=1)) if not sample.empty else float("nan"),
                    "bootstrap_mean": mean_value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_seeds": n_used,
                    "n_boot": n_boot,
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario", "metric", "method"], na_position="last").reset_index(drop=True)


def _with_paper_diff_columns(significance_frame: pd.DataFrame) -> pd.DataFrame:
    if significance_frame.empty:
        return significance_frame
    table = significance_frame.copy()
    if "mean_diff" in table.columns:
        table["paper_mean_diff_method_minus_reference"] = -pd.to_numeric(table["mean_diff"], errors="coerce")
    if "median_diff" in table.columns:
        table["paper_median_diff_method_minus_reference"] = -pd.to_numeric(table["median_diff"], errors="coerce")
    table["paper_diff_direction"] = "method_minus_reference"
    return table


def _build_real_r2_wilcoxon_table(economics_r2: pd.DataFrame, energy_r2: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for domain, domain_label, frame in (
        ("economics", "Economics", economics_r2),
        ("energy", "Energy", energy_r2),
    ):
        if frame.empty:
            continue
        domain_frame = frame.copy()
        domain_frame.insert(0, "domain", domain)
        domain_frame.insert(1, "domain_label", domain_label)
        frames.append(domain_frame)
    if not frames:
        return pd.DataFrame()
    table = pd.concat(frames, ignore_index=True, sort=False)
    table = _with_paper_diff_columns(table)
    sort_columns = [column for column in ["domain", "target_column", "feature_bundle", "method"] if column in table.columns]
    return table.sort_values(sort_columns, na_position="last").reset_index(drop=True)


def _read_proxy_shuffle_summary(proxy_root: Path) -> pd.DataFrame:
    comparison_dir = proxy_root / "comparison"
    combined_path = comparison_dir / "proxy_shuffle_summary.csv"
    if combined_path.exists():
        return pd.read_csv(combined_path)
    frames = [
        pd.read_csv(path)
        for path in sorted(comparison_dir.glob("*_proxy_shuffle_summary.csv"))
        if path.is_file()
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _build_proxy_shuffle_compact_table(proxy_root: Path) -> pd.DataFrame:
    summary = _read_proxy_shuffle_summary(proxy_root)
    columns = [
        "domain",
        "domain_label",
        "model",
        "mean_abs_rho",
        "kstar_std_mean",
        "frac_seed_p_lt_05",
        "fisher_p_min",
        "fisher_p_max",
        "test_r2_mean",
        "n_stratifiers",
        "source_path",
    ]
    if summary.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    model_specs = [
        ("AC-GATE", "original", "original_seed_p_lt_05_share"),
        ("Proxy-shuf.", "proxy_shuffled", "proxy_shuffled_seed_p_lt_05_share"),
    ]
    for domain, group in summary.groupby("domain", dropna=False):
        domain_text = str(domain)
        for model, prefix, share_column in model_specs:
            fisher = pd.to_numeric(group.get(f"{prefix}_fisher_p"), errors="coerce").dropna()
            rows.append(
                {
                    "domain": domain_text,
                    "domain_label": DOMAIN_DISPLAY_LABELS.get(domain_text, domain_text.title()),
                    "model": model,
                    "mean_abs_rho": _safe_mean(group[f"{prefix}_abs_rho_mean"]),
                    "kstar_std_mean": _safe_mean(group[f"{prefix}_kstar_std_mean"]),
                    "frac_seed_p_lt_05": _safe_mean(group[share_column]),
                    "fisher_p_min": float(fisher.min()) if not fisher.empty else float("nan"),
                    "fisher_p_max": float(fisher.max()) if not fisher.empty else float("nan"),
                    "test_r2_mean": _safe_mean(group[f"{prefix}_test_r2_mean"]),
                    "n_stratifiers": int(len(group)),
                    "source_path": str(proxy_root / "comparison"),
                }
            )
    table = pd.DataFrame(rows, columns=columns)
    domain_order = {"economics": 0, "energy": 1}
    model_order = {"AC-GATE": 0, "Proxy-shuf.": 1}
    table["_domain_order"] = table["domain"].map(domain_order).fillna(99)
    table["_model_order"] = table["model"].map(model_order).fillna(99)
    return table.sort_values(["_domain_order", "_model_order", "domain", "model"]).drop(
        columns=["_domain_order", "_model_order"]
    ).reset_index(drop=True)


def _build_baseline_compact_table(
    synthetic_summary: pd.DataFrame,
    economics_compact: pd.DataFrame,
    energy_compact: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in synthetic_summary.iterrows():
        rows.append(
            {
                "domain": "synthetic",
                "setting": row.get("scenario"),
                "method": row.get("display_name"),
                "n_seeds": row.get("n_seeds"),
                "primary_metric": "task_loss",
                "primary_mean": row.get("task_loss_mean"),
                "primary_std": row.get("task_loss_std"),
                "secondary_metric": "effective_kstar_mae",
                "secondary_mean": row.get("effective_kstar_mae_mean"),
                "secondary_std": row.get("effective_kstar_mae_std"),
            }
        )

    for domain, frame in (("economics", economics_compact), ("energy", energy_compact)):
        for _, row in frame.iterrows():
            rows.append(
                {
                    "domain": domain,
                    "setting": row.get("target_column"),
                    "method": row.get("display_name"),
                    "n_seeds": row.get("n_seeds"),
                    "primary_metric": "test_r2",
                    "primary_mean": row.get("test_r2_mean"),
                    "primary_std": row.get("test_r2_std"),
                    "secondary_metric": "anchor_adjusted_rho",
                    "secondary_mean": row.get("anchor_adjusted_rho_mean"),
                    "secondary_std": float("nan"),
                }
            )

    return pd.DataFrame(rows).sort_values(["domain", "setting", "method"], na_position="last").reset_index(drop=True)


def _build_realdata_forecast_table(economics_compact: pd.DataFrame, energy_compact: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for domain, frame in (("economics", economics_compact), ("energy", energy_compact)):
        if frame.empty:
            continue
        domain_frame = frame.copy()
        domain_frame.insert(0, "domain", domain)
        frames.append(domain_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _build_stratified_main_table(domain: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    table = frame.copy()
    table.insert(0, "domain", domain)
    return table.sort_values(["method", "abs_rho_mean", "stratifier"], ascending=[True, False, True], na_position="last")


def _build_ablation_degeneracy_table(
    economics_compact: pd.DataFrame,
    energy_compact: pd.DataFrame,
    economics_stratified: pd.DataFrame,
    energy_stratified: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    method_focus = ["CMDL", "No Recon Regularization", "Plain LSTM", "TFT", "GA-Net", "No AC Encoder", "Uniform Lag"]
    for domain, compact, stratified in (
        ("economics", economics_compact, economics_stratified),
        ("energy", energy_compact, energy_stratified),
    ):
        if compact.empty:
            continue
        valid_counts = {}
        if not stratified.empty:
            valid_counts = (
                stratified.assign(valid_flag=pd.to_numeric(stratified["n_seeds_valid"], errors="coerce").fillna(0.0) > 0.0)
                .groupby("method", dropna=False)["valid_flag"]
                .sum()
                .to_dict()
            )
        for _, row in compact.iterrows():
            method = row.get("display_name")
            if method not in method_focus:
                continue
            kstar_std = row.get("kstar_std_mean")
            valid_stratifiers = int(valid_counts.get(method, 0))
            structured_lag_valid = valid_stratifiers > 0
            degenerate_control = bool(
                (pd.notna(kstar_std) and float(kstar_std) == 0.0) or not structured_lag_valid
            )
            rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "kstar_std_mean": kstar_std,
                    "lag_gate_sensitivity_range_mean": row.get("lag_gate_sensitivity_range_mean"),
                    "anchor_adjusted_rho_mean": row.get("anchor_adjusted_rho_mean"),
                    "valid_stratifiers": valid_stratifiers,
                    "structured_lag_valid": structured_lag_valid,
                    "degenerate_control": degenerate_control,
                }
            )
    return pd.DataFrame(rows).sort_values(["domain", "method"], na_position="last").reset_index(drop=True)


def _parse_listish(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _build_proxy_metadata_table(domain: str, comparison_frame: pd.DataFrame) -> pd.DataFrame:
    if comparison_frame.empty:
        return pd.DataFrame()

    identity_columns = ["target_column", "feature_bundle", "anchor_proxy_name", "anchor_expected_sign", "proxy_aggregate_name", "auxiliary_proxy_names", "proxy_expected_signs"]
    source = comparison_frame.drop_duplicates(subset=["target_column", "feature_bundle"])
    rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        target = row.get("target_column")
        feature_bundle = row.get("feature_bundle")
        anchor_name = row.get("anchor_proxy_name")
        anchor_sign = row.get("anchor_expected_sign")
        if anchor_name:
            rows.append(
                {
                    "domain": domain,
                    "target_column": target,
                    "feature_bundle": feature_bundle,
                    "proxy_name": anchor_name,
                    "proxy_role": "anchor",
                    "expected_sign": anchor_sign,
                    "source_hint": "fill_from_domain_writeup",
                    "caveat_template": "fill_from_domain_writeup",
                }
            )
        aggregate_name = row.get("proxy_aggregate_name")
        if aggregate_name:
            rows.append(
                {
                    "domain": domain,
                    "target_column": target,
                    "feature_bundle": feature_bundle,
                    "proxy_name": aggregate_name,
                    "proxy_role": "aggregate",
                    "expected_sign": None,
                    "source_hint": "fill_from_domain_writeup",
                    "caveat_template": "fill_from_domain_writeup",
                }
            )
        auxiliary_names = _parse_listish(row.get("auxiliary_proxy_names"))
        expected_signs = _parse_listish(row.get("proxy_expected_signs"))
        for index, proxy_name in enumerate(auxiliary_names):
            proxy_sign = expected_signs[index] if index < len(expected_signs) else None
            rows.append(
                {
                    "domain": domain,
                    "target_column": target,
                    "feature_bundle": feature_bundle,
                    "proxy_name": proxy_name,
                    "proxy_role": "auxiliary",
                    "expected_sign": proxy_sign,
                    "source_hint": "fill_from_domain_writeup",
                    "caveat_template": "fill_from_domain_writeup",
                }
            )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def _synthetic_layer_status(
    synthetic_summary: pd.DataFrame,
    synthetic_significance_task: pd.DataFrame,
    synthetic_significance_kstar: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    plain_task = synthetic_significance_task.loc[synthetic_significance_task["method"] == "Plain LSTM"].copy()
    plain_kstar = synthetic_significance_kstar.loc[synthetic_significance_kstar["method"] == "Plain LSTM"].copy()
    cmdl_rows = synthetic_summary.loc[synthetic_summary["display_name"] == "CMDL"].copy()
    ablation_rows = synthetic_summary.loc[synthetic_summary["display_name"].isin(["No AC Encoder", "Uniform Lag"])]

    forecast_certified = (
        len(plain_task) >= 2
        and bool((plain_task["reference_better_mean"] & (plain_task["wilcoxon_p"] < 0.05)).all())
    )
    structured_certified = (
        len(plain_kstar) >= 2
        and bool((plain_kstar["reference_better_mean"] & (plain_kstar["wilcoxon_p"] < 0.05)).all())
        and not cmdl_rows.empty
        and not ablation_rows.empty
        and bool((pd.to_numeric(ablation_rows["effective_kstar_spearman_rho_mean"], errors="coerce").fillna(0.0) == 0.0).all())
    )
    directional_certified = (
        not cmdl_rows.empty
        and bool((pd.to_numeric(cmdl_rows["effective_kstar_spearman_rho_mean"], errors="coerce") > 0.8).all())
        and bool((pd.to_numeric(cmdl_rows["kstar_positive_seed_share"], errors="coerce") >= 0.8).all())
    )

    rows.extend(
        [
            {
                "domain": "synthetic",
                "layer": "L0_forecast",
                "verdict": "certified" if forecast_certified else "not_certified",
                "evidence": "CMDL beats Plain LSTM on task loss in both linear and nonlinear scenarios.",
            },
            {
                "domain": "synthetic",
                "layer": "L2_structured_mechanism",
                "verdict": "certified" if structured_certified else "not_certified",
                "evidence": "CMDL beats Plain LSTM on k* MAE and ablation controls collapse lag recovery.",
            },
            {
                "domain": "synthetic",
                "layer": "L3_directional_alignment",
                "verdict": "certified" if directional_certified else "not_certified",
                "evidence": "CMDL keeps positive-seed share high and rank alignment strong across both scenarios.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _realdata_layer_status(
    domain: str,
    compact_summary: pd.DataFrame,
    stratified_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if compact_summary.empty:
        return pd.DataFrame(rows)

    cmdl_row = compact_summary.loc[compact_summary["display_name"] == "CMDL"]
    if cmdl_row.empty:
        return pd.DataFrame(rows)
    cmdl = cmdl_row.iloc[0]
    competitor_r2 = pd.to_numeric(
        compact_summary.loc[compact_summary["display_name"] != "CMDL", "test_r2_mean"], errors="coerce"
    ).dropna()
    forecast_verdict = "certified"
    if not competitor_r2.empty and float(cmdl.get("test_r2_mean")) < float(competitor_r2.max()):
        forecast_verdict = "not_certified"

    l3_verdict = "mixed"
    anchor_rho = pd.to_numeric(pd.Series([cmdl.get("anchor_adjusted_rho_mean")]), errors="coerce").iloc[0]
    anchor_share = pd.to_numeric(pd.Series([cmdl.get("anchor_positive_seed_share")]), errors="coerce").iloc[0]
    if pd.notna(anchor_rho) and pd.notna(anchor_share) and anchor_rho > 0.0 and anchor_share >= (2.0 / 3.0):
        l3_verdict = "certified"
    elif pd.isna(anchor_rho) or pd.isna(anchor_share):
        l3_verdict = "not_certified"

    stratified_cmdl = stratified_summary.loc[stratified_summary["method"] == "CMDL"].copy()
    stratified_controls = stratified_summary.loc[stratified_summary["method"].isin(["No AC Encoder", "Uniform Lag"])]
    l2_certified = (
        not stratified_cmdl.empty
        and bool((pd.to_numeric(stratified_cmdl["n_seeds_valid"], errors="coerce") > 0).all())
        and bool((pd.to_numeric(stratified_cmdl["fisher_combined_p"], errors="coerce") < 0.05).all())
        and (
            stratified_controls.empty
            or bool((pd.to_numeric(stratified_controls["n_seeds_valid"], errors="coerce").fillna(0.0) == 0.0).all())
        )
    )

    rows.extend(
        [
            {
                "domain": domain,
                "layer": "L0_forecast",
                "verdict": forecast_verdict,
                "evidence": f"CMDL mean test_r2={cmdl.get('test_r2_mean')}; best competing mean test_r2={competitor_r2.max() if not competitor_r2.empty else None}.",
            },
            {
                "domain": domain,
                "layer": "L2_structured_mechanism",
                "verdict": "certified" if l2_certified else "not_certified",
                "evidence": "CMDL stratified rows remain valid while No AC Encoder and Uniform Lag are structurally degenerate.",
            },
            {
                "domain": domain,
                "layer": "L3_directional_alignment",
                "verdict": l3_verdict,
                "evidence": f"anchor_adjusted_rho_mean={anchor_rho}, anchor_positive_seed_share={anchor_share}.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _build_verdict_matrix(
    synthetic_summary: pd.DataFrame,
    synthetic_significance_task: pd.DataFrame,
    synthetic_significance_kstar: pd.DataFrame,
    economics_compact: pd.DataFrame,
    economics_stratified: pd.DataFrame,
    energy_compact: pd.DataFrame,
    energy_stratified: pd.DataFrame,
) -> pd.DataFrame:
    frames = [
        _synthetic_layer_status(synthetic_summary, synthetic_significance_task, synthetic_significance_kstar),
        _realdata_layer_status("economics", economics_compact, economics_stratified),
        _realdata_layer_status("energy", energy_compact, energy_stratified),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["domain", "layer", "verdict", "evidence"])
    return pd.concat(frames, ignore_index=True)


def _select_top_stratifier(stratified_summary: pd.DataFrame) -> str:
    cmdl = stratified_summary.loc[stratified_summary["method"] == "CMDL"].copy()
    if cmdl.empty:
        raise ValueError("No CMDL stratified rows were available")
    cmdl["abs_rho_mean"] = pd.to_numeric(cmdl["abs_rho_mean"], errors="coerce")
    return str(cmdl.sort_values("abs_rho_mean", ascending=False, na_position="last").iloc[0]["stratifier"])


def _select_representative_seed(stratified_per_seed: pd.DataFrame, stratifier_name: str) -> int:
    source = stratified_per_seed.loc[
        (stratified_per_seed["method"] == "CMDL")
        & (stratified_per_seed["stratifier"] == stratifier_name)
        & (~pd.to_numeric(stratified_per_seed["degenerate"], errors="coerce").fillna(False).astype(bool))
    ].copy()
    if source.empty:
        raise ValueError(f"No non-degenerate per-seed rows were found for {stratifier_name}")
    source["abs_rho"] = pd.to_numeric(source["spearman_rho"], errors="coerce").abs()
    median_abs = float(source["abs_rho"].median())
    source["distance_to_median"] = (source["abs_rho"] - median_abs).abs()
    source["perm_p_two_sided"] = pd.to_numeric(source["perm_p_two_sided"], errors="coerce")
    chosen = source.sort_values(["distance_to_median", "perm_p_two_sided", "seed"], na_position="last").iloc[0]
    return int(chosen["seed"])


def _case_study_seed_values(stratified_per_seed: pd.DataFrame, stratifier_name: str) -> list[int]:
    source = stratified_per_seed.loc[
        (stratified_per_seed["method"] == "CMDL")
        & (stratified_per_seed["stratifier"] == stratifier_name)
        & (~pd.to_numeric(stratified_per_seed["degenerate"], errors="coerce").fillna(False).astype(bool))
    ].copy()
    if source.empty:
        raise ValueError(f"No non-degenerate per-seed rows were found for {stratifier_name}")
    seeds = pd.to_numeric(source["seed"], errors="coerce").dropna().astype(int).sort_values().unique().tolist()
    if not seeds:
        raise ValueError(f"No seed values were available for {stratifier_name}")
    return [int(seed) for seed in seeds]


def _prediction_path(cmdl_root: Path, seed: int) -> Path:
    matches = sorted(cmdl_root.glob(f"*seed{seed}"))
    if not matches:
        raise FileNotFoundError(f"No CMDL run directory matched seed{seed} under {cmdl_root}")
    target = matches[0] / "predictions.csv"
    if not target.exists():
        raise FileNotFoundError(f"Missing predictions.csv for seed {seed}: {target}")
    return target


def _display_stratifier_label(stratifier_name: str) -> str:
    return STRATIFIER_DISPLAY_LABELS.get(stratifier_name, stratifier_name.replace("_", " ").title())


def _load_seed_entity_kstar(cmdl_root: Path, seed: int) -> pd.DataFrame:
    predictions = pd.read_csv(_prediction_path(cmdl_root, seed))
    required_columns = {"entity_code", "entity_name", "k_star"}
    if not required_columns.issubset(predictions.columns):
        raise ValueError(f"Missing prediction columns for seed {seed}: {sorted(required_columns)}")
    entity_frame = (
        predictions.groupby(["entity_code", "entity_name"], dropna=False)["k_star"]
        .first()
        .rename("k_star")
        .reset_index()
    )
    entity_frame["seed"] = int(seed)
    return entity_frame


def _build_case_study_frame(
    domain: str,
    comparison_frame: pd.DataFrame,
    stratified_summary: pd.DataFrame,
    stratified_per_seed: pd.DataFrame,
    cmdl_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if comparison_frame.empty:
        raise ValueError(f"No comparison frame available for {domain}")

    source_path = Path(str(comparison_frame["source_path"].dropna().iloc[0]))
    top_stratifier = _select_top_stratifier(stratified_summary)
    seed_values = _case_study_seed_values(stratified_per_seed, top_stratifier)

    if domain == "economics":
        stratifier_specs = {spec.name: spec.series for spec in build_economics_stratifiers(source_path)}
        x_label = top_stratifier
    elif domain == "energy":
        stratifier_specs = {spec.name: spec.series for spec in build_energy_stratifiers(source_path)}
        x_label = top_stratifier
    else:
        raise ValueError(f"Unsupported domain for case study: {domain}")

    if top_stratifier not in stratifier_specs:
        raise KeyError(f"Stratifier {top_stratifier} was not built for {domain}")

    all_seed_entities = pd.concat(
        [_load_seed_entity_kstar(cmdl_root=cmdl_root, seed=seed) for seed in seed_values],
        ignore_index=True,
        sort=False,
    )
    entity_frame = (
        all_seed_entities.groupby(["entity_code", "entity_name"], dropna=False)
        .agg(
            k_star_mean=("k_star", "mean"),
            k_star_std=("k_star", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    entity_frame["k_star"] = entity_frame["k_star_mean"]
    stratifier_series = stratifier_specs[top_stratifier].rename("stratifier_value").rename_axis("entity_code").reset_index()
    case_frame = entity_frame.merge(stratifier_series, on="entity_code", how="inner")
    case_frame.insert(0, "domain", domain)
    case_frame.insert(1, "aggregation", f"{len(seed_values)}_seed_mean")
    case_frame.insert(2, "stratifier", top_stratifier)
    x_label = _display_stratifier_label(top_stratifier)
    domain_label = DOMAIN_DISPLAY_LABELS.get(domain, domain.title())
    metadata = {
        "domain": domain,
        "aggregation": f"{len(seed_values)}-seed mean",
        "n_seeds": int(len(seed_values)),
        "stratifier": top_stratifier,
        "x_label": x_label,
        "y_label": f"Mean learned effective lag across {len(seed_values)} seeds",
        "title": f"{domain_label}: {x_label}",
        "point_color": CASE_STUDY_POINT_COLORS.get(domain, "#2E6F95"),
    }
    return case_frame.sort_values("stratifier_value", na_position="last").reset_index(drop=True), metadata


def _write_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Path]:
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_paths: dict[str, Path] = {}
    for name, frame in tables.items():
        target = output_dir / name
        frame.to_csv(target, index=False)
        table_paths[name] = target
    return table_paths


def _generate_figures(
    output_dir: Path,
    synthetic_frame: pd.DataFrame,
    economics_frame: pd.DataFrame,
    energy_frame: pd.DataFrame,
    economics_stratified_seed: pd.DataFrame,
    energy_stratified_seed: pd.DataFrame,
    economics_case_study: pd.DataFrame,
    economics_case_meta: dict[str, Any],
    energy_case_study: pd.DataFrame,
    energy_case_meta: dict[str, Any],
) -> dict[str, Path]:
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}

    figure_paths["workflow_overview.png"] = output_dir / "workflow_overview.png"
    plt.close(plot_workflow_overview(figure_paths["workflow_overview.png"]))

    synthetic_order = [
        method
        for method in ["CMDL", "No Recon Regularization", "Plain LSTM", "TFT", "GA-Net", "No AC Encoder", "Uniform Lag"]
        if method in synthetic_frame["display_name"].dropna().unique().tolist()
    ]
    for scenario_name in ("linear", "nonlinear"):
        scenario_frame = synthetic_frame.loc[synthetic_frame["scenario"] == scenario_name].copy()
        if scenario_frame.empty:
            continue
        file_name = f"synthetic_kstar_mae_seed_distribution_{scenario_name}.png"
        figure_paths[file_name] = output_dir / file_name
        plt.close(plot_seed_distribution(
            scenario_frame,
            value_col="effective_kstar_mae",
            category_col="display_name",
            order=synthetic_order,
            title=f"Synthetic Seed Distribution: k* MAE ({scenario_name})",
            ylabel="k* MAE",
            save_path=figure_paths[file_name],
            figsize=(6.2, 4.8),
        ))

    realdata_order = [
        method
        for method in ["CMDL", "No Recon Regularization", "Plain LSTM", "TFT", "GA-Net", "Grouped ARDL", "No AC Encoder", "Uniform Lag"]
        if method in pd.concat([economics_frame["display_name"], energy_frame["display_name"]]).dropna().unique().tolist()
    ]
    figure_paths["economics_forecast_seed_distribution.png"] = output_dir / "economics_forecast_seed_distribution.png"
    plt.close(plot_seed_distribution(
        economics_frame,
        value_col="test_r2",
        category_col="display_name",
        order=realdata_order,
        title="Economics Seed Distribution: test R2",
        ylabel="test R2",
        save_path=figure_paths["economics_forecast_seed_distribution.png"],
    ))

    figure_paths["energy_forecast_seed_distribution.png"] = output_dir / "energy_forecast_seed_distribution.png"
    plt.close(plot_seed_distribution(
        energy_frame,
        value_col="test_r2",
        category_col="display_name",
        order=realdata_order,
        title="Energy Seed Distribution: test R2",
        ylabel="test R2",
        save_path=figure_paths["energy_forecast_seed_distribution.png"],
    ))

    figure_paths["economics_structured_mechanism_seed_distribution.png"] = (
        output_dir / "economics_structured_mechanism_seed_distribution.png"
    )
    plt.close(plot_stratified_seed_distribution(
        economics_stratified_seed.loc[economics_stratified_seed["method"] == "CMDL"],
        title="Economics Seed Distribution: |rho| by Stratifier",
        ylabel="|Spearman rho|",
        save_path=figure_paths["economics_structured_mechanism_seed_distribution.png"],
    ))

    figure_paths["energy_structured_mechanism_seed_distribution.png"] = (
        output_dir / "energy_structured_mechanism_seed_distribution.png"
    )
    plt.close(plot_stratified_seed_distribution(
        energy_stratified_seed.loc[energy_stratified_seed["method"] == "CMDL"],
        title="Energy Seed Distribution: |rho| by Stratifier",
        ylabel="|Spearman rho|",
        save_path=figure_paths["energy_structured_mechanism_seed_distribution.png"],
    ))

    figure_paths["economics_case_study.png"] = output_dir / "economics_case_study.png"
    plt.close(plot_case_study_scatter(
        economics_case_study,
        x_col="stratifier_value",
        y_col="k_star",
        label_col=None,
        title=economics_case_meta["title"],
        xlabel=economics_case_meta["x_label"],
        ylabel=economics_case_meta["y_label"],
        save_path=figure_paths["economics_case_study.png"],
        annotate_top_n=0,
        point_color=economics_case_meta["point_color"],
    ))

    figure_paths["energy_case_study.png"] = output_dir / "energy_case_study.png"
    plt.close(plot_case_study_scatter(
        energy_case_study,
        x_col="stratifier_value",
        y_col="k_star",
        label_col=None,
        title=energy_case_meta["title"],
        xlabel=energy_case_meta["x_label"],
        ylabel=energy_case_meta["y_label"],
        save_path=figure_paths["energy_case_study.png"],
        annotate_top_n=0,
        point_color=energy_case_meta["point_color"],
    ))

    figure_paths["realdata_case_study_panel.png"] = output_dir / "realdata_case_study_panel.png"
    combined_fig, axes = plt.subplots(2, 1, figsize=(7.3, 9.1), sharey=False)
    plot_case_study_scatter(
        economics_case_study,
        x_col="stratifier_value",
        y_col="k_star",
        label_col=None,
        title=economics_case_meta["title"],
        xlabel=economics_case_meta["x_label"],
        ylabel=economics_case_meta["y_label"],
        annotate_top_n=0,
        point_color=economics_case_meta["point_color"],
        axis=axes[0],
    )
    plot_case_study_scatter(
        energy_case_study,
        x_col="stratifier_value",
        y_col="k_star",
        label_col=None,
        title=energy_case_meta["title"],
        xlabel=energy_case_meta["x_label"],
        ylabel=energy_case_meta["y_label"],
        annotate_top_n=0,
        point_color=energy_case_meta["point_color"],
        axis=axes[1],
    )
    combined_fig.tight_layout(h_pad=1.4)
    combined_fig.savefig(figure_paths["realdata_case_study_panel.png"], dpi=200, bbox_inches="tight")
    plt.close(combined_fig)

    return figure_paths


def generate_paper_assets(
    paths: PaperPaths,
    output_root: Path | str | None = None,
    include_figures: bool = True,
    n_boot: int = 5000,
    bootstrap_seed: int = 0,
    proxy_root: Path | str | None = None,
) -> PaperArtifactBundle:
    """Build paper-facing tables and figures without touching model code paths."""

    output_root = Path(output_root) if output_root is not None else paths.workspace_root / "outputs" / "paper_assets" / paths.plan_name
    proxy_root = Path(proxy_root) if proxy_root is not None else paths.proxy_shuffle_root
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"

    synthetic_frame = build_synthetic_comparison(
        cmdl_root=paths.synthetic_root / "cmdl",
        baseline_root=paths.synthetic_root / "plain_lstm",
        tft_root=paths.synthetic_root / "tft",
        ganet_root=paths.synthetic_root / "ganet",
        ablation_root=paths.synthetic_root / "ablation",
    )
    economics_frame = build_economics_comparison(
        cmdl_root=paths.economics_root / "cmdl",
        baseline_root=paths.economics_root / "plain_lstm",
        tft_root=paths.economics_root / "tft",
        ganet_root=paths.economics_root / "ganet",
        ablation_root=paths.economics_root / "ablation",
        grouped_ardl_root=paths.economics_root / "grouped_ardl",
    )
    energy_frame = build_energy_comparison(
        cmdl_root=paths.energy_root / "cmdl",
        baseline_root=paths.energy_root / "plain_lstm",
        tft_root=paths.energy_root / "tft",
        ganet_root=paths.energy_root / "ganet",
        ablation_root=paths.energy_root / "ablation",
        grouped_ardl_root=paths.energy_root / "grouped_ardl",
    )

    synthetic_summary = _aggregate_synthetic_summary(synthetic_frame)
    economics_compact = _aggregate_realdata_compact(economics_frame)
    energy_compact = _aggregate_realdata_compact(energy_frame)

    synthetic_significance = build_synthetic_significance_tables(synthetic_frame)
    synthetic_significance_task = synthetic_significance.get("synthetic_significance_task_loss.csv", pd.DataFrame())
    synthetic_significance_kstar = synthetic_significance.get("synthetic_significance_kstar_mae.csv", pd.DataFrame())
    synthetic_significance_tables = {
        name: _with_paper_diff_columns(frame)
        for name, frame in synthetic_significance.items()
    }

    economics_significance = build_economics_significance_tables(economics_frame, split="test")
    energy_significance = build_energy_significance_tables(energy_frame, split="test")
    economics_significance_r2 = _with_paper_diff_columns(
        economics_significance.get("economics_significance_test_r2.csv", pd.DataFrame())
    )
    energy_significance_r2 = _with_paper_diff_columns(
        energy_significance.get("energy_significance_test_r2.csv", pd.DataFrame())
    )
    real_r2_wilcoxon = _build_real_r2_wilcoxon_table(economics_significance_r2, energy_significance_r2)
    proxy_shuffle_compact = _build_proxy_shuffle_compact_table(proxy_root)

    economics_stratified = _read_csv(paths.economics_root / "comparison" / "economics_stratified_kstar_aggregated.csv")
    economics_stratified_seed = _read_csv(paths.economics_root / "comparison" / "economics_stratified_kstar_per_seed.csv")
    energy_stratified = _read_csv(paths.energy_root / "comparison" / "energy_stratified_kstar_aggregated.csv")
    energy_stratified_seed = _read_csv(paths.energy_root / "comparison" / "energy_stratified_kstar_per_seed.csv")

    economics_result_log = build_economics_result_log(economics_frame)
    energy_result_log = build_energy_result_log(energy_frame)

    proxy_metadata = pd.concat(
        [
            _build_proxy_metadata_table("economics", economics_frame),
            _build_proxy_metadata_table("energy", energy_frame),
        ],
        ignore_index=True,
        sort=False,
    )

    ablation_degeneracy = _build_ablation_degeneracy_table(
        economics_compact,
        energy_compact,
        economics_stratified,
        energy_stratified,
    )
    verdict_matrix = build_verdict_matrix(
        synthetic_summary=synthetic_summary,
        synthetic_significance_task=synthetic_significance_task,
        synthetic_significance_kstar=synthetic_significance_kstar,
        economics_compact=economics_compact,
        economics_stratified=economics_stratified,
        energy_compact=energy_compact,
        energy_stratified=energy_stratified,
        ablation_degeneracy=ablation_degeneracy,
    )

    economics_case_study, economics_case_meta = _build_case_study_frame(
        domain="economics",
        comparison_frame=economics_frame,
        stratified_summary=economics_stratified,
        stratified_per_seed=economics_stratified_seed,
        cmdl_root=paths.economics_root / "cmdl",
    )
    energy_case_study, energy_case_meta = _build_case_study_frame(
        domain="energy",
        comparison_frame=energy_frame,
        stratified_summary=energy_stratified,
        stratified_per_seed=energy_stratified_seed,
        cmdl_root=paths.energy_root / "cmdl",
    )

    tables = {
        "synthetic_main_table.csv": synthetic_summary,
        **synthetic_significance_tables,
        "synthetic_bootstrap_ci.csv": _build_synthetic_bootstrap_table(
            synthetic_frame,
            n_boot=n_boot,
            seed=bootstrap_seed,
        ),
        "realdata_forecast_table.csv": _build_realdata_forecast_table(economics_compact, energy_compact),
        "economics_significance_test_r2.csv": economics_significance_r2,
        "energy_significance_test_r2.csv": energy_significance_r2,
        "real_r2_wilcoxon.csv": real_r2_wilcoxon,
        "baseline_compact_table.csv": _build_baseline_compact_table(synthetic_summary, economics_compact, energy_compact),
        "economics_stratified_main_table.csv": _build_stratified_main_table("economics", economics_stratified),
        "energy_stratified_main_table.csv": _build_stratified_main_table("energy", energy_stratified),
        "ablation_degeneracy_table.csv": ablation_degeneracy,
        "verdict_matrix.csv": verdict_matrix,
        "proxy_shuffle_compact_table.csv": proxy_shuffle_compact,
        "proxy_metadata_table.csv": proxy_metadata,
        "economics_case_study_source.csv": economics_case_study,
        "energy_case_study_source.csv": energy_case_study,
        "economics_mechanism_result_log.csv": economics_result_log,
        "energy_mechanism_result_log.csv": energy_result_log,
    }

    table_paths = _write_tables(tables, tables_dir)
    figure_paths: dict[str, Path] = {}
    if include_figures:
        figure_paths = _generate_figures(
            figures_dir,
            synthetic_frame,
            economics_frame,
            energy_frame,
            economics_stratified_seed,
            energy_stratified_seed,
            economics_case_study,
            economics_case_meta,
            energy_case_study,
            energy_case_meta,
        )

    manifest = {
        "plan_name": paths.plan_name,
        "output_root": str(output_root),
        "tables": {name: str(path) for name, path in table_paths.items()},
        "figures": {name: str(path) for name, path in figure_paths.items()},
        "proxy_shuffle_root": str(proxy_root),
        "case_studies": {
            "economics": economics_case_meta,
            "energy": energy_case_meta,
        },
    }
    manifest_path = output_root / "paper_artifacts_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    return PaperArtifactBundle(tables=tables, figure_paths=figure_paths, manifest_path=manifest_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-facing reporting artifacts from existing CMDL outputs.")
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan-name", type=str, default="complete_20seed_20260426")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    bundle = generate_paper_assets(
        paths=PaperPaths(workspace_root=args.workspace_root.resolve(), plan_name=args.plan_name),
        output_root=args.output_root,
        include_figures=not args.skip_figures,
        n_boot=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(f"Wrote paper artifacts manifest: {bundle.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PaperArtifactBundle", "PaperPaths", "generate_paper_assets", "main"]