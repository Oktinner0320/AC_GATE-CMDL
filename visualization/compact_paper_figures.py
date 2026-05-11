"""Build compact paper figures from locked CMDL experiment outputs.

This script is intentionally separate from the experiment runners. It only reads
CSV artifacts under outputs/notebook_* and outputs/paper_assets, then writes new
figures under outputs/paper_assets/<plan>/compact_figures. Use
--copy-to-paper-img to also copy the generated files into paper_draft/img.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle


DEFAULT_PLAN = "complete_20seed_20260426"
DEFAULT_TFT_PLAN = "complete_20seed"

METHOD_ORDER = [
    "CMDL",
    "No Recon Regularization",
    "Plain LSTM",
    "TFT",
    "GA-Net",
    "Grouped ARDL",
    "No AC Encoder",
    "Uniform Lag",
]

SYNTHETIC_METHOD_ORDER = [
    "CMDL",
    "No Recon Regularization",
    "Plain LSTM",
    "TFT",
    "GA-Net",
    "No AC Encoder",
    "Uniform Lag",
]

METHOD_LABEL = {
    "CMDL": "AC-GATE",
    "No Recon Regularization": "No-Recon",
    "Plain LSTM": "Plain LSTM",
    "TFT": "TFT",
    "GA-Net": "GA-Net",
    "Grouped ARDL": "ARDL",
    "No AC Encoder": "No-AC",
    "Uniform Lag": "Uniform-Lag",
}

METHOD_ALIASES = {
    "No Recon Reg": "No Recon Regularization",
}

STRATIFIER_LABEL = {
    "hc_mean_train": "Human capital",
    "log_gdp_per_worker_train": "GDP per worker",
    "log_capital_per_worker_train": "Capital per worker",
    "rule_of_law_train": "Rule of law",
    "government_effectiveness_train": "Gov. effectiveness",
    "log_gdp_per_capita_train": "GDP per capita",
}

DOMAIN_LABEL = {
    "synthetic": "Synthetic",
    "economics": "Economics",
    "energy": "Energy",
}

METHOD_COLOR = {
    "CMDL": "#1F77B4",
    "No Recon Regularization": "#2CA02C",
    "Plain LSTM": "#7F7F7F",
    "TFT": "#17BECF",
    "GA-Net": "#8C564B",
    "Grouped ARDL": "#9467BD",
    "No AC Encoder": "#D62728",
    "Uniform Lag": "#FF7F0E",
}

SCENARIO_COLOR = {
    "linear": "#1F77B4",
    "nonlinear": "#FF7F0E",
}

DOMAIN_MARKER = {
    "economics": "o",
    "energy": "s",
}

STATUS_STYLE = {
    "certified": {"label": "yes", "color": "#2E7D32"},
    "not_certified": {"label": "n/c", "color": "#F9A825"},
    "ruled_out": {"label": "no", "color": "#C62828"},
    "not_claimed": {"label": "n/a", "color": "#9E9E9E"},
}


@dataclass(frozen=True)
class FigureInputs:
    synthetic: pd.DataFrame
    economics: pd.DataFrame
    energy: pd.DataFrame
    economics_stratified: pd.DataFrame
    energy_stratified: pd.DataFrame
    verdict: pd.DataFrame
    synthetic_summary: pd.DataFrame
    realdata_forecast: pd.DataFrame
    ablation_degeneracy: pd.DataFrame


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required CSV: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _normalize_method_names(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("display_name", "method"):
        if column in result.columns:
            result[column] = result[column].replace(METHOD_ALIASES)
    return result


def _load_tft_synthetic(root: Path, plan: str) -> pd.DataFrame:
    """Build a synthetic comparison frame for TFT runs.

    Mirrors the columns the synthetic figure consumes (display_name, scenario, seed,
    effective_kstar_mae, effective_kstar_spearman_rho). For TFT — which has no learned
    omega — the post-hoc lag-occlusion k* is mapped onto the effective_kstar columns,
    matching how Plain LSTM is recorded in the locked synthetic_comparison.csv.
    """

    import json

    base = root / "outputs" / "notebook_synthetic" / plan / "tft"
    rows: list[dict[str, object]] = []
    if not base.exists():
        return pd.DataFrame(rows)
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open() as handle:
            data = json.load(handle)
        metrics = data.get("metrics", {})
        rows.append(
            {
                "family": "tft",
                "display_name": "TFT",
                "experiment": data.get("experiment", run_dir.name),
                "scenario": data.get("scenario"),
                "seed": data.get("config", {}).get("seed"),
                "model": data.get("model"),
                "effective_kstar_mae": metrics.get("posthoc_kstar_mae"),
                "effective_kstar_spearman_rho": metrics.get("posthoc_kstar_spearman_rho"),
                "posthoc_kstar_mae": metrics.get("posthoc_kstar_mae"),
                "posthoc_kstar_spearman_rho": metrics.get("posthoc_kstar_spearman_rho"),
                "task_r2": metrics.get("task_r2"),
            }
        )
    return pd.DataFrame(rows)


def _load_tft_realdata(root: Path, plan: str, domain: str) -> pd.DataFrame:
    """Build a real-data comparison frame for TFT runs (test_r2 + display_name)."""

    import json

    base = root / "outputs" / f"notebook_{domain}" / plan / "tft"
    rows: list[dict[str, object]] = []
    if not base.exists():
        return pd.DataFrame(rows)
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open() as handle:
            data = json.load(handle)
        test_metrics = data.get("metrics", {}).get("test", {}) or {}
        cfg = data.get("config", {}) or {}
        rows.append(
            {
                "family": "tft",
                "display_name": "TFT",
                "experiment": data.get("experiment", run_dir.name),
                "domain": domain,
                "seed": cfg.get("seed"),
                "model": data.get("model"),
                "test_r2": test_metrics.get("r2"),
                "test_mae": test_metrics.get("mae"),
                "test_mse": test_metrics.get("mse"),
                "posthoc_kstar_mae": test_metrics.get("posthoc_kstar_mae"),
                "posthoc_kstar_spearman_rho": test_metrics.get("posthoc_kstar_spearman_rho"),
            }
        )
    return pd.DataFrame(rows)


def _load_tft_stratified(root: Path, plan: str, domain: str) -> pd.DataFrame:
    """Compute per-seed stratified k* analysis for TFT runs.

    Reuses the stratifier specs and evaluation logic from
    :mod:`evaluation.stratified_kstar` so the resulting rows match the schema
    of the locked ``*_stratified_kstar_per_seed.csv`` files.
    """

    import sys

    repo_root = root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from evaluation.stratified_kstar import (
        build_economics_stratifiers,
        build_energy_stratifiers,
        collect_seed_dirs,
        evaluate_method,
    )

    base = root / "outputs" / f"notebook_{domain}" / plan / "tft"
    if not base.exists():
        return pd.DataFrame()
    if domain == "economics":
        raw_csv = root / "data" / "economics" / "processed" / "economics_cleaned_long_v2.csv"
        stratifiers = build_economics_stratifiers(raw_csv)
        prefix = "economics_tft_"
    else:
        raw_csv = root / "data" / "energy" / "raw" / "energy_wgi_merged.csv"
        stratifiers = build_energy_stratifiers(raw_csv)
        prefix = "energy_tft_"
    seed_dirs = collect_seed_dirs(base, prefix)
    if not seed_dirs:
        return pd.DataFrame()
    return evaluate_method("TFT", seed_dirs, stratifiers, n_perm=2000)


def _load_ganet_synthetic(root: Path, plan: str) -> pd.DataFrame:
    """Build a synthetic comparison frame for GA-Net runs.

    Mirrors :func:`_load_tft_synthetic`. The GA-Net baseline shares the same
    post-hoc lag-occlusion diagnostic as the LSTM/TFT baselines, so its
    ``posthoc_kstar_*`` metrics map onto the ``effective_kstar_*`` columns the
    synthetic figure expects.
    """

    import json

    base = root / "outputs" / "notebook_synthetic" / plan / "ganet"
    rows: list[dict[str, object]] = []
    if not base.exists():
        return pd.DataFrame(rows)
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open() as handle:
            data = json.load(handle)
        metrics = data.get("metrics", {})
        rows.append(
            {
                "family": "ganet",
                "display_name": "GA-Net",
                "experiment": data.get("experiment", run_dir.name),
                "scenario": data.get("scenario"),
                "seed": data.get("config", {}).get("seed"),
                "model": data.get("model"),
                "effective_kstar_mae": metrics.get("posthoc_kstar_mae"),
                "effective_kstar_spearman_rho": metrics.get("posthoc_kstar_spearman_rho"),
                "posthoc_kstar_mae": metrics.get("posthoc_kstar_mae"),
                "posthoc_kstar_spearman_rho": metrics.get("posthoc_kstar_spearman_rho"),
                "task_r2": metrics.get("task_r2"),
            }
        )
    return pd.DataFrame(rows)


def _load_ganet_realdata(root: Path, plan: str, domain: str) -> pd.DataFrame:
    """Build a real-data comparison frame for GA-Net runs (test_r2 + display_name)."""

    import json

    base = root / "outputs" / f"notebook_{domain}" / plan / "ganet"
    rows: list[dict[str, object]] = []
    if not base.exists():
        return pd.DataFrame(rows)
    for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open() as handle:
            data = json.load(handle)
        test_metrics = data.get("metrics", {}).get("test", {}) or {}
        cfg = data.get("config", {}) or {}
        rows.append(
            {
                "family": "ganet",
                "display_name": "GA-Net",
                "experiment": data.get("experiment", run_dir.name),
                "domain": domain,
                "seed": cfg.get("seed"),
                "model": data.get("model"),
                "test_r2": test_metrics.get("r2"),
                "test_mae": test_metrics.get("mae"),
                "test_mse": test_metrics.get("mse"),
                "posthoc_kstar_mae": test_metrics.get("posthoc_kstar_mae"),
                "posthoc_kstar_spearman_rho": test_metrics.get("posthoc_kstar_spearman_rho"),
            }
        )
    return pd.DataFrame(rows)


def _load_ganet_stratified(root: Path, plan: str, domain: str) -> pd.DataFrame:
    """Compute per-seed stratified k* analysis for GA-Net runs.

    Reuses the stratifier specs and evaluation logic from
    :mod:`evaluation.stratified_kstar` so the resulting rows match the schema
    of the locked ``*_stratified_kstar_per_seed.csv`` files.
    """

    import sys

    repo_root = root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from evaluation.stratified_kstar import (
        build_economics_stratifiers,
        build_energy_stratifiers,
        collect_seed_dirs,
        evaluate_method,
    )

    base = root / "outputs" / f"notebook_{domain}" / plan / "ganet"
    if not base.exists():
        return pd.DataFrame()
    if domain == "economics":
        raw_csv = root / "data" / "economics" / "processed" / "economics_cleaned_long_v2.csv"
        stratifiers = build_economics_stratifiers(raw_csv)
        prefix = "economics_ganet_"
    else:
        raw_csv = root / "data" / "energy" / "raw" / "energy_wgi_merged.csv"
        stratifiers = build_energy_stratifiers(raw_csv)
        prefix = "energy_ganet_"
    seed_dirs = collect_seed_dirs(base, prefix)
    if not seed_dirs:
        return pd.DataFrame()
    return evaluate_method("GA-Net", seed_dirs, stratifiers, n_perm=2000)


def _append_unique(base: pd.DataFrame, extra: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if extra is None or extra.empty:
        return base
    if base.empty:
        return extra.copy()
    aligned = extra.reindex(columns=base.columns.union(extra.columns))
    combined = pd.concat([base.reindex(columns=aligned.columns), aligned], ignore_index=True)
    if all(col in combined.columns for col in key_cols):
        combined = combined.drop_duplicates(subset=key_cols, keep="first")
    return combined.reset_index(drop=True)


def _load_inputs(root: Path, plan: str, tft_plan: str | None = None) -> FigureInputs:
    paper_tables = root / "outputs" / "paper_assets" / plan / "tables"
    synthetic_root = root / "outputs" / "notebook_synthetic" / plan / "comparison"
    economics_root = root / "outputs" / "notebook_economics" / plan / "comparison"
    energy_root = root / "outputs" / "notebook_energy" / plan / "comparison"

    synthetic = _normalize_method_names(_read_csv(synthetic_root / "synthetic_comparison.csv"))
    economics = _normalize_method_names(_read_csv(economics_root / "economics_comparison.csv"))
    energy = _normalize_method_names(_read_csv(energy_root / "energy_comparison.csv"))
    economics_strat = _normalize_method_names(_read_csv(economics_root / "economics_stratified_kstar_per_seed.csv"))
    energy_strat = _normalize_method_names(_read_csv(energy_root / "energy_stratified_kstar_per_seed.csv"))

    if tft_plan:
        tft_synth = _load_tft_synthetic(root, tft_plan)
        tft_econ = _load_tft_realdata(root, tft_plan, "economics")
        tft_energy = _load_tft_realdata(root, tft_plan, "energy")
        tft_econ_strat = _load_tft_stratified(root, tft_plan, "economics")
        tft_energy_strat = _load_tft_stratified(root, tft_plan, "energy")
        synthetic = _append_unique(synthetic, tft_synth, ["display_name", "scenario", "seed"])
        economics = _append_unique(economics, tft_econ, ["display_name", "seed"])
        energy = _append_unique(energy, tft_energy, ["display_name", "seed"])
        economics_strat = _append_unique(economics_strat, tft_econ_strat, ["method", "seed", "stratifier"])
        energy_strat = _append_unique(energy_strat, tft_energy_strat, ["method", "seed", "stratifier"])

        ganet_synth = _load_ganet_synthetic(root, tft_plan)
        ganet_econ = _load_ganet_realdata(root, tft_plan, "economics")
        ganet_energy = _load_ganet_realdata(root, tft_plan, "energy")
        ganet_econ_strat = _load_ganet_stratified(root, tft_plan, "economics")
        ganet_energy_strat = _load_ganet_stratified(root, tft_plan, "energy")
        synthetic = _append_unique(synthetic, ganet_synth, ["display_name", "scenario", "seed"])
        economics = _append_unique(economics, ganet_econ, ["display_name", "seed"])
        energy = _append_unique(energy, ganet_energy, ["display_name", "seed"])
        economics_strat = _append_unique(economics_strat, ganet_econ_strat, ["method", "seed", "stratifier"])
        energy_strat = _append_unique(energy_strat, ganet_energy_strat, ["method", "seed", "stratifier"])

    return FigureInputs(
        synthetic=synthetic,
        economics=economics,
        energy=energy,
        economics_stratified=economics_strat,
        energy_stratified=energy_strat,
        verdict=_normalize_method_names(_read_csv(paper_tables / "verdict_matrix.csv")),
        synthetic_summary=_normalize_method_names(_read_csv(paper_tables / "synthetic_main_table.csv")),
        realdata_forecast=_normalize_method_names(_read_csv(paper_tables / "realdata_forecast_table.csv")),
        ablation_degeneracy=_normalize_method_names(_read_csv(paper_tables / "ablation_degeneracy_table.csv")),
    )


def _clean_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def _bootstrap_mean_ci(values: pd.Series | np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float, int]:
    if isinstance(values, pd.Series):
        numeric = _clean_numeric(values)
    else:
        numeric = np.asarray(values, dtype=float)
        numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return np.nan, np.nan, np.nan, 0
    if numeric.size == 1:
        value = float(numeric[0])
        return value, value, value, 1
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, numeric.size, size=(n_boot, numeric.size))
    means = numeric[sample_index].mean(axis=1)
    return float(numeric.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), int(numeric.size)


def _synthetic_summary(frame: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    required = {"scenario", "display_name", "effective_kstar_mae", "effective_kstar_spearman_rho"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Synthetic comparison is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    metrics = ["effective_kstar_mae", "effective_kstar_spearman_rho"]
    for (scenario, method), group in frame.groupby(["scenario", "display_name"], dropna=False):
        if method not in SYNTHETIC_METHOD_ORDER:
            continue
        for metric_index, metric in enumerate(metrics):
            mean, ci_low, ci_high, n_used = _bootstrap_mean_ci(
                group[metric], n_boot=n_boot, seed=101 + metric_index + 17 * len(rows)
            )
            rows.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_seeds": n_used,
                }
            )
    return pd.DataFrame(rows)


def _forecast_summary(frame: pd.DataFrame, domain: str, n_boot: int) -> pd.DataFrame:
    required = {"display_name", "test_r2"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{domain} comparison is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for method, group in frame.groupby("display_name", dropna=False):
        if method not in METHOD_ORDER:
            continue
        mean, ci_low, ci_high, n_used = _bootstrap_mean_ci(group["test_r2"], n_boot=n_boot, seed=211 + len(rows))
        rows.append(
            {
                "domain": domain,
                "method": method,
                "test_r2_mean": mean,
                "test_r2_ci_low": ci_low,
                "test_r2_ci_high": ci_high,
                "n_seeds": n_used,
            }
        )
    return pd.DataFrame(rows)


def _mechanism_summary(frame: pd.DataFrame, domain: str, n_boot: int) -> pd.DataFrame:
    required = {"method", "seed", "stratifier", "spearman_rho", "degenerate"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{domain} stratified per-seed table is missing columns: {sorted(missing)}")

    working = frame.copy()
    working["spearman_rho"] = pd.to_numeric(working["spearman_rho"], errors="coerce")
    if working["degenerate"].dtype != bool:
        working["degenerate"] = working["degenerate"].astype(str).str.lower().eq("true")
    valid = working.loc[~working["degenerate"] & working["spearman_rho"].notna()].copy()
    valid["abs_rho"] = valid["spearman_rho"].abs()

    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        group = valid.loc[valid["method"] == method]
        if group.empty:
            rows.append(
                {
                    "domain": domain,
                    "method": method,
                    "max_abs_rho_mean": np.nan,
                    "max_abs_rho_ci_low": np.nan,
                    "max_abs_rho_ci_high": np.nan,
                    "n_seeds": 0,
                    "valid_l2": False,
                }
            )
            continue
        per_seed = group.groupby("seed", dropna=False)["abs_rho"].max()
        mean, ci_low, ci_high, n_used = _bootstrap_mean_ci(per_seed, n_boot=n_boot, seed=307 + len(rows))
        rows.append(
            {
                "domain": domain,
                "method": method,
                "max_abs_rho_mean": mean,
                "max_abs_rho_ci_low": ci_low,
                "max_abs_rho_ci_high": ci_high,
                "n_seeds": n_used,
                "valid_l2": True,
            }
        )
    return pd.DataFrame(rows)


def _decoupling_summary(inputs: FigureInputs, n_boot: int) -> pd.DataFrame:
    forecast = pd.concat(
        [
            _forecast_summary(inputs.economics, "economics", n_boot),
            _forecast_summary(inputs.energy, "energy", n_boot),
        ],
        ignore_index=True,
    )
    mechanism = pd.concat(
        [
            _mechanism_summary(inputs.economics_stratified, "economics", n_boot),
            _mechanism_summary(inputs.energy_stratified, "energy", n_boot),
        ],
        ignore_index=True,
    )
    return forecast.merge(mechanism, on=["domain", "method"], how="left", suffixes=("_forecast", "_mechanism"))


def _stratifier_summary(inputs: FigureInputs, n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, frame in (
        ("economics", inputs.economics_stratified),
        ("energy", inputs.energy_stratified),
    ):
        working = frame.loc[frame["method"] == "CMDL"].copy()
        working["spearman_rho"] = pd.to_numeric(working["spearman_rho"], errors="coerce")
        working["perm_p_two_sided"] = pd.to_numeric(working["perm_p_two_sided"], errors="coerce")
        if working["degenerate"].dtype != bool:
            working["degenerate"] = working["degenerate"].astype(str).str.lower().eq("true")
        working = working.loc[~working["degenerate"] & working["spearman_rho"].notna()].copy()
        working["abs_rho"] = working["spearman_rho"].abs()
        for stratifier, group in working.groupby("stratifier", dropna=False):
            mean, ci_low, ci_high, n_used = _bootstrap_mean_ci(group["abs_rho"], n_boot=n_boot, seed=401 + len(rows))
            p_values = _clean_numeric(group["perm_p_two_sided"])
            pass_rate = float(np.mean(p_values < 0.05)) if p_values.size else np.nan
            rows.append(
                {
                    "domain": domain,
                    "stratifier": stratifier,
                    "abs_rho_mean": mean,
                    "abs_rho_ci_low": ci_low,
                    "abs_rho_ci_high": ci_high,
                    "pass_rate": pass_rate,
                    "n_seeds": n_used,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    domain_rank = {"economics": 0, "energy": 1}
    result["domain_rank"] = result["domain"].map(domain_rank).fillna(99)
    return result.sort_values(["domain_rank", "abs_rho_mean"], ascending=[True, False]).reset_index(drop=True)


def _derive_l1_status(inputs: FigureInputs) -> dict[str, str]:
    status: dict[str, str] = {}
    synthetic = inputs.synthetic_summary.copy()
    synthetic["kstar_positive_seed_share"] = pd.to_numeric(
        synthetic.get("kstar_positive_seed_share"), errors="coerce"
    )
    cmdl_share = synthetic.loc[synthetic["display_name"] == "CMDL", "kstar_positive_seed_share"].dropna()
    collapsed = synthetic.loc[
        synthetic["display_name"].isin(["No AC Encoder", "Uniform Lag"]), "kstar_positive_seed_share"
    ].dropna()
    status["synthetic"] = "certified" if not cmdl_share.empty and cmdl_share.min() > 0 and not collapsed.empty and collapsed.max() == 0 else "not_certified"

    degeneracy = inputs.ablation_degeneracy.copy()
    degeneracy["kstar_std_mean"] = pd.to_numeric(degeneracy["kstar_std_mean"], errors="coerce")
    if degeneracy["degenerate_control"].dtype != bool:
        degeneracy["degenerate_control"] = degeneracy["degenerate_control"].astype(str).str.lower().eq("true")
    for domain in ("economics", "energy"):
        domain_frame = degeneracy.loc[degeneracy["domain"] == domain]
        cmdl_ok = bool(
            (domain_frame.loc[domain_frame["method"] == "CMDL", "kstar_std_mean"].dropna() > 0).any()
        )
        controls = domain_frame.loc[domain_frame["method"].isin(["No AC Encoder", "Uniform Lag"]), "degenerate_control"]
        control_ok = bool(not controls.empty and controls.all())
        status[domain] = "certified" if cmdl_ok and control_ok else "not_certified"
    return status


def _derive_l0_status(inputs: FigureInputs) -> dict[str, str]:
    status = {"synthetic": "certified"}
    forecast = inputs.realdata_forecast.copy()
    forecast["test_r2_mean"] = pd.to_numeric(forecast["test_r2_mean"], errors="coerce")
    for domain in ("economics", "energy"):
        domain_frame = forecast.loc[forecast["domain"] == domain]
        cmdl = domain_frame.loc[domain_frame["display_name"] == "CMDL", "test_r2_mean"].dropna()
        best_competing = domain_frame.loc[domain_frame["display_name"] != "CMDL", "test_r2_mean"].dropna()
        if cmdl.empty or best_competing.empty:
            status[domain] = "not_certified"
        elif float(cmdl.iloc[0]) < 0.0 and float(best_competing.max()) > 0.0:
            status[domain] = "ruled_out"
        elif float(cmdl.iloc[0]) >= float(best_competing.max()):
            status[domain] = "certified"
        else:
            status[domain] = "not_certified"
    return status


def _derive_layer_status(inputs: FigureInputs) -> pd.DataFrame:
    l0 = _derive_l0_status(inputs)
    l1 = _derive_l1_status(inputs)
    verdict = inputs.verdict.copy()
    verdict["domain"] = verdict["domain"].astype(str)
    verdict["layer"] = verdict["layer"].astype(str)
    verdict["verdict"] = verdict["verdict"].astype(str)

    rows: list[dict[str, str]] = []
    for domain in ("synthetic", "economics", "energy"):
        rows.append({"domain": domain, "layer": "L0", "status": l0.get(domain, "not_certified")})
        rows.append({"domain": domain, "layer": "L1", "status": l1.get(domain, "not_certified")})

        l2 = verdict.loc[
            (verdict["domain"] == domain) & (verdict["layer"] == "L2_structured_mechanism"), "verdict"
        ]
        rows.append(
            {
                "domain": domain,
                "layer": "L2",
                "status": "certified" if not l2.empty and l2.iloc[0] == "certified" else "not_certified",
            }
        )

        l3 = verdict.loc[
            (verdict["domain"] == domain) & (verdict["layer"] == "L3_directional_alignment"), "verdict"
        ]
        if not l3.empty and l3.iloc[0] == "certified":
            l3_status = "certified"
        elif not l3.empty and l3.iloc[0] == "mixed":
            l3_status = "not_claimed"
        else:
            l3_status = "not_certified"
        rows.append({"domain": domain, "layer": "L3", "status": l3_status})
    return pd.DataFrame(rows)


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _ordered_methods(methods: Iterable[str], include_grouped_ardl: bool = True) -> list[str]:
    order = METHOD_ORDER if include_grouped_ardl else SYNTHETIC_METHOD_ORDER
    available = set(methods)
    return [method for method in order if method in available]


def _draw_synthetic_panel(
    fig: plt.Figure,
    spec: gridspec.SubplotSpec,
    summary: pd.DataFrame,
    title: str = "A. Synthetic recovery",
    inside_ylabels: bool = False,
) -> list[plt.Axes]:
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1.15, 1.0], wspace=0.14)
    ax_mae = fig.add_subplot(inner[0, 0])
    ax_rho = fig.add_subplot(inner[0, 1], sharey=ax_mae)

    methods = _ordered_methods(summary["method"].dropna().unique(), include_grouped_ardl=False)
    y_lookup = {method: len(methods) - 1 - index for index, method in enumerate(methods)}
    offsets = {"linear": 0.13, "nonlinear": -0.13}

    for scenario in ("linear", "nonlinear"):
        scenario_frame = summary.loc[summary["scenario"] == scenario]
        for method in methods:
            y = y_lookup[method] + offsets[scenario]
            for metric, axis in (
                ("effective_kstar_mae", ax_mae),
                ("effective_kstar_spearman_rho", ax_rho),
            ):
                row = scenario_frame.loc[(scenario_frame["method"] == method) & (scenario_frame["metric"] == metric)]
                if row.empty:
                    continue
                mean = float(row["mean"].iloc[0])
                ci_low = float(row["ci_low"].iloc[0])
                ci_high = float(row["ci_high"].iloc[0])
                axis.errorbar(
                    mean,
                    y,
                    xerr=[[max(0.0, mean - ci_low)], [max(0.0, ci_high - mean)]],
                    fmt="o",
                    color=SCENARIO_COLOR[scenario],
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    markersize=4.3,
                    linewidth=1.2,
                    capsize=2.0,
                    zorder=3,
                )

    for axis in (ax_mae, ax_rho):
        axis.set_yticks(list(y_lookup.values()))
        axis.set_yticklabels([METHOD_LABEL[method] for method in methods])
        axis.grid(axis="x", color="#D7DCE2", linewidth=0.6, alpha=0.8)
        axis.set_ylim(-0.6, len(methods) - 0.4)
    if inside_ylabels:
        ax_mae.tick_params(axis="y", labelleft=False)
        for method in methods:
            ax_mae.text(
                0.018,
                y_lookup[method],
                METHOD_LABEL[method],
                transform=ax_mae.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=7.2,
                color="#102A43",
            )
    ax_rho.tick_params(axis="y", labelleft=False, left=False)
    ax_mae.set_xlabel("k* MAE (lower better)")
    ax_rho.set_xlabel("Spearman rho (higher better)")
    ax_mae.set_title(title, loc="left", fontweight="bold")
    ax_rho.set_xlim(-0.05, 1.05)
    ax_mae.set_xlim(left=0.0)

    legend_handles = [
        Line2D([0], [0], marker="o", color=SCENARIO_COLOR[scenario], linestyle="", label=scenario.title())
        for scenario in ("linear", "nonlinear")
    ]
    ax_rho.legend(
        handles=legend_handles,
        frameon=True,
        fancybox=False,
        edgecolor="#D7DCE2",
        facecolor="white",
        framealpha=0.94,
        loc="lower right",
        handlelength=1.0,
        borderpad=0.35,
    )
    return [ax_mae, ax_rho]


def _draw_decoupling_legend_box(axis: plt.Axes) -> None:
    box_x = 0.500
    box_y = 0.525
    box_width = 0.475
    box_height = 0.465
    axis.add_patch(
        Rectangle(
            (box_x, box_y),
            box_width,
            box_height,
            transform=axis.transAxes,
            facecolor="white",
            edgecolor="#BCCCDC",
            linewidth=0.7,
            alpha=0.94,
            zorder=4,
        )
    )

    left_x = box_x + 0.055
    right_x = box_x + 0.255
    text_dx = 0.030
    axis.text(box_x + 0.025, box_y + box_height - 0.045, "Method color", transform=axis.transAxes, fontsize=5.9, color="#52606D", va="center", zorder=5)
    method_rows = [
        ("CMDL", "Grouped ARDL"),
        ("No Recon Regularization", "No AC Encoder"),
        ("Plain LSTM", "Uniform Lag"),
        ("TFT", "GA-Net"),
    ]
    for row_index, row_methods in enumerate(method_rows):
        y = box_y + box_height - 0.100 - 0.065 * row_index
        for x, method in ((left_x, row_methods[0]), (right_x, row_methods[1])):
            if not method:
                continue
            axis.plot(
                x,
                y,
                marker="o",
                color=METHOD_COLOR[method],
                markerfacecolor=METHOD_COLOR[method],
                markeredgecolor=METHOD_COLOR[method],
                markersize=4.4,
                linestyle="",
                transform=axis.transAxes,
                zorder=5,
            )
            axis.text(x + text_dx, y, METHOD_LABEL[method], transform=axis.transAxes, fontsize=5.7, color="#243B53", va="center", zorder=5)

    separator_y = box_y + 0.132
    axis.plot([box_x + 0.020, box_x + box_width - 0.020], [separator_y, separator_y], color="#D7DCE2", linewidth=0.6, transform=axis.transAxes, zorder=5)
    axis.text(box_x + 0.025, box_y + 0.095, "Domain shape", transform=axis.transAxes, fontsize=5.9, color="#52606D", va="center", zorder=5)
    for x, domain in ((left_x, "economics"), (right_x, "energy")):
        axis.plot(
            x,
            box_y + 0.045,
            marker=DOMAIN_MARKER[domain],
            color="#243B53",
            markerfacecolor="white",
            markeredgecolor="#243B53",
            markeredgewidth=1.0,
            markersize=4.7,
            linestyle="",
            transform=axis.transAxes,
            zorder=5,
        )
        axis.text(x + text_dx, box_y + 0.045, DOMAIN_LABEL[domain], transform=axis.transAxes, fontsize=5.7, color="#243B53", va="center", zorder=5)


def _draw_decoupling_panel(axis: plt.Axes, summary: pd.DataFrame, title: str = "B. Forecast-mechanism decoupling") -> None:
    plot_frame = summary.copy()
    plot_frame["valid_l2"] = plot_frame["valid_l2"].fillna(False).astype(bool)
    undefined_y = 0.055
    undefined_lanes = {
        "Grouped ARDL": 0.035,
        "GA-Net": 0.045,
        "Plain LSTM": 0.055,
        "No AC Encoder": 0.075,
        "Uniform Lag": 0.095,
        "TFT": 0.115,
    }
    valid_jitter = {
        "CMDL": 0.032,
        "No Recon Regularization": -0.032,
        "TFT": 0.016,
    }

    axis.axhspan(0.0, 0.12, color="#F2F4F7", zorder=0)
    axis.axhline(0.0, color="#9AA5B1", linewidth=0.8, linestyle="--", zorder=1)
    axis.axvline(0.0, color="#9AA5B1", linewidth=0.8, linestyle="--", zorder=1)

    for _, row in plot_frame.iterrows():
        method = str(row["method"])
        domain = str(row["domain"])
        if method not in METHOD_COLOR:
            continue
        x = float(row["test_r2_mean"])
        x_low = float(row["test_r2_ci_low"])
        x_high = float(row["test_r2_ci_high"])
        valid_l2 = bool(row["valid_l2"])
        if valid_l2 and pd.notna(row["max_abs_rho_mean"]):
            y = float(row["max_abs_rho_mean"]) + valid_jitter.get(method, 0.0)
        else:
            y = undefined_lanes.get(method, undefined_y)
        marker = DOMAIN_MARKER.get(domain, "o")
        color = METHOD_COLOR[method]
        facecolor = "white"
        alpha = 0.95 if valid_l2 else 0.75
        axis.errorbar(
            x,
            y,
            xerr=[[max(0.0, x - x_low)], [max(0.0, x_high - x)]],
            fmt=marker,
            color=color,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=1.25,
            markersize=5.3,
            linewidth=1.0,
            capsize=2.0,
            alpha=alpha,
            zorder=3,
        )
        if valid_l2 and pd.notna(row["max_abs_rho_ci_low"]):
            y_low = float(row["max_abs_rho_ci_low"]) + valid_jitter.get(method, 0.0)
            y_high = float(row["max_abs_rho_ci_high"]) + valid_jitter.get(method, 0.0)
            axis.errorbar(
                x,
                y,
                yerr=[[max(0.0, y - y_low)], [max(0.0, y_high - y)]],
                fmt="none",
                ecolor=color,
                linewidth=1.0,
                capsize=2.0,
                alpha=alpha,
                zorder=2,
            )
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel("Test R2")
    axis.set_ylabel("Max stratifier |rho|")
    axis.set_xlim(-0.12, 0.68)
    axis.set_ylim(0.0, 1.0)
    axis.grid(color="#D7DCE2", linewidth=0.6, alpha=0.8)
    _draw_decoupling_legend_box(axis)


def _draw_stratifier_panel(
    axis: plt.Axes,
    summary: pd.DataFrame,
    title: str = "C. Stratifier alignment of learned lags",
    xlabel: str = "Seed-mean |Spearman rho| with 95% bootstrap CI",
) -> None:
    plot_frame = summary.copy().reset_index(drop=True)
    plot_frame["row_label"] = plot_frame.apply(
        lambda row: f"{DOMAIN_LABEL.get(row['domain'], row['domain'])}: {STRATIFIER_LABEL.get(row['stratifier'], row['stratifier'])}",
        axis=1,
    )
    y_values = np.arange(len(plot_frame))[::-1]
    domain_color = {"economics": "#1F77B4", "energy": "#2CA02C"}

    for y, (_, row) in zip(y_values, plot_frame.iterrows()):
        x = float(row["abs_rho_mean"])
        ci_low = float(row["abs_rho_ci_low"])
        ci_high = float(row["abs_rho_ci_high"])
        color = domain_color.get(str(row["domain"]), "#52606D")
        axis.errorbar(
            x,
            y,
            xerr=[[max(0.0, x - ci_low)], [max(0.0, ci_high - x)]],
            fmt="o",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=5.2,
            linewidth=1.25,
            capsize=2.2,
            zorder=3,
        )
        pass_rate = row.get("pass_rate")
        pass_text = "" if pd.isna(pass_rate) else f"{100.0 * float(pass_rate):.0f}%"
        axis.text(0.99, y, pass_text, ha="right", va="center", fontsize=6.7, color="#243B53")

    axis.set_yticks(y_values)
    axis.set_yticklabels(plot_frame["row_label"].tolist())
    axis.set_title(title, loc="left", fontweight="bold", pad=6)
    axis.set_xlabel(xlabel)
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(-0.6, len(plot_frame) - 0.05)
    axis.grid(axis="x", color="#D7DCE2", linewidth=0.6, alpha=0.8)
    axis.text(
        0.985,
        0.970,
        "Sig. seed %",
        ha="right",
        va="top",
        fontsize=6.3,
        color="#52606D",
        transform=axis.transAxes,
    )


def _draw_status_icon(axis: plt.Axes, x: float, y: float, status: str, radius: float = 0.095) -> None:
    style = STATUS_STYLE.get(status, STATUS_STYLE["not_certified"])
    color = style["color"]
    circle = Circle((x, y), radius=radius, facecolor=color, edgecolor="none", transform=axis.transAxes, zorder=2)
    axis.add_patch(circle)

    if status == "certified":
        axis.plot(
            [x - 0.040, x - 0.012, x + 0.052],
            [y - 0.002, y - 0.034, y + 0.040],
            color="white",
            linewidth=1.55,
            solid_capstyle="round",
            transform=axis.transAxes,
            zorder=3,
        )
    elif status == "ruled_out":
        axis.plot([x - 0.040, x + 0.040], [y - 0.040, y + 0.040], color="white", linewidth=1.45, transform=axis.transAxes, zorder=3)
        axis.plot([x - 0.040, x + 0.040], [y + 0.040, y - 0.040], color="white", linewidth=1.45, transform=axis.transAxes, zorder=3)
    elif status == "not_claimed":
        axis.plot([x - 0.048, x + 0.048], [y, y], color="white", linewidth=1.55, transform=axis.transAxes, zorder=3)
    else:
        axis.text(x, y, "?", ha="center", va="center", color="white", fontsize=9.0, fontweight="bold", transform=axis.transAxes, zorder=3)


def _draw_verdict_status_marker(axis: plt.Axes, x: float, y: float, status: str, size: float = 190.0, scale: float = 1.0) -> None:
    style = STATUS_STYLE.get(status, STATUS_STYLE["not_certified"])
    color = style["color"]
    axis.scatter([x], [y], s=size, color=color, edgecolor="white", linewidth=0.8, zorder=4)
    dx = 0.055 * scale
    dy = 0.050 * scale
    if status == "certified":
        axis.plot(
            [x - 1.08 * dx, x - 0.25 * dx, x + 1.20 * dx],
            [y - 0.02 * dy, y - 1.00 * dy, y + 1.15 * dy],
            color="white",
            linewidth=1.45 * scale,
            solid_capstyle="round",
            zorder=5,
        )
    elif status == "ruled_out":
        axis.plot([x - dx, x + dx], [y - dy, y + dy], color="white", linewidth=1.35 * scale, zorder=5)
        axis.plot([x - dx, x + dx], [y + dy, y - dy], color="white", linewidth=1.35 * scale, zorder=5)
    elif status == "not_claimed":
        axis.plot([x - 1.10 * dx, x + 1.10 * dx], [y, y], color="white", linewidth=1.45 * scale, zorder=5)
    else:
        axis.text(x, y - 0.004 * scale, "?", ha="center", va="center", color="white", fontsize=7.2 * scale, fontweight="bold", zorder=5)


def _draw_verdict_matrix(axis: plt.Axes, status_frame: pd.DataFrame, title: str = "Verdict matrix") -> None:
    axis.set_axis_off()
    axis.set_xlim(0, 4.20)
    axis.set_ylim(0, 4.15)
    axis.text(0.12, 3.94, title, ha="left", va="top", fontsize=9.3, fontweight="bold", color="#102A43")

    domains = ["synthetic", "economics", "energy"]
    layers = ["L0", "L1", "L2", "L3"]
    x_positions = [1.60, 2.20, 2.80, 3.40]
    y_positions = [2.92, 2.20, 1.48]
    table_left = 0.12
    table_right = 3.92

    for band_index, y in enumerate(y_positions):
        if band_index % 2 == 0:
            axis.add_patch(Rectangle((table_left, y - 0.32), table_right - table_left, 0.64, facecolor="#F8FAFC", edgecolor="none", zorder=0))

    for y in [3.36, 2.56, 1.84, 1.12]:
        axis.plot([table_left, table_right], [y, y], color="#D7DCE2", linewidth=0.75, zorder=1)
    for x in [1.25, 1.90, 2.50, 3.10, 3.70]:
        axis.plot([x, x], [1.12, 3.36], color="#EEF2F6", linewidth=0.55, zorder=1)

    axis.text(0.34, 3.53, "Domain", ha="left", va="center", fontsize=7.0, color="#52606D")
    for x, layer in zip(x_positions, layers):
        axis.text(x, 3.53, layer, ha="center", va="center", fontsize=7.0, color="#52606D", fontweight="bold")

    for domain, y in zip(domains, y_positions):
        axis.text(0.34, y, DOMAIN_LABEL[domain], ha="left", va="center", fontsize=7.3, color="#102A43")
        for x, layer in zip(x_positions, layers):
            status = status_frame.loc[
                (status_frame["domain"] == domain) & (status_frame["layer"] == layer), "status"
            ]
            _draw_verdict_status_marker(axis, x, y, status.iloc[0] if not status.empty else "not_certified", size=205.0, scale=1.0)

    legend_items = [
        ("certified", "supported"),
        ("not_certified", "not cert."),
        ("ruled_out", "ruled out"),
        ("not_claimed", "n/a"),
    ]
    for index, (status, label) in enumerate(legend_items):
        x = 0.42 + 0.92 * index
        y = 0.50
        _draw_verdict_status_marker(axis, x, y, status, size=95.0, scale=0.72)
        axis.text(x + 0.18, y, label, ha="left", va="center", fontsize=6.2, color="#52606D")


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: Iterable[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        target = output_dir / f"{stem}.{fmt}"
        fig.savefig(target, bbox_inches="tight", dpi=300 if fmt == "png" else None)
        paths.append(target)
    return paths


def build_figures(root: Path, plan: str, output_dir: Path, formats: Iterable[str], n_boot: int, tft_plan: str | None = None) -> list[Path]:
    _setup_style()
    inputs = _load_inputs(root, plan, tft_plan=tft_plan)
    synthetic = _synthetic_summary(inputs.synthetic, n_boot=n_boot)
    decoupling = _decoupling_summary(inputs, n_boot=n_boot)
    stratifiers = _stratifier_summary(inputs, n_boot=n_boot)
    verdict_status = _derive_layer_status(inputs)

    created: list[Path] = []

    fig = plt.figure(figsize=(7.8, 5.5), constrained_layout=False)
    top_grid = fig.add_gridspec(
        1,
        2,
        left=0.085,
        right=0.985,
        top=0.960,
        bottom=0.620,
        width_ratios=[1.14, 1.0],
        wspace=0.270,
    )
    bottom_grid = fig.add_gridspec(1, 1, left=0.220, right=0.985, top=0.465, bottom=0.100)
    _draw_synthetic_panel(fig, top_grid[0, 0], synthetic)
    decoupling_axis = fig.add_subplot(top_grid[0, 1])
    _draw_decoupling_panel(decoupling_axis, decoupling)
    stratifier_axis = fig.add_subplot(bottom_grid[0, 0])
    _draw_stratifier_panel(stratifier_axis, stratifiers)
    created.extend(_save_figure(fig, output_dir, "compact_evidence_panel", formats))
    plt.close(fig)

    fig = plt.figure(figsize=(6.2, 3.35), constrained_layout=False)
    spec = gridspec.GridSpec(1, 1, figure=fig)[0]
    _draw_synthetic_panel(fig, spec, synthetic, title="Synthetic recovery")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.17)
    created.extend(_save_figure(fig, output_dir, "panel_synthetic_recovery", formats))
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.7, 3.10))
    _draw_decoupling_panel(axis, decoupling, title="Forecast-mechanism decoupling")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.86, bottom=0.18)
    created.extend(_save_figure(fig, output_dir, "panel_realdata_decoupling", formats))
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.9, 3.10))
    _draw_stratifier_panel(axis, stratifiers, title="Stratifier alignment of learned lags", xlabel="Mean |rho| (95% bootstrap CI)")
    fig.subplots_adjust(left=0.48, right=0.94, top=0.86, bottom=0.18)
    created.extend(_save_figure(fig, output_dir, "panel_stratifier_alignment", formats))
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(3.35, 2.15))
    _draw_verdict_matrix(axis, verdict_status, title="Verdict matrix")
    fig.tight_layout(pad=0.15)
    created.extend(_save_figure(fig, output_dir, "verdict_matrix_icon", formats))
    plt.close(fig)

    return created


def _copy_to_paper_img(paths: Iterable[Path], root: Path) -> list[Path]:
    target_dir = root / "paper_draft" / "img"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in paths:
        target = target_dir / path.name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact CMDL paper figures from locked CSV outputs.")
    parser.add_argument("--plan", default=DEFAULT_PLAN, help="Experiment plan name under outputs/notebook_* and outputs/paper_assets.")
    parser.add_argument("--root", type=Path, default=_repo_root(), help="Repository root. Defaults to the current script's repo root.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Figure output directory. Defaults to outputs/paper_assets/<plan>/compact_figures.")
    parser.add_argument("--formats", nargs="+", default=["png", "svg"], choices=["png", "svg", "pdf"], help="Output formats.")
    parser.add_argument("--n-boot", type=int, default=5000, help="Bootstrap resamples for displayed confidence intervals.")
    parser.add_argument("--copy-to-paper-img", action="store_true", help="Copy generated figures into paper_draft/img after building them.")
    parser.add_argument(
        "--tft-plan",
        default=DEFAULT_TFT_PLAN,
        help="Plan name under outputs/notebook_*/<tft_plan>/tft holding TFT runs. Empty string disables TFT.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = root / "outputs" / "paper_assets" / args.plan / "compact_figures"
    else:
        output_dir = output_dir.resolve()

    tft_plan = args.tft_plan or None
    created = build_figures(root=root, plan=args.plan, output_dir=output_dir, formats=args.formats, n_boot=args.n_boot, tft_plan=tft_plan)
    copied = _copy_to_paper_img(created, root) if args.copy_to_paper_img else []

    print("Generated compact figures:")
    for path in created:
        print(f"  {path}")
    if copied:
        print("Copied to paper_draft/img:")
        for path in copied:
            print(f"  {path}")


if __name__ == "__main__":
    main()