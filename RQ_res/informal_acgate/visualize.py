"""Visualization utilities for isolated Informal RQ experiments."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PACKAGE_ROOT = Path(__file__).resolve().parent
RQ_RES_ROOT = PACKAGE_ROOT.parent
if str(RQ_RES_ROOT) not in sys.path:
    sys.path.insert(0, str(RQ_RES_ROOT))

from informal_acgate.aggregate import aggregate_summaries


DEFAULT_OUTPUT_DIR = RQ_RES_ROOT / "outputs" / "informal_acgate" / "suite_region_proxy"
DEFAULT_FIGURE_DIR = RQ_RES_ROOT / "outputs" / "informal_acgate" / "figures_region_proxy"
DEFAULT_FULLSPAN_FIGURE_DIR = RQ_RES_ROOT / "outputs" / "informal_acgate" / "figures_fullspan_region_proxy"
PLOT_STYLE = "whitegrid"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Informal RQ result visualizations.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--figure-dir", type=str, default=None)
    parser.add_argument("--comparison-csv", type=str, default=None)
    parser.add_argument("--matrix-runs-csv", type=str, default=None)
    return parser.parse_args()


def _default_figure_dir(output_dir: str | Path) -> Path:
    output_name = Path(output_dir).resolve().name
    if "fullspan" in output_name:
        return DEFAULT_FULLSPAN_FIGURE_DIR
    return DEFAULT_FIGURE_DIR


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_group(row: pd.Series) -> str:
    feature_bundle = str(row.get("feature_bundle", ""))
    model = str(row.get("model", ""))
    ablation = str(row.get("ablation", ""))
    scenario = str(row.get("scenario", ""))
    is_fullspan = "fullspan" in scenario or "fullspan" in feature_bundle
    if is_fullspan:
        if model == "plain_lstm":
            return "Fullspan Plain LSTM"
        if ablation == "no_ac_encoder":
            return "Fullspan No AC Encoder"
        if ablation == "uniform_lag":
            return "Fullspan Uniform Lag"
        if ablation == "no_recon_regularization":
            return "Fullspan No Recon"
        return "Fullspan CMDL"
    if feature_bundle.startswith("single"):
        return "Single CMDL"
    if model == "plain_lstm":
        return "Multiseq Plain LSTM"
    if ablation == "no_ac_encoder":
        return "No AC Encoder"
    if ablation == "uniform_lag":
        return "Uniform Lag"
    if ablation == "no_recon_regularization":
        return "No Recon"
    return "Multiseq CMDL"


def _ordered_groups(dataframe: pd.DataFrame) -> list[str]:
    preferred = [
        "Single CMDL",
        "Multiseq CMDL",
        "Multiseq Plain LSTM",
        "No AC Encoder",
        "Uniform Lag",
        "No Recon",
        "Fullspan CMDL",
        "Fullspan Plain LSTM",
        "Fullspan No AC Encoder",
        "Fullspan Uniform Lag",
        "Fullspan No Recon",
    ]
    present = set(dataframe["run_group"].dropna().astype(str))
    return [name for name in preferred if name in present]


def load_comparison(output_dir: str | Path, comparison_csv: str | Path | None = None) -> pd.DataFrame:
    output_root = Path(output_dir).resolve()
    if comparison_csv is None:
        comparison_path = output_root / "comparison.csv"
        aggregate_summaries(output_dir=output_root, output_csv=comparison_path)
    else:
        comparison_path = Path(comparison_csv).resolve()
    dataframe = pd.read_csv(comparison_path)
    if dataframe.empty:
        raise ValueError(f"No experiment summaries were found under {output_root}")
    dataframe["run_group"] = dataframe.apply(_run_group, axis=1)
    return dataframe


def write_seed_summary(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    metric_columns = [
        "test_mse",
        "test_mae",
        "test_kstar_proxy_spearman_adjusted_rho",
        "test_kstar_proxy_mean_spearman_adjusted_rho",
        "test_kstar_std",
        "test_lag_gate_sensitivity_range",
        "test_omega_entropy_mean",
    ]
    summary_rows: list[dict[str, Any]] = []
    for run_group, group in dataframe.groupby("run_group", sort=False):
        row: dict[str, Any] = {"run_group": run_group, "seed_count": int(group["seed"].nunique())}
        for metric_name in metric_columns:
            if metric_name not in group.columns:
                continue
            values = pd.to_numeric(group[metric_name], errors="coerce")
            row[f"{metric_name}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{metric_name}_std"] = float(values.std(ddof=0)) if values.notna().sum() > 1 else 0.0
            if "rho" in metric_name:
                row[f"{metric_name}_positive_seed_share"] = float((values > 0.0).mean()) if values.notna().any() else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    output_path = figure_dir / "seed_summary.csv"
    summary.to_csv(output_path, index=False)
    return output_path


def plot_error_bars(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    order = _ordered_groups(dataframe)
    long_frame = dataframe.melt(
        id_vars=["run_group", "seed"],
        value_vars=["test_mse", "test_mae"],
        var_name="metric",
        value_name="value",
    )
    long_frame["metric"] = long_frame["metric"].map({"test_mse": "MSE", "test_mae": "MAE"})
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, metric in zip(axes, ["MSE", "MAE"]):
        subset = long_frame.loc[long_frame["metric"] == metric]
        sns.barplot(data=subset, x="run_group", y="value", order=order, errorbar="sd", ax=axis, color="#5b8db8")
        sns.stripplot(data=subset, x="run_group", y="value", order=order, ax=axis, color="#23395b", size=3, alpha=0.45)
        axis.set_title(f"Test {metric} Across Seeds")
        axis.set_xlabel("")
        axis.set_ylabel(metric)
        axis.tick_params(axis="x", rotation=35)
    output_path = figure_dir / "test_error_by_model.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_mechanism_metrics(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    order = _ordered_groups(dataframe)
    metrics = [
        ("test_kstar_proxy_spearman_adjusted_rho", "Anchor-Adjusted rho"),
        ("test_kstar_std", "k* Std"),
        ("test_omega_entropy_mean", "Omega Entropy"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, (column_name, title) in zip(axes, metrics):
        values = dataframe.copy()
        values[column_name] = pd.to_numeric(values[column_name], errors="coerce")
        sns.boxplot(data=values, x="run_group", y=column_name, order=order, ax=axis, color="#c9d8b6")
        sns.stripplot(data=values, x="run_group", y=column_name, order=order, ax=axis, color="#2f4f4f", size=3, alpha=0.55)
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=35)
    output_path = figure_dir / "mechanism_metrics_by_model.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_seed_stability(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    order = _ordered_groups(dataframe)
    figure, axis = plt.subplots(figsize=(12, 5.2), constrained_layout=True)
    plot_frame = dataframe.copy()
    plot_frame["run_group"] = pd.Categorical(plot_frame["run_group"], categories=order, ordered=True)
    sns.lineplot(data=plot_frame.sort_values(["run_group", "seed"]), x="seed", y="test_mse", hue="run_group", marker="o", ax=axis)
    axis.set_title("Test MSE Seed Stability")
    axis.set_xlabel("Seed")
    axis.set_ylabel("Test MSE")
    axis.legend(title="Run", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    output_path = figure_dir / "seed_stability_mse.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _best_multiseq_cmdl_row(dataframe: pd.DataFrame) -> pd.Series:
    mask = (dataframe["run_group"] == "Multiseq CMDL") & dataframe["test_mse"].notna()
    if not mask.any():
        mask = dataframe["test_mse"].notna()
    if not mask.any():
        raise ValueError("No run with finite test_mse is available for detailed plots")
    return dataframe.loc[mask].sort_values("test_mse", ascending=True).iloc[0]


def _detail_cmdl_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    mask = (dataframe["model"] == "cmdl") & (dataframe["ablation"] == "none") & dataframe["summary_path"].notna()
    if not mask.any():
        raise ValueError("No CMDL non-ablation runs are available for detail plots")
    candidates = dataframe.loc[mask].copy()
    multiseq = candidates.loc[candidates["feature_bundle"].astype(str).str.contains("multiseq", case=False, na=False)]
    if not multiseq.empty:
        return multiseq
    return candidates


def _prediction_path(summary_path: str | Path) -> Path:
    return Path(summary_path).resolve().parent / "predictions.csv"


def _omega_columns(predictions: pd.DataFrame) -> list[str]:
    columns = [column for column in predictions.columns if re.fullmatch(r"omega_\d+", column)]
    if not columns:
        raise ValueError("No omega_* columns were found in predictions.csv")
    return columns


def _load_detail_predictions(dataframe: pd.DataFrame) -> pd.DataFrame:
    prediction_frames: list[pd.DataFrame] = []
    for _, row in _detail_cmdl_rows(dataframe).iterrows():
        path = _prediction_path(row["summary_path"])
        if not path.exists():
            continue
        predictions = pd.read_csv(path)
        predictions["seed"] = row.get("seed")
        predictions["experiment"] = row.get("experiment")
        predictions["run_group"] = row.get("run_group")
        predictions["scenario"] = row.get("scenario", "")
        prediction_frames.append(predictions)
    if not prediction_frames:
        raise ValueError("No predictions.csv files were available for detail plots")
    return pd.concat(prediction_frames, ignore_index=True)


def plot_best_omega_heatmap(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    best_row = _best_multiseq_cmdl_row(dataframe)
    predictions = pd.read_csv(_prediction_path(best_row["summary_path"]))
    omega_columns = _omega_columns(predictions)
    heatmap_frame = predictions.drop_duplicates("entity_code").set_index("entity_code")[omega_columns]
    figure, axis = plt.subplots(figsize=(6.5, 8.0), constrained_layout=True)
    sns.heatmap(heatmap_frame, cmap="viridis", vmin=0.0, vmax=1.0, cbar_kws={"label": "omega"}, ax=axis)
    axis.set_title(f"Representative Best-Seed Omega: {best_row['experiment']}")
    axis.set_xlabel("Lag")
    axis.set_ylabel("Region")
    output_path = figure_dir / "best_cmdl_omega_heatmap.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_best_proxy_kstar(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    best_row = _best_multiseq_cmdl_row(dataframe)
    summary = _read_json(Path(best_row["summary_path"]))
    proxy_names = list(summary.get("data", {}).get("proxy_columns", []))
    predictions = pd.read_csv(_prediction_path(best_row["summary_path"])).drop_duplicates("entity_code")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for proxy_index, axis in enumerate(axes, start=1):
        column_name = f"proxy_{proxy_index}_true"
        if column_name not in predictions.columns:
            axis.set_visible(False)
            continue
        sns.regplot(data=predictions, x=column_name, y="k_star", ax=axis, ci=None, scatter_kws={"s": 38, "alpha": 0.8})
        label = proxy_names[proxy_index - 1] if proxy_index - 1 < len(proxy_names) else column_name
        axis.set_title(label)
        axis.set_xlabel("Standardized proxy")
        axis.set_ylabel("k*" if proxy_index == 1 else "")
    output_path = figure_dir / "best_cmdl_proxy_kstar.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_best_predictions(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    best_row = _best_multiseq_cmdl_row(dataframe)
    predictions = pd.read_csv(_prediction_path(best_row["summary_path"]))
    figure, axis = plt.subplots(figsize=(5.8, 5.2), constrained_layout=True)
    sns.scatterplot(data=predictions, x="y_true", y="y_pred", hue="entity_code", legend=False, ax=axis)
    lower = float(min(predictions["y_true"].min(), predictions["y_pred"].min()))
    upper = float(max(predictions["y_true"].max(), predictions["y_pred"].max()))
    axis.plot([lower, upper], [lower, upper], color="#555555", linewidth=1.0)
    axis.set_title(f"Predictions: {best_row['experiment']}")
    axis.set_xlabel("y true")
    axis.set_ylabel("y predicted")
    output_path = figure_dir / "best_cmdl_predictions.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_seed_mean_omega_heatmap(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    predictions = _load_detail_predictions(dataframe)
    omega_columns = _omega_columns(predictions)
    heatmap_frame = predictions.groupby("entity_code", sort=True)[omega_columns].mean()
    figure, axis = plt.subplots(figsize=(7.0, 8.2), constrained_layout=True)
    sns.heatmap(heatmap_frame, cmap="viridis", vmin=0.0, vmax=1.0, cbar_kws={"label": "mean omega"}, ax=axis)
    axis.set_title("Seed-Mean CMDL Omega by Region")
    axis.set_xlabel("Lag")
    axis.set_ylabel("Region")
    output_path = figure_dir / "seed_mean_cmdl_omega_heatmap.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_seed_std_omega_heatmap(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    predictions = _load_detail_predictions(dataframe)
    omega_columns = _omega_columns(predictions)
    seed_entity = predictions.groupby(["seed", "entity_code"], sort=True)[omega_columns].mean().reset_index()
    heatmap_frame = seed_entity.groupby("entity_code", sort=True)[omega_columns].std(ddof=0).fillna(0.0)
    figure, axis = plt.subplots(figsize=(7.0, 8.2), constrained_layout=True)
    sns.heatmap(heatmap_frame, cmap="mako", vmin=0.0, cbar_kws={"label": "seed std"}, ax=axis)
    axis.set_title("CMDL Omega Seed Uncertainty by Region")
    axis.set_xlabel("Lag")
    axis.set_ylabel("Region")
    output_path = figure_dir / "seed_std_cmdl_omega_heatmap.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_year_lag_omega_heatmap(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    predictions = _load_detail_predictions(dataframe)
    omega_columns = _omega_columns(predictions)
    year_lag = predictions.groupby("year", sort=True)[omega_columns].mean().transpose()
    year_lag.index = [column.replace("omega_", "lag ") for column in year_lag.index]
    figure, axis = plt.subplots(figsize=(max(6.0, 0.9 * len(year_lag.columns)), 3.8), constrained_layout=True)
    sns.heatmap(year_lag, cmap="viridis", vmin=0.0, vmax=1.0, cbar_kws={"label": "mean omega"}, ax=axis)
    axis.set_title("CMDL Mean Omega by Prediction Year")
    axis.set_xlabel("Prediction year")
    axis.set_ylabel("Lag")
    output_path = figure_dir / "year_lag_cmdl_omega_heatmap.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_kstar_distribution(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    predictions = _load_detail_predictions(dataframe)
    plot_frame = predictions.copy()
    plot_frame["year"] = plot_frame["year"].astype(str)
    figure, axis = plt.subplots(figsize=(max(7.0, 1.2 * plot_frame["year"].nunique()), 5.0), constrained_layout=True)
    sns.boxplot(data=plot_frame, x="year", y="k_star", ax=axis, color="#d3d8c8")
    sns.stripplot(data=plot_frame, x="year", y="k_star", ax=axis, color="#38546b", size=2.5, alpha=0.35)
    axis.set_title("CMDL k* Distribution Across Seeds and Regions")
    axis.set_xlabel("Prediction year")
    axis.set_ylabel("k*")
    output_path = figure_dir / "cmdl_kstar_distribution.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_seed_mean_proxy_kstar(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    predictions = _load_detail_predictions(dataframe)
    proxy_columns = [column for column in predictions.columns if re.fullmatch(r"proxy_\d+_true", column)]
    if not proxy_columns:
        raise ValueError("No proxy_*_true columns were found in predictions.csv")
    proxy_column = proxy_columns[0]
    entity_frame = (
        predictions.groupby("entity_code", sort=True)
        .agg(proxy_value=(proxy_column, "mean"), kstar_mean=("k_star", "mean"), kstar_std=("k_star", "std"))
        .reset_index()
    )
    entity_frame["kstar_std"] = entity_frame["kstar_std"].fillna(0.0)
    figure, axis = plt.subplots(figsize=(7.0, 5.4), constrained_layout=True)
    axis.errorbar(
        entity_frame["proxy_value"],
        entity_frame["kstar_mean"],
        yerr=entity_frame["kstar_std"],
        fmt="o",
        color="#34675c",
        ecolor="#8aa39b",
        elinewidth=1.0,
        capsize=2.0,
        alpha=0.85,
    )
    sns.regplot(data=entity_frame, x="proxy_value", y="kstar_mean", ax=axis, scatter=False, ci=None, color="#4b4b4b")
    axis.set_title("Seed-Mean CMDL Proxy vs k*")
    axis.set_xlabel(proxy_column)
    axis.set_ylabel("Mean k*")
    output_path = figure_dir / "seed_mean_cmdl_proxy_kstar.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def build_visualizations(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    figure_dir: str | Path | None = None,
    comparison_csv: str | Path | None = None,
) -> dict[str, str]:
    sns.set_theme(style=PLOT_STYLE)
    figure_root = _default_figure_dir(output_dir) if figure_dir is None else Path(figure_dir).resolve()
    figure_root.mkdir(parents=True, exist_ok=True)
    dataframe = load_comparison(output_dir=output_dir, comparison_csv=comparison_csv)
    comparison_path = figure_root / "comparison_for_figures.csv"
    dataframe.to_csv(comparison_path, index=False)

    best_cmdl_omega = str(plot_best_omega_heatmap(dataframe, figure_root))
    best_cmdl_proxy_kstar = str(plot_best_proxy_kstar(dataframe, figure_root))
    best_cmdl_predictions = str(plot_best_predictions(dataframe, figure_root))
    paths = {
        "comparison_csv": str(comparison_path),
        "seed_summary_csv": str(write_seed_summary(dataframe, figure_root)),
        "test_error_by_model": str(plot_error_bars(dataframe, figure_root)),
        "mechanism_metrics_by_model": str(plot_mechanism_metrics(dataframe, figure_root)),
        "seed_stability_mse": str(plot_seed_stability(dataframe, figure_root)),
        "seed_mean_cmdl_omega_heatmap": str(plot_seed_mean_omega_heatmap(dataframe, figure_root)),
        "seed_std_cmdl_omega_heatmap": str(plot_seed_std_omega_heatmap(dataframe, figure_root)),
        "year_lag_cmdl_omega_heatmap": str(plot_year_lag_omega_heatmap(dataframe, figure_root)),
        "cmdl_kstar_distribution": str(plot_kstar_distribution(dataframe, figure_root)),
        "seed_mean_cmdl_proxy_kstar": str(plot_seed_mean_proxy_kstar(dataframe, figure_root)),
        "best_cmdl_omega_heatmap": best_cmdl_omega,
        "best_cmdl_proxy_kstar": best_cmdl_proxy_kstar,
        "best_cmdl_predictions": best_cmdl_predictions,
        "best_multiseq_cmdl_omega_heatmap": best_cmdl_omega,
        "best_multiseq_cmdl_proxy_kstar": best_cmdl_proxy_kstar,
        "best_multiseq_cmdl_predictions": best_cmdl_predictions,
    }
    with (figure_root / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(paths, handle, indent=2, ensure_ascii=True)
    return paths


def _matrix_figure_dir(all_runs_csv: str | Path, figure_dir: str | Path | None) -> Path:
    if figure_dir is not None:
        return Path(figure_dir).resolve()
    return Path(all_runs_csv).resolve().parent / "figures"


def _prepare_matrix_frame(all_runs_csv: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(all_runs_csv)
    if dataframe.empty:
        raise ValueError(f"No matrix runs are available in {all_runs_csv}")
    if "run_group" not in dataframe.columns:
        dataframe["run_group"] = dataframe.apply(_run_group, axis=1)
    for column_name in ["test_mse", "test_mae", "test_kstar_std", "test_kstar_proxy_spearman_adjusted_rho"]:
        if column_name in dataframe.columns:
            dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors="coerce")
    return dataframe


def plot_matrix_error_by_variant(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    plot_frame = dataframe.loc[dataframe["test_mse"].notna()].copy()
    order = plot_frame.groupby("variant_id")["test_mse"].mean().sort_values().index.tolist()
    figure, axis = plt.subplots(figsize=(max(10.0, 0.42 * len(order)), 6.2), constrained_layout=True)
    sns.barplot(data=plot_frame, x="variant_id", y="test_mse", hue="run_group", order=order, errorbar="sd", ax=axis)
    axis.set_title("Matrix Test MSE by Variant")
    axis.set_xlabel("")
    axis.set_ylabel("Test MSE")
    axis.tick_params(axis="x", rotation=70)
    axis.legend(title="Run", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    output_path = figure_dir / "matrix_test_mse_by_variant.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_matrix_mechanism_by_variant(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    cmdl = dataframe.loc[dataframe["run_group"].eq("CMDL")].copy()
    metrics = [
        ("test_kstar_proxy_spearman_adjusted_rho", "Proxy-k* rho"),
        ("test_kstar_std", "k* Std"),
        ("test_omega_entropy_mean", "Omega Entropy"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    for axis, (column_name, title) in zip(axes, metrics):
        if column_name not in cmdl.columns:
            axis.set_visible(False)
            continue
        values = cmdl.loc[cmdl[column_name].notna()].copy()
        order = values.groupby("variant_id")[column_name].mean().sort_values().index.tolist()
        sns.boxplot(data=values, x="variant_id", y=column_name, order=order, color="#d9dec8", ax=axis)
        sns.stripplot(data=values, x="variant_id", y=column_name, order=order, color="#314d5a", size=2.5, alpha=0.45, ax=axis)
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=70)
    output_path = figure_dir / "matrix_mechanism_by_variant.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_matrix_sample_counts(dataframe: pd.DataFrame, figure_dir: Path) -> Path:
    count_columns = ["train_effective_samples", "val_effective_samples", "test_effective_samples"]
    available = [column for column in count_columns if column in dataframe.columns]
    sample_frame = dataframe.drop_duplicates("variant_id")[["variant_id", "track", *available]].copy()
    long_frame = sample_frame.melt(id_vars=["variant_id", "track"], value_vars=available, var_name="split", value_name="samples")
    long_frame["split"] = long_frame["split"].str.replace("_effective_samples", "", regex=False)
    figure, axis = plt.subplots(figsize=(max(10.0, 0.42 * sample_frame["variant_id"].nunique()), 5.8), constrained_layout=True)
    sns.barplot(data=long_frame, x="variant_id", y="samples", hue="split", ax=axis)
    axis.set_title("Effective Supervised Samples by Variant")
    axis.set_xlabel("")
    axis.set_ylabel("Samples after lag warm-up")
    axis.tick_params(axis="x", rotation=70)
    output_path = figure_dir / "matrix_effective_samples.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_matrix_falsification(dataframe: pd.DataFrame, figure_dir: Path) -> Path | None:
    if "proxy_perturbation" not in dataframe.columns:
        return None
    cmdl = dataframe.loc[dataframe["run_group"].eq("CMDL")].copy()
    subset = cmdl.loc[
        cmdl["track"].isin(["fullspan_income", "falsification"])
        & cmdl["test_kstar_proxy_spearman_adjusted_rho"].notna()
    ].copy()
    if subset.empty:
        return None
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    sns.boxplot(data=subset, x="proxy_perturbation", y="test_kstar_proxy_spearman_adjusted_rho", hue="track", ax=axis)
    sns.stripplot(
        data=subset,
        x="proxy_perturbation",
        y="test_kstar_proxy_spearman_adjusted_rho",
        color="#283845",
        size=3,
        alpha=0.5,
        dodge=True,
        ax=axis,
    )
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_title("Real vs Falsified Proxy Mechanism Metric")
    axis.set_xlabel("Proxy perturbation")
    axis.set_ylabel("Adjusted proxy-k* rho")
    output_path = figure_dir / "matrix_falsification_proxy_kstar.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def plot_matrix_grid(dataframe: pd.DataFrame, figure_dir: Path) -> Path | None:
    grid = dataframe.loc[(dataframe["track"] == "capacity_gate_grid") & dataframe["test_mse"].notna()].copy()
    if grid.empty:
        return None
    summary = grid.groupby(["variant_id", "d_model", "temperature", "dropout"], dropna=False)["test_mse"].mean().reset_index()
    figure, axis = plt.subplots(figsize=(8.0, 5.8), constrained_layout=True)
    scatter = axis.scatter(
        summary["d_model"],
        summary["temperature"],
        c=summary["test_mse"],
        s=80 + 350 * pd.to_numeric(summary["dropout"], errors="coerce").fillna(0.05),
        cmap="viridis_r",
        alpha=0.85,
        edgecolors="#333333",
        linewidths=0.5,
    )
    figure.colorbar(scatter, ax=axis, label="Mean test MSE")
    axis.set_title("Capacity/Gate Grid Screen")
    axis.set_xlabel("d_model")
    axis.set_ylabel("temperature")
    output_path = figure_dir / "matrix_capacity_gate_grid.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def build_matrix_visualizations(all_runs_csv: str | Path, figure_dir: str | Path | None = None) -> dict[str, str]:
    sns.set_theme(style=PLOT_STYLE)
    figure_root = _matrix_figure_dir(all_runs_csv, figure_dir)
    figure_root.mkdir(parents=True, exist_ok=True)
    dataframe = _prepare_matrix_frame(all_runs_csv)
    paths: dict[str, str] = {
        "matrix_test_mse_by_variant": str(plot_matrix_error_by_variant(dataframe, figure_root)),
        "matrix_mechanism_by_variant": str(plot_matrix_mechanism_by_variant(dataframe, figure_root)),
        "matrix_effective_samples": str(plot_matrix_sample_counts(dataframe, figure_root)),
    }
    falsification_path = plot_matrix_falsification(dataframe, figure_root)
    if falsification_path is not None:
        paths["matrix_falsification_proxy_kstar"] = str(falsification_path)
    grid_path = plot_matrix_grid(dataframe, figure_root)
    if grid_path is not None:
        paths["matrix_capacity_gate_grid"] = str(grid_path)
    with (figure_root / "matrix_figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(paths, handle, indent=2, ensure_ascii=True)
    return paths


def main() -> None:
    args = parse_args()
    if args.matrix_runs_csv:
        paths = build_matrix_visualizations(args.matrix_runs_csv, args.figure_dir)
        print(f"Built {len(paths)} Informal matrix visualization artifacts.")
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    paths = build_visualizations(args.output_dir, args.figure_dir, args.comparison_csv)
    print(f"Built {len(paths)} Informal visualization artifacts.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
