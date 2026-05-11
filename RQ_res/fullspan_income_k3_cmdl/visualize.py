from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
REPORT_PATH = ROOT / "FIGURE_EXPLANATIONS_ZH.md"
LEGACY_REPORT_PATH = ROOT / "FIGURE_EXPLANATIONS.md"

COLORS = {
    "blue": "#2f6f9f",
    "teal": "#3f8f8a",
    "green": "#5f8f4e",
    "gold": "#c08a2b",
    "red": "#b55a4b",
    "purple": "#7b6aa8",
    "gray": "#6b7280",
    "light_gray": "#e5e7eb",
    "dark": "#1f2937",
}


def configure_plot_style() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.edgecolor"] = "#d1d5db"
    plt.rcParams["axes.labelcolor"] = COLORS["dark"]
    plt.rcParams["xtick.color"] = COLORS["dark"]
    plt.rcParams["ytick.color"] = COLORS["dark"]


def load_data() -> dict[str, pd.DataFrame]:
    tables = {
        "runs": pd.read_csv(DATA_DIR / "runs_summary.csv"),
        "predictions": pd.read_csv(DATA_DIR / "predictions_all_seeds.csv"),
        "entity_mean": pd.read_csv(DATA_DIR / "entity_summary_seed_mean.csv"),
        "entity_seed": pd.read_csv(DATA_DIR / "entity_summary_by_seed.csv"),
        "baselines": pd.read_csv(DATA_DIR / "baseline_summary.csv"),
        "history": pd.read_csv(DATA_DIR / "history_all_seeds.csv"),
    }
    text_columns = {
        "experiment",
        "scenario",
        "model",
        "ablation",
        "feature_bundle",
        "missing_policy",
        "proxy_mode",
        "proxy_construction_fit_window",
        "proxy_construction_description",
        "summary_path",
        "matrix_name",
        "variant_id",
        "track",
        "matrix_scenario",
        "interpretation",
        "run_spec",
        "matrix_output_dir",
        "run_dir",
        "proxy_perturbation",
        "evidence_scope",
        "sequence_transform",
        "run_group",
        "entity_code",
        "entity_name",
    }
    for frame in tables.values():
        for column in frame.columns.difference(text_columns):
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().any():
                frame[column] = converted
    return tables


def save_figure(fig: plt.Figure, filename: str, title: str, description: str, manifest: list[dict[str, str]]) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    manifest.append(
        {
            "filename": filename,
            "title": title,
            "description": description,
        }
    )
    return path


def format_stat(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def plot_timeline(manifest: list[dict[str, str]]) -> None:
    spans = [
        ("完整面板", 2006, 2023, COLORS["light_gray"]),
        ("训练/统计拟合窗口", 2006, 2018, COLORS["blue"]),
        ("验证窗口", 2016, 2020, COLORS["gold"]),
        ("配置测试窗口", 2018, 2023, COLORS["purple"]),
        ("K=3后有效预测年", 2021, 2023, COLORS["red"]),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for idx, (label, start, end, color) in enumerate(spans):
        y = len(spans) - idx - 1
        ax.broken_barh([(start, end - start + 1)], (y - 0.32, 0.64), facecolors=color, alpha=0.9)
        text_color = "white" if color != COLORS["light_gray"] else COLORS["dark"]
        ax.text((start + end) / 2 + 0.5, y, f"{label}\n{start}-{end}", ha="center", va="center", color=text_color, fontsize=9)
    ax.set_xlim(2005.5, 2024.5)
    ax.set_ylim(-0.8, len(spans) - 0.2)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.set_xlabel("年份")
    ax.set_title("fullspan_income_k3 + CMDL 的数据窗口与有效预测年", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    save_figure(
        fig,
        "01_data_window_timeline.png",
        "数据窗口与有效预测年",
        "展示完整面板、训练统计窗口、验证窗口、配置测试窗口，以及K=3后真正进入预测表的2021-2023年。",
        manifest,
    )


def plot_seed_robustness(runs: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    metrics = [("test_mse", "Test MSE"), ("test_mae", "Test MAE")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    rng = np.random.default_rng(20260510)
    for ax, (column, label) in zip(axes, metrics):
        values = runs[column].astype(float).to_numpy()
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.boxplot(values, positions=[0], widths=0.28, patch_artist=True, boxprops={"facecolor": "#dbeafe", "edgecolor": COLORS["blue"]}, medianprops={"color": COLORS["red"], "linewidth": 1.8}, whiskerprops={"color": COLORS["blue"]}, capprops={"color": COLORS["blue"]})
        ax.scatter(jitter, values, color=COLORS["dark"], s=24, alpha=0.75, zorder=3)
        mean = values.mean()
        std = values.std(ddof=1)
        ax.axhline(mean, color=COLORS["gold"], linestyle="--", linewidth=1.5)
        ax.text(0.2, mean, f"均值 {mean:.4f}\nstd {std:.4f}", va="center", fontsize=9, color=COLORS["dark"])
        ax.set_xticks([0])
        ax.set_xticklabels([label])
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#e5e7eb")
    fig.suptitle("20个seed下的预测稳健性", x=0.02, y=1.02, ha="left", fontsize=13, fontweight="bold")
    save_figure(
        fig,
        "02_seed_forecast_robustness.png",
        "20个seed下的预测稳健性",
        "展示test MSE和test MAE在20个seed上的分布，避免只依赖最佳seed。",
        manifest,
    )


def plot_forecast_trajectory(predictions: pd.DataFrame, manifest: list[dict[str, str]]) -> pd.DataFrame:
    grouped = (
        predictions.groupby("year")
        .agg(
            y_true=("y_true", "mean"),
            y_pred_mean=("y_pred", "mean"),
            y_pred_q10=("y_pred", lambda values: values.quantile(0.10)),
            y_pred_q90=("y_pred", lambda values: values.quantile(0.90)),
            mse=("squared_error", "mean"),
            mae=("abs_error", "mean"),
        )
        .reset_index()
    )
    years = grouped["year"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), gridspec_kw={"height_ratios": [2.2, 1.2]})
    ax = axes[0]
    ax.fill_between(years, grouped["y_pred_q10"].to_numpy(dtype=float), grouped["y_pred_q90"].to_numpy(dtype=float), color=COLORS["blue"], alpha=0.16, label="预测10-90分位")
    ax.plot(years, grouped["y_true"].to_numpy(dtype=float), color=COLORS["red"], marker="o", linewidth=2.2, label="真实值")
    ax.plot(years, grouped["y_pred_mean"].to_numpy(dtype=float), color=COLORS["blue"], marker="o", linewidth=2.2, label="平均预测")
    ax.set_title("测试期真实值与CMDL平均预测", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("标准化目标值")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(frameon=False)
    ax2 = axes[1]
    width = 0.34
    ax2.bar(years - width / 2, grouped["mse"], width=width, color=COLORS["gold"], label="MSE")
    ax2.bar(years + width / 2, grouped["mae"], width=width, color=COLORS["teal"], label="MAE")
    ax2.set_title("逐年预测误差", loc="left", fontsize=12, fontweight="bold")
    ax2.set_ylabel("误差")
    ax2.set_xticks(years)
    ax2.set_xticklabels([str(int(year)) for year in years])
    ax2.grid(axis="y", color="#e5e7eb")
    ax2.legend(frameon=False)
    save_figure(
        fig,
        "03_test_year_forecast_trajectory.png",
        "测试年真实值与预测轨迹",
        "展示2021-2023年真实值、平均预测和预测分散度，并标出逐年MSE/MAE。",
        manifest,
    )
    return grouped


def plot_baseline_comparison(baselines: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    order = baselines.sort_values("test_mse_mean", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    y = np.arange(len(order))
    colors = [COLORS["red"] if model == "CMDL" else COLORS["gray"] for model in order["model"]]
    axes[0].barh(y, order["test_mse_mean"], color=colors, alpha=0.9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order["model"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Test MSE")
    axes[0].set_title("MSE对比", loc="left", fontsize=12, fontweight="bold")
    axes[1].barh(y, order["test_mae_mean"], color=colors, alpha=0.9)
    axes[1].set_xlabel("Test MAE")
    axes[1].set_title("MAE对比", loc="left", fontsize=12, fontweight="bold")
    for ax, column in zip(axes, ["test_mse_mean", "test_mae_mean"]):
        ax.grid(axis="x", color="#e5e7eb")
        for idx, value in enumerate(order[column]):
            ax.text(value, idx, f" {value:.4f}", va="center", fontsize=8)
    fig.suptitle("CMDL与内部基线的测试误差对比", x=0.02, y=1.02, ha="left", fontsize=13, fontweight="bold")
    save_figure(
        fig,
        "04_baseline_comparison.png",
        "CMDL与内部基线对比",
        "比较CMDL、训练均值、persistence、Panel OLS和Grouped ARDL的测试误差。",
        manifest,
    )


def plot_region_error(entity_mean: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    ordered = entity_mean.sort_values("entity_test_mse_mean", ascending=True).reset_index(drop=True)
    labels = ordered["entity_code"] + " " + ordered["entity_name"]
    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    y = np.arange(len(ordered))
    bars = ax.barh(y, ordered["entity_test_mse_mean"], color=COLORS["blue"], alpha=0.86)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Seed平均实体MSE")
    ax.set_title("区域层面的预测误差集中度", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb")
    for bar, value in zip(bars, ordered["entity_test_mse_mean"]):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.2f}", va="center", fontsize=7)
    save_figure(
        fig,
        "05_region_error_concentration.png",
        "区域层面的预测误差集中度",
        "按seed平均实体MSE排序，显示哪些区域的预测残差最大。",
        manifest,
    )


def plot_omega_composition(entity_mean: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    ordered = entity_mean.sort_values("proxy_income_level_true", ascending=False).reset_index(drop=True)
    labels = ordered["entity_code"] + " " + ordered["entity_name"]
    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    y = np.arange(len(ordered))
    left = np.zeros(len(ordered))
    for column, label, color in [
        ("omega_1_mean", "lag 1", COLORS["blue"]),
        ("omega_2_mean", "lag 2", COLORS["gold"]),
        ("omega_3_mean", "lag 3", COLORS["green"]),
    ]:
        values = ordered[column].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=color, label=label, alpha=0.9)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Seed平均omega权重")
    ax.set_title("按收入proxy排序的lag-gate omega组成", loc="left", fontsize=13, fontweight="bold")
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.11))
    ax.grid(axis="x", color="#e5e7eb")
    save_figure(
        fig,
        "06_omega_composition_by_region.png",
        "区域lag-gate omega组成",
        "以seed平均omega权重展示各区域lag 1/2/3的组成，避免只看单个best seed。",
        manifest,
    )


def plot_proxy_vs_kstar(entity_mean: pd.DataFrame, runs: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    rho_mean = runs["test_kstar_proxy_spearman_adjusted_rho"].astype(float).mean()
    rho_std = runs["test_kstar_proxy_spearman_adjusted_rho"].astype(float).std(ddof=1)
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.errorbar(
        entity_mean["proxy_income_level_true"],
        entity_mean["k_star_mean"],
        yerr=entity_mean["k_star_std"],
        fmt="o",
        color=COLORS["blue"],
        ecolor="#9ca3af",
        elinewidth=0.9,
        capsize=2.5,
        alpha=0.86,
    )
    for _, row in entity_mean.iterrows():
        ax.text(row["proxy_income_level_true"], row["k_star_mean"], f" {row['entity_code']}", fontsize=7, alpha=0.75)
    ax.axhline(entity_mean["k_star_mean"].mean(), color=COLORS["gold"], linestyle="--", linewidth=1.2)
    ax.set_xlabel("income level proxy（标准化）")
    ax.set_ylabel("Seed平均有效滞后 k*")
    ax.set_title("收入proxy与有效滞后k*的关系诊断", loc="left", fontsize=13, fontweight="bold")
    ax.text(0.02, 0.98, f"seed层面调整Spearman rho均值={rho_mean:.4f}\nstd={rho_std:.4f}", transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d1d5db"})
    ax.grid(color="#e5e7eb")
    save_figure(
        fig,
        "07_proxy_vs_kstar.png",
        "收入proxy与k*关系诊断",
        "检验收入proxy是否稳定对应更长或更短的有效滞后。",
        manifest,
    )


def plot_mechanism_stability(runs: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    specs = [
        ("test_kstar_std", "实体间k*标准差", COLORS["blue"]),
        ("test_kstar_proxy_spearman_adjusted_rho", "proxy-k*调整rho", COLORS["purple"]),
        ("test_lag_gate_sensitivity_range", "lag gate敏感度范围", COLORS["green"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    rng = np.random.default_rng(20260510)
    for ax, (column, label, color) in zip(axes, specs):
        values = runs[column].astype(float).to_numpy()
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.boxplot(values, positions=[0], widths=0.26, patch_artist=True, boxprops={"facecolor": "#f3f4f6", "edgecolor": color}, medianprops={"color": COLORS["red"], "linewidth": 1.6}, whiskerprops={"color": color}, capprops={"color": color})
        ax.scatter(jitter, values, color=color, s=24, alpha=0.8)
        ax.axhline(0, color="#9ca3af", linewidth=0.9)
        ax.set_xticks([0])
        ax.set_xticklabels([label])
        ax.grid(axis="y", color="#e5e7eb")
    fig.suptitle("AC-GATE机制指标的seed稳定性", x=0.02, y=1.02, ha="left", fontsize=13, fontweight="bold")
    save_figure(
        fig,
        "08_mechanism_stability_across_seeds.png",
        "机制指标的seed稳定性",
        "展示k*变异、proxy-k*相关和lag-gate敏感度在20个seed上的不稳定性。",
        manifest,
    )


def plot_training_dynamics(history: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    grouped = (
        history.groupby("epoch")
        .agg(
            val_mse_median=("val_mse", "median"),
            val_mse_q25=("val_mse", lambda values: values.quantile(0.25)),
            val_mse_q75=("val_mse", lambda values: values.quantile(0.75)),
            train_task_median=("train_task_loss", "median"),
            seed_count=("seed", "nunique"),
        )
        .reset_index()
        .sort_values("epoch")
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    epochs = grouped["epoch"].to_numpy(dtype=float)
    ax.fill_between(epochs, grouped["val_mse_q25"].to_numpy(dtype=float), grouped["val_mse_q75"].to_numpy(dtype=float), color=COLORS["blue"], alpha=0.16, label="验证MSE IQR")
    ax.plot(epochs, grouped["val_mse_median"].to_numpy(dtype=float), color=COLORS["blue"], linewidth=2.0, label="验证MSE中位数")
    ax.plot(epochs, grouped["train_task_median"].to_numpy(dtype=float), color=COLORS["gold"], linewidth=1.8, label="训练task loss中位数")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / MSE（log尺度）")
    ax.set_title("跨seed训练动态", loc="left", fontsize=13, fontweight="bold")
    ax.grid(color="#e5e7eb", which="both")
    ax.legend(frameon=False)
    save_figure(
        fig,
        "09_training_dynamics.png",
        "跨seed训练动态",
        "展示训练task loss与验证MSE的中位数和验证MSE四分位范围。",
        manifest,
    )


def plot_proxy_reconstruction(entity_seed: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    proxies = [
        ("proxy_income_level_true", "proxy_income_level_pred", "收入水平proxy"),
        ("proxy_income_recent_level_true", "proxy_income_recent_level_pred", "近期收入水平proxy"),
        ("proxy_income_growth_signal_true", "proxy_income_growth_signal_pred", "收入增长信号proxy"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (true_col, pred_col, title) in zip(axes, proxies):
        x = entity_seed[true_col].astype(float)
        y = entity_seed[pred_col].astype(float)
        ax.scatter(x, y, color=COLORS["blue"], alpha=0.42, s=22)
        lower = min(x.min(), y.min())
        upper = max(x.max(), y.max())
        ax.plot([lower, upper], [lower, upper], color=COLORS["red"], linestyle="--", linewidth=1.0)
        corr = x.corr(y)
        ax.set_title(f"{title}\nr={corr:.3f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("真实proxy")
        ax.set_ylabel("重构proxy")
        ax.grid(color="#e5e7eb")
    fig.suptitle("proxy重构诊断", x=0.02, y=1.02, ha="left", fontsize=13, fontweight="bold")
    save_figure(
        fig,
        "10_proxy_reconstruction_diagnostic.png",
        "proxy重构诊断",
        "检查AC编码器/重构头对三个income proxy的重构质量。",
        manifest,
    )


def plot_main_bundle(year_stats: pd.DataFrame, baselines: pd.DataFrame, entity_mean: pd.DataFrame, manifest: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax = axes[0, 0]
    timeline_spans = [
        ("训练统计", 2006, 2018, COLORS["blue"]),
        ("验证", 2016, 2020, COLORS["gold"]),
        ("有效测试", 2021, 2023, COLORS["red"]),
    ]
    for idx, (label, start, end, color) in enumerate(timeline_spans):
        y = len(timeline_spans) - idx - 1
        ax.broken_barh([(start, end - start + 1)], (y - 0.32, 0.64), facecolors=color, alpha=0.9)
        ax.text((start + end) / 2 + 0.5, y, label, ha="center", va="center", color="white", fontsize=9)
    ax.set_xlim(2005.5, 2024.5)
    ax.set_yticks([])
    ax.set_title("A. 数据窗口", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb")

    ax = axes[0, 1]
    years = year_stats["year"].to_numpy(dtype=float)
    ax.plot(years, year_stats["y_true"].to_numpy(dtype=float), marker="o", color=COLORS["red"], linewidth=2, label="真实值")
    ax.plot(years, year_stats["y_pred_mean"].to_numpy(dtype=float), marker="o", color=COLORS["blue"], linewidth=2, label="平均预测")
    ax.fill_between(years, year_stats["y_pred_q10"].to_numpy(dtype=float), year_stats["y_pred_q90"].to_numpy(dtype=float), color=COLORS["blue"], alpha=0.16)
    ax.set_title("B. 测试年预测轨迹", loc="left", fontweight="bold")
    ax.set_xticks(years)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e5e7eb")

    ax = axes[1, 0]
    baseline_order = baselines.sort_values("test_mse_mean", ascending=True)
    colors = [COLORS["red"] if model == "CMDL" else COLORS["gray"] for model in baseline_order["model"]]
    ax.barh(np.arange(len(baseline_order)), baseline_order["test_mse_mean"], color=colors)
    ax.set_yticks(np.arange(len(baseline_order)))
    ax.set_yticklabels(baseline_order["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Test MSE")
    ax.set_title("C. 内部基线对比", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb")

    ax = axes[1, 1]
    omega = entity_mean.sort_values("proxy_income_level_true", ascending=False).head(10).reset_index(drop=True)
    labels = omega["entity_code"]
    y = np.arange(len(omega))
    left = np.zeros(len(omega))
    for column, label, color in [
        ("omega_1_mean", "lag 1", COLORS["blue"]),
        ("omega_2_mean", "lag 2", COLORS["gold"]),
        ("omega_3_mean", "lag 3", COLORS["green"]),
    ]:
        values = omega[column].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=color, label=label)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("D. 高收入proxy区域omega", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="x", color="#e5e7eb")
    fig.suptitle("RQ主结果可视化组合：fullspan_income_k3 + CMDL", x=0.02, y=1.01, ha="left", fontsize=14, fontweight="bold")
    save_figure(
        fig,
        "11_rq_main_figure_bundle.png",
        "RQ主结果四联图",
        "将数据窗口、预测轨迹、内部基线和omega诊断合并为适合论文RQ段落的四联图。",
        manifest,
    )


def write_report(manifest: list[dict[str, str]], runs: pd.DataFrame, year_stats: pd.DataFrame, baselines: pd.DataFrame, entity_mean: pd.DataFrame) -> None:
    mse_mean = runs["test_mse"].astype(float).mean()
    mse_std = runs["test_mse"].astype(float).std(ddof=1)
    mae_mean = runs["test_mae"].astype(float).mean()
    rho_mean = runs["test_kstar_proxy_spearman_adjusted_rho"].astype(float).mean()
    rho_std = runs["test_kstar_proxy_spearman_adjusted_rho"].astype(float).std(ddof=1)
    kstar_std = runs["test_kstar_std"].astype(float).mean()
    cmdl_row = baselines[baselines["model"] == "CMDL"].iloc[0]
    persistence_row = baselines[baselines["model"] == "Persistence"].iloc[0]
    grouped_ardl_row = baselines[baselines["model"] == "Grouped ARDL"].iloc[0]
    highest_error = entity_mean.sort_values("entity_test_mse_mean", ascending=False).head(5)

    lines = [
        "# fullspan_income_k3 + CMDL 可视化解释",
        "",
        "本报告基于 `RQ_res/fullspan_income_k3_cmdl/data` 中的聚合数据自动生成，所有说明均服务于 RQ 主方案的审慎表述。",
        "",
        "## 总体读法",
        "",
        f"- 该方案包含 20 个 seed，平均 test MSE 为 `{mse_mean:.4f}`，标准差为 `{mse_std:.4f}`，平均 test MAE 为 `{mae_mean:.4f}`。",
        "- 有效预测年份是 `2021-2023`，因为 `K=3` 的滞后结构会消耗配置测试窗口前几年的观测。",
        "- 目标变量在同一年内基本不随区域变化，因此区域图展示的是模型响应、误差和 lag gate 差异，不应解释为区域真实目标差异。",
        f"- 机制证据偏弱：平均 `test_kstar_std={kstar_std:.4f}`，proxy-`k*` 调整 Spearman rho 均值 `{rho_mean:.4f}`，跨 seed 标准差 `{rho_std:.4f}`。",
        f"- CMDL 的 MSE `{cmdl_row['test_mse_mean']:.4f}` 低于训练均值基线，但高于 persistence `{persistence_row['test_mse_mean']:.4f}` 和 grouped ARDL `{grouped_ardl_row['test_mse_mean']:.4f}`。",
        "",
    ]

    figure_explanations = {
        "01_data_window_timeline.png": [
            "这张图说明 full-span 方案如何扩大有效样本：完整面板覆盖 2006-2023，训练统计窗口到 2018，验证窗口到 2020。由于 `K=3`，真正进入预测表的是 2021-2023。",
            "RQ表述中应强调：本方案的主要改进首先来自更长的可用历史窗口，而不是单纯来自模型结构变化。",
        ],
        "02_seed_forecast_robustness.png": [
            f"20 个 seed 的 MSE 分布集中在 `{runs['test_mse'].min():.4f}` 到 `{runs['test_mse'].max():.4f}` 之间，均值 `{mse_mean:.4f}`。这说明该方案比 best-seed 叙事更稳健。",
            "但箱线图也显示 seed 之间仍有可见差异，因此论文中应报告均值和波动，而不是只引用最低误差。",
        ],
        "03_test_year_forecast_trajectory.png": [
            "真实值从 2021 到 2023 持续上升，而模型平均预测也上升但幅度不足，因此误差逐年扩大。",
            "这张图适合解释 CMDL 的主要预测失败模式：不是完全无趋势，而是对测试期上升幅度估计偏保守。",
        ],
        "04_baseline_comparison.png": [
            "CMDL 明显优于 train-mean baseline，但不优于 persistence、Panel OLS 和 Grouped ARDL。",
            "这张图应放在 RQ 结果中约束结论：fullspan CMDL 是可报告的 AC-GATE 主设定，但不能宣称其为预测最优模型。",
        ],
        "05_region_error_concentration.png": [
            "该图显示区域层面的 seed 平均误差差异。误差最高的区域包括 " + ", ".join(highest_error["entity_name"].astype(str).tolist()) + "。",
            "由于真实目标对区域退化，区域误差应解释为模型对区域 proxy 的预测响应差异，而不是区域真实 outcome 的差异。",
        ],
        "06_omega_composition_by_region.png": [
            "该图展示每个区域在 20 个 seed 平均后的 lag 1/2/3 权重组成。整体上 lag 1 和 lag 2 占比较高，lag 3 权重较低。",
            "它比单一 omega heatmap 更适合当前结果，因为 seed 间 lag peak 不稳定，seed 平均能避免过度解释某一个 seed 的模式。",
        ],
        "07_proxy_vs_kstar.png": [
            f"收入 proxy 与有效滞后 `k*` 没有稳定单调关系：seed层面的 proxy-`k*` 调整 Spearman rho 均值为 `{rho_mean:.4f}`，且波动很大。",
            "这张图是机制解释的关键限制证据：可以说 lag gate 产生了可诊断的异质性，但不宜说它稳定学习到了收入条件下的滞后机制。",
        ],
        "08_mechanism_stability_across_seeds.png": [
            "三个机制指标都显示 seed 敏感性：`k*` 的实体间变异偏低，proxy-`k*` 相关在正负之间摆动，lag-gate sensitivity 也不稳定。",
            "这支持把 AC-GATE 机制结果写成 exploratory diagnostic，而不是 confirmatory mechanism evidence。",
        ],
        "09_training_dynamics.png": [
            "训练 task loss 通常下降，而验证 MSE 在 early stopping 前后出现明显变化。log尺度使早期大误差和后期收敛同时可见。",
            "该图适合作为补充材料，说明训练过程并非完全失控，但泛化误差仍受测试期目标上升影响。",
        ],
        "10_proxy_reconstruction_diagnostic.png": [
            "三个 proxy 的重构散点显示 AC encoder/reconstruction 只部分恢复 proxy 空间，尤其增长信号通常更难重构。",
            "这解释了为什么 proxy-conditioned lag 机制较弱：如果 proxy 表征本身不够稳定，后续 lag gate 很难形成强关系。",
        ],
        "11_rq_main_figure_bundle.png": [
            "四联图把 RQ 主叙事压缩到一张图：样本窗口、测试期预测、内部基线对比、lag-gate 诊断。",
            "建议作为主文候选图，但 caption 要明确：该图支持 full-span RQ 方案的可报告性，不支持 AC-GATE 预测最优或强机制结论。",
        ],
    }

    lines.extend(["## 图逐项解释", ""])
    for item in manifest:
        filename = item["filename"]
        title = item["title"]
        lines.extend(
            [
                f"### {title}",
                "",
                f"![{title}](figures/{filename})",
                "",
                f"- 文件：`figures/{filename}`",
                f"- 图意：{item['description']}",
            ]
        )
        for explanation in figure_explanations.get(filename, []):
            lines.append(f"- 解释：{explanation}")
        lines.append("")

    lines.extend(
        [
            "## 建议用于论文/RQ的表述",
            "",
            "可以写：full-span income/RPCYD CMDL 设定显著改善了 RQ 实验的样本覆盖和运行稳定性；在 20 个 seed 上，预测误差分布较稳定，并能产生可诊断的 lag-gate 权重。",
            "",
            "需要避免写：AC-GATE 明确优于所有基线，或收入 proxy 稳定决定了有效滞后。当前可视化显示 persistence 与 grouped ARDL 更强，且 proxy-`k*` 机制关系不稳定。",
            "",
        ]
    )
    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    LEGACY_REPORT_PATH.write_text(report_text, encoding="utf-8")


def main() -> None:
    configure_plot_style()
    data = load_data()
    manifest: list[dict[str, str]] = []
    plot_timeline(manifest)
    plot_seed_robustness(data["runs"], manifest)
    year_stats = plot_forecast_trajectory(data["predictions"], manifest)
    plot_baseline_comparison(data["baselines"], manifest)
    plot_region_error(data["entity_mean"], manifest)
    plot_omega_composition(data["entity_mean"], manifest)
    plot_proxy_vs_kstar(data["entity_mean"], data["runs"], manifest)
    plot_mechanism_stability(data["runs"], manifest)
    plot_training_dynamics(data["history"], manifest)
    plot_proxy_reconstruction(data["entity_seed"], manifest)
    plot_main_bundle(year_stats, data["baselines"], data["entity_mean"], manifest)
    write_report(manifest, data["runs"], year_stats, data["baselines"], data["entity_mean"])
    (FIGURE_DIR / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"figures={len(manifest)}")
    print(f"figure_dir={FIGURE_DIR}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()