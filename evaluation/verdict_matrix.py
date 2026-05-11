"""Data-driven layered verdict matrix utilities.

The rules mirror the L0-L3 audit protocol in paper_draft/main.tex:
forecast calibration, non-degenerate lag discovery, structured heterogeneity,
and optional ground-truth alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


DOMAIN_ORDER = ["synthetic", "economics", "energy"]
DOMAIN_LABEL = {
    "synthetic": "Synthetic",
    "economics": "Economics",
    "energy": "Energy",
}
LAYER_ORDER = ["L0", "L1", "L2", "L3"]
LAYER_NAME = {
    "L0": "forecast",
    "L1": "nondegenerate_lag",
    "L2": "structured_heterogeneity",
    "L3": "ground_truth_alignment",
}
SYNTHETIC_COMPETITORS = ["Plain LSTM", "TFT", "GA-Net"]
STATUS_LABEL = {
    "certified": "yes",
    "not_certified": "n/c",
    "ruled_out": "no",
    "not_claimed": "n/a",
}
STATUS_COLOR = {
    "certified": "#C8F7CF",
    "not_certified": "#FFD7A8",
    "ruled_out": "#F4A6A6",
    "not_claimed": "#DADDE3",
}
STATUS_TEXT = {
    "certified": "#0B3D19",
    "not_certified": "#4B2B00",
    "ruled_out": "#5B0000",
    "not_claimed": "#2F343B",
}


@dataclass(frozen=True)
class VerdictConfig:
    alpha: float = 0.05
    l1_epsilon: float = 1e-6
    l2_min_rejection_share: float = 0.0
    l3_synthetic_rho_threshold: float = 0.8
    l3_synthetic_positive_share: float = 0.8


def _numeric(series: pd.Series | Any) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(pd.Series([series]), errors="coerce")


def _fmt(value: Any, digits: int = 3) -> str:
    numeric = _numeric(value).iloc[0]
    if pd.isna(numeric):
        return "nan"
    return f"{float(numeric):.{digits}g}"


def _layer_code(layer: Any) -> str:
    text = str(layer)
    for code in LAYER_ORDER:
        if text == code or text.startswith(f"{code}_"):
            return code
    return text


def _row(
    domain: str,
    layer: str,
    verdict: str,
    evidence: str,
    **metrics: Any,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "domain_label": DOMAIN_LABEL.get(domain, domain.title()),
        "layer": layer,
        "layer_name": LAYER_NAME.get(layer, layer),
        "verdict": verdict,
        "label": STATUS_LABEL.get(verdict, "n/c"),
        "evidence": evidence,
        **metrics,
    }


def _method_rows(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    if frame.empty or "display_name" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["display_name"].astype(str) == method].copy()


def _missing_methods(frame: pd.DataFrame, methods: list[str], column: str = "method") -> list[str]:
    if frame.empty or column not in frame.columns:
        return methods.copy()
    present = set(frame[column].dropna().astype(str).unique().tolist())
    return [method for method in methods if method not in present]


def _synthetic_l0(
    summary: pd.DataFrame,
    significance_task: pd.DataFrame,
    config: VerdictConfig,
) -> dict[str, Any]:
    task_tests = significance_task.loc[
        significance_task.get("method", pd.Series(dtype=str)).astype(str).isin(SYNTHETIC_COMPETITORS)
    ].copy()
    if not task_tests.empty and {"reference_better_mean", "wilcoxon_p"}.issubset(task_tests.columns):
        missing = _missing_methods(task_tests, SYNTHETIC_COMPETITORS)
        passed = bool(
            not missing
            and (task_tests["reference_better_mean"].astype(bool)
            & (_numeric(task_tests["wilcoxon_p"]) < config.alpha)).all()
        )
        max_p = _numeric(task_tests["wilcoxon_p"]).max()
        tested = "/".join(method for method in SYNTHETIC_COMPETITORS if method not in missing)
        missing_text = f" missing={','.join(missing)}" if missing else ""
        return _row(
            "synthetic",
            "L0",
            "certified" if passed else "not_certified",
            f"Paired task-loss tests versus {tested}; max p={_fmt(max_p)}.{missing_text}",
            task_test_max_p=max_p,
            task_missing_competitors=",".join(missing),
        )

    cmdl = _method_rows(summary, "CMDL")
    if cmdl.empty or "task_loss_mean" not in summary.columns:
        return _row("synthetic", "L0", "not_certified", "Missing synthetic task-loss comparison rows.")
    missing = [method for method in SYNTHETIC_COMPETITORS if _method_rows(summary, method).empty]
    comparisons = []
    for method in SYNTHETIC_COMPETITORS:
        competitor = _method_rows(summary, method)
        if competitor.empty:
            continue
        merged = cmdl[["scenario", "task_loss_mean"]].merge(
            competitor[["scenario", "task_loss_mean"]],
            on="scenario",
            suffixes=("_cmdl", "_competitor"),
        )
        comparisons.extend((_numeric(merged["task_loss_mean_cmdl"]) < _numeric(merged["task_loss_mean_competitor"])).tolist())
    passed = bool(not missing and comparisons and all(comparisons))
    return _row(
        "synthetic",
        "L0",
        "certified" if passed else "not_certified",
        f"CMDL task loss is lower than all synthetic competitors in {sum(bool(value) for value in comparisons)}/{len(comparisons)} scenario-method comparisons.",
        task_missing_competitors=",".join(missing),
    )


def _synthetic_l1(summary: pd.DataFrame, config: VerdictConfig) -> dict[str, Any]:
    cmdl = _method_rows(summary, "CMDL")
    controls = summary.loc[summary.get("display_name", pd.Series(dtype=str)).astype(str).isin(["No AC Encoder", "Uniform Lag"])].copy()
    if "kstar_std_mean" in summary.columns:
        cmdl_std = _numeric(cmdl.get("kstar_std_mean", pd.Series(dtype=float))).dropna()
        control_std = _numeric(controls.get("kstar_std_mean", pd.Series(dtype=float))).dropna()
        passed = bool(not cmdl_std.empty and (cmdl_std > config.l1_epsilon).all() and not control_std.empty and (control_std <= config.l1_epsilon).all())
        return _row(
            "synthetic",
            "L1",
            "certified" if passed else "not_certified",
            f"CMDL k* std is nonzero and collapse controls are at epsilon={config.l1_epsilon:g}.",
            cmdl_min_kstar_std=float(cmdl_std.min()) if not cmdl_std.empty else float("nan"),
        )

    cmdl_share = _numeric(cmdl.get("kstar_positive_seed_share", pd.Series(dtype=float))).dropna()
    control_share = _numeric(controls.get("kstar_positive_seed_share", pd.Series(dtype=float))).dropna()
    passed = bool(not cmdl_share.empty and (cmdl_share > 0.0).all() and not control_share.empty and (control_share <= config.l1_epsilon).all())
    return _row(
        "synthetic",
        "L1",
        "certified" if passed else "not_certified",
        "CMDL retains nonzero seed-level lag-rank signal while No AC Encoder/Uniform Lag collapse.",
        cmdl_min_positive_share=float(cmdl_share.min()) if not cmdl_share.empty else float("nan"),
        control_max_positive_share=float(control_share.max()) if not control_share.empty else float("nan"),
    )


def _synthetic_l2(
    summary: pd.DataFrame,
    significance_kstar: pd.DataFrame,
    config: VerdictConfig,
) -> dict[str, Any]:
    kstar_tests = significance_kstar.loc[
        significance_kstar.get("method", pd.Series(dtype=str)).astype(str).isin(SYNTHETIC_COMPETITORS)
    ].copy()
    controls = summary.loc[summary.get("display_name", pd.Series(dtype=str)).astype(str).isin(["No AC Encoder", "Uniform Lag"])].copy()
    control_signal = _numeric(controls.get("effective_kstar_spearman_rho_mean", pd.Series(dtype=float))).abs().dropna()
    controls_collapsed = bool(not control_signal.empty and (control_signal <= config.l1_epsilon).all())

    if not kstar_tests.empty and {"reference_better_mean", "wilcoxon_p"}.issubset(kstar_tests.columns):
        missing = _missing_methods(kstar_tests, SYNTHETIC_COMPETITORS)
        test_passed = bool(
            not missing
            and (kstar_tests["reference_better_mean"].astype(bool)
            & (_numeric(kstar_tests["wilcoxon_p"]) < config.alpha)).all()
        )
        max_p = _numeric(kstar_tests["wilcoxon_p"]).max()
        passed = bool(test_passed and controls_collapsed)
        tested = "/".join(method for method in SYNTHETIC_COMPETITORS if method not in missing)
        missing_text = f" missing={','.join(missing)}" if missing else ""
        return _row(
            "synthetic",
            "L2",
            "certified" if passed else "not_certified",
            f"CMDL k* recovery beats {tested} and collapse controls have no rank signal; max p={_fmt(max_p)}.{missing_text}",
            kstar_test_max_p=max_p,
            controls_collapsed=controls_collapsed,
            kstar_missing_competitors=",".join(missing),
        )

    return _row(
        "synthetic",
        "L2",
        "certified" if controls_collapsed else "not_certified",
        "Synthetic L2 falls back to ablation-control collapse because paired k* tests are unavailable.",
        controls_collapsed=controls_collapsed,
    )


def _synthetic_l3(summary: pd.DataFrame, config: VerdictConfig) -> dict[str, Any]:
    cmdl = _method_rows(summary, "CMDL")
    rho = _numeric(cmdl.get("effective_kstar_spearman_rho_mean", pd.Series(dtype=float))).dropna()
    share = _numeric(cmdl.get("kstar_positive_seed_share", pd.Series(dtype=float))).dropna()
    passed = bool(
        not rho.empty
        and not share.empty
        and (rho >= config.l3_synthetic_rho_threshold).all()
        and (share >= config.l3_synthetic_positive_share).all()
    )
    return _row(
        "synthetic",
        "L3",
        "certified" if passed else "not_certified",
        f"Known-lag alignment requires Spearman rho>={config.l3_synthetic_rho_threshold:g} and positive-seed share>={config.l3_synthetic_positive_share:g}.",
        cmdl_min_rho=float(rho.min()) if not rho.empty else float("nan"),
        cmdl_min_positive_share=float(share.min()) if not share.empty else float("nan"),
    )


def _realdata_l0(domain: str, compact: pd.DataFrame) -> dict[str, Any]:
    cmdl = _method_rows(compact, "CMDL")
    if cmdl.empty:
        return _row(domain, "L0", "not_certified", "Missing CMDL real-data forecast row.")
    cmdl_r2 = _numeric(cmdl.iloc[0].get("test_r2_mean")).iloc[0]
    competitor_r2 = _numeric(compact.loc[compact["display_name"].astype(str) != "CMDL", "test_r2_mean"]).dropna()
    if competitor_r2.empty or pd.isna(cmdl_r2):
        verdict = "not_certified"
        best = float("nan")
    else:
        best = float(competitor_r2.max())
        verdict = "certified" if float(cmdl_r2) >= best else "not_certified"
    return _row(
        domain,
        "L0",
        verdict,
        f"CMDL mean test R2={_fmt(cmdl_r2)}; best competing mean test R2={_fmt(best)}.",
        cmdl_test_r2_mean=cmdl_r2,
        best_competing_test_r2_mean=best,
    )


def _realdata_l1(domain: str, compact: pd.DataFrame, degeneracy: pd.DataFrame, config: VerdictConfig) -> dict[str, Any]:
    source = degeneracy.loc[degeneracy.get("domain", pd.Series(dtype=str)).astype(str) == domain].copy() if not degeneracy.empty else pd.DataFrame()
    if not source.empty:
        method_col = "method"
        std_col = "kstar_std_mean"
    else:
        source = compact.copy()
        method_col = "display_name"
        std_col = "kstar_std_mean"

    if source.empty:
        return _row(domain, "L1", "not_certified", "Missing lag-degeneracy diagnostics.")

    cmdl_std = _numeric(source.loc[source[method_col].astype(str) == "CMDL", std_col]).dropna()
    controls = source.loc[source[method_col].astype(str).isin(["No AC Encoder", "Uniform Lag"])].copy()
    if "degenerate_control" in controls.columns:
        control_flags = controls["degenerate_control"]
        if control_flags.dtype != bool:
            control_flags = control_flags.astype(str).str.lower().isin(["true", "1", "yes"])
        controls_ok = bool(not control_flags.empty and control_flags.all())
    else:
        control_std = _numeric(controls.get(std_col, pd.Series(dtype=float))).dropna()
        controls_ok = bool(not control_std.empty and (control_std <= config.l1_epsilon).all())
    cmdl_ok = bool(not cmdl_std.empty and (cmdl_std > config.l1_epsilon).all())
    return _row(
        domain,
        "L1",
        "certified" if cmdl_ok and controls_ok else "not_certified",
        f"CMDL k* std must exceed epsilon={config.l1_epsilon:g}, while No AC Encoder/Uniform Lag are degenerate controls.",
        cmdl_kstar_std_mean=float(cmdl_std.iloc[0]) if not cmdl_std.empty else float("nan"),
        controls_degenerate=controls_ok,
    )


def _realdata_l2(domain: str, stratified: pd.DataFrame, config: VerdictConfig) -> dict[str, Any]:
    if stratified.empty:
        return _row(domain, "L2", "not_certified", "Missing stratified k* diagnostics.")
    cmdl = stratified.loc[stratified["method"].astype(str) == "CMDL"].copy()
    controls = stratified.loc[stratified["method"].astype(str).isin(["No AC Encoder", "Uniform Lag"])].copy()
    if cmdl.empty:
        return _row(domain, "L2", "not_certified", "Missing CMDL stratified rows.")

    valid = _numeric(cmdl.get("n_seeds_valid", pd.Series(dtype=float))) > 0
    fisher = _numeric(cmdl.get("fisher_combined_p", pd.Series(dtype=float))) < config.alpha
    if "share_seeds_p_lt_05" in cmdl.columns:
        rejection = _numeric(cmdl["share_seeds_p_lt_05"]) >= config.l2_min_rejection_share
    else:
        rejection = pd.Series([True] * len(cmdl), index=cmdl.index)
    controls_valid = _numeric(controls.get("n_seeds_valid", pd.Series(dtype=float))).fillna(0.0)
    controls_collapsed = bool(controls.empty or (controls_valid == 0.0).all())
    passed = bool(valid.all() and fisher.all() and rejection.all() and controls_collapsed)
    return _row(
        domain,
        "L2",
        "certified" if passed else "not_certified",
        f"CMDL stratifier rows require valid seeds, Fisher p<{config.alpha:g}, and degenerate controls.",
        cmdl_rows=int(len(cmdl)),
        cmdl_max_fisher_p=float(_numeric(cmdl.get("fisher_combined_p", pd.Series(dtype=float))).max()),
        controls_collapsed=controls_collapsed,
    )


def _realdata_l3(domain: str) -> dict[str, Any]:
    return _row(
        domain,
        "L3",
        "not_claimed",
        "No optional ground-truth lag is available for this real-data domain.",
    )


def build_verdict_matrix(
    synthetic_summary: pd.DataFrame,
    synthetic_significance_task: pd.DataFrame,
    synthetic_significance_kstar: pd.DataFrame,
    economics_compact: pd.DataFrame,
    economics_stratified: pd.DataFrame,
    energy_compact: pd.DataFrame,
    energy_stratified: pd.DataFrame,
    ablation_degeneracy: pd.DataFrame | None = None,
    config: VerdictConfig | None = None,
) -> pd.DataFrame:
    """Build a long-form Domain x L0-L3 verdict matrix from experiment summaries."""

    config = config or VerdictConfig()
    degeneracy = ablation_degeneracy if ablation_degeneracy is not None else pd.DataFrame()
    rows = [
        _synthetic_l0(synthetic_summary, synthetic_significance_task, config),
        _synthetic_l1(synthetic_summary, config),
        _synthetic_l2(synthetic_summary, synthetic_significance_kstar, config),
        _synthetic_l3(synthetic_summary, config),
    ]
    for domain, compact, stratified in (
        ("economics", economics_compact, economics_stratified),
        ("energy", energy_compact, energy_stratified),
    ):
        rows.extend(
            [
                _realdata_l0(domain, compact),
                _realdata_l1(domain, compact, degeneracy, config),
                _realdata_l2(domain, stratified, config),
                _realdata_l3(domain),
            ]
        )

    frame = pd.DataFrame(rows)
    frame["domain_order"] = frame["domain"].map({name: idx for idx, name in enumerate(DOMAIN_ORDER)})
    frame["layer_order"] = frame["layer"].map({name: idx for idx, name in enumerate(LAYER_ORDER)})
    frame = frame.sort_values(["domain_order", "layer_order"], na_position="last").drop(columns=["domain_order", "layer_order"])
    return frame.reset_index(drop=True)


def build_verdict_display_table(verdict: pd.DataFrame) -> pd.DataFrame:
    """Return a compact display table with text labels yes/n/c/no/n/a."""

    if verdict.empty:
        return pd.DataFrame(columns=["Domain", *LAYER_ORDER])
    frame = verdict.copy()
    frame["domain"] = frame["domain"].astype(str)
    frame["domain_label"] = frame.get("domain_label", frame["domain"].map(DOMAIN_LABEL)).fillna(frame["domain"])
    frame["layer_code"] = frame["layer"].map(_layer_code)
    if "label" in frame.columns:
        frame["label"] = frame["label"].fillna(frame["verdict"].map(STATUS_LABEL)).fillna("n/c")
    else:
        frame["label"] = frame["verdict"].map(STATUS_LABEL).fillna("n/c")
    rows: list[dict[str, str]] = []
    for domain in DOMAIN_ORDER:
        source = frame.loc[frame["domain"] == domain]
        row = {"Domain": DOMAIN_LABEL.get(domain, domain.title())}
        for layer in LAYER_ORDER:
            label = source.loc[source["layer_code"] == layer, "label"]
            row[layer] = str(label.iloc[0]) if not label.empty else "n/c"
        rows.append(row)
    return pd.DataFrame(rows, columns=["Domain", *LAYER_ORDER])


def verdict_status_frame(verdict: pd.DataFrame) -> pd.DataFrame:
    """Return compact domain/layer/status rows for downstream figure builders."""

    if verdict.empty:
        return pd.DataFrame(columns=["domain", "layer", "status"])
    frame = verdict.copy()
    frame["domain"] = frame["domain"].astype(str)
    frame["layer"] = frame["layer"].map(_layer_code)
    frame["status"] = frame["verdict"].astype(str)
    rows: list[dict[str, str]] = []
    for domain in DOMAIN_ORDER:
        for layer in LAYER_ORDER:
            status = frame.loc[(frame["domain"] == domain) & (frame["layer"] == layer), "status"]
            rows.append({"domain": domain, "layer": layer, "status": str(status.iloc[0]) if not status.empty else "not_certified"})
    return pd.DataFrame(rows)


def plot_verdict_matrix(verdict: pd.DataFrame, save_path: str | Path | None = None) -> plt.Figure:
    """Render the LaTeX-style verdict matrix as colored text pills."""

    status_rows = verdict_status_frame(verdict)
    fig, ax = plt.subplots(figsize=(5.25, 2.25))
    ax.set_axis_off()
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    ax.text(2.5, 4.70, "TABLE V", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(2.5, 4.30, "VERDICT MATRIX.", ha="center", va="center", fontsize=10, fontweight="bold", fontvariant="small-caps")
    ax.plot([0.05, 4.95], [3.92, 3.92], color="black", linewidth=0.9)
    ax.plot([0.05, 4.95], [3.15, 3.15], color="black", linewidth=0.55)

    x_positions = [0.40, 1.45, 2.45, 3.45, 4.45]
    y_header = 3.50
    y_positions = [2.65, 2.15, 1.65]
    headers = ["Domain", *LAYER_ORDER]
    for x, header in zip(x_positions, headers):
        ha = "left" if header == "Domain" else "center"
        ax.text(x, y_header, header, ha=ha, va="center", fontsize=9, fontweight="bold")

    for domain, y in zip(DOMAIN_ORDER, y_positions):
        ax.text(0.15, y, DOMAIN_LABEL[domain], ha="left", va="center", fontsize=9, fontweight="bold")
        for x, layer in zip(x_positions[1:], LAYER_ORDER):
            status = status_rows.loc[(status_rows["domain"] == domain) & (status_rows["layer"] == layer), "status"]
            status_value = str(status.iloc[0]) if not status.empty else "not_certified"
            label = STATUS_LABEL.get(status_value, "n/c")
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=STATUS_TEXT.get(status_value, "#2F343B"),
                bbox={
                    "boxstyle": "round,pad=0.24,rounding_size=0.28",
                    "facecolor": STATUS_COLOR.get(status_value, STATUS_COLOR["not_certified"]),
                    "edgecolor": "none",
                },
            )

    ax.plot([0.05, 4.95], [1.25, 1.25], color="black", linewidth=0.9)
    ax.text(
        0.05,
        0.70,
        "Green denotes supported, amber not certified, red ruled out, and gray not claimed.",
        ha="left",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        wrap=True,
    )
    fig.tight_layout(pad=0.35)
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=300 if target.suffix.lower() == ".png" else None, bbox_inches="tight")
    return fig


__all__ = [
    "DOMAIN_ORDER",
    "LAYER_ORDER",
    "STATUS_LABEL",
    "VerdictConfig",
    "build_verdict_display_table",
    "build_verdict_matrix",
    "plot_verdict_matrix",
    "verdict_status_frame",
]
