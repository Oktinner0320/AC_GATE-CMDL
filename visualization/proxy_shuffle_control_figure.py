"""Render a single-column proxy-shuffle negative-control figure.

The figure summarizes the formal 20-seed proxy-shuffle experiment:
L2 stratifier alignment collapses after shuffling entity-level proxies,
while test R^2 stays essentially unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]

STRATIFIER_LABEL = {
    "hc_mean_train": "HC",
    "log_gdp_per_worker_train": "GDP/wkr",
    "log_capital_per_worker_train": "Capital/wkr",
    "rule_of_law_train": "Rule of law",
    "government_effectiveness_train": "Gov. eff.",
    "log_gdp_per_capita_train": "GDP/cap",
}
DOMAIN_LABEL = {"economics": "Econ.", "energy": "Energy"}
DOMAIN_COLOR = {"economics": "#2F6DA8", "energy": "#D0643C"}
GRID_COLOR = "#E3E8EF"


def _default_proxy_root() -> Path:
    roots = sorted(p for p in (REPO / "outputs" / "negative_controls").glob("proxy_shuffle_20seed_*") if p.is_dir())
    if not roots:
        raise FileNotFoundError("No proxy_shuffle_20seed_* directory found under outputs/negative_controls")
    return roots[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build proxy-shuffle negative-control figure.")
    parser.add_argument("--proxy-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--formats", nargs="+", default=["png", "svg"])
    parser.add_argument("--copy-to-paper-img", action="store_true")
    return parser.parse_args()


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.7,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.grid": False,
        }
    )


def _load_summary(proxy_root: Path) -> pd.DataFrame:
    path = proxy_root / "comparison" / "proxy_shuffle_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    summary = pd.read_csv(path)
    summary["stratifier_label"] = summary["stratifier"].map(STRATIFIER_LABEL).fillna(summary["stratifier"])
    summary["domain_label"] = summary["domain"].map(DOMAIN_LABEL).fillna(summary["domain"])
    return summary


def _domain_rows(summary: pd.DataFrame, domain: str) -> pd.DataFrame:
    rows = summary[summary["domain"] == domain].copy()
    return rows.sort_values("original_abs_rho_mean", ascending=False).reset_index(drop=True)


def _draw_alignment_domain_panel(ax: plt.Axes, rows: pd.DataFrame, domain: str, *, show_section_title: bool, show_xlabel: bool) -> None:
    color = DOMAIN_COLOR[domain]
    y = np.arange(len(rows), dtype=float)
    label_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

    for idx, row in enumerate(rows.itertuples()):
        ax.plot(
            [row.proxy_shuffled_abs_rho_mean, row.original_abs_rho_mean],
            [idx, idx],
            color="#A8B6C7",
            linewidth=2.25,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            row.original_abs_rho_mean,
            idx,
            s=46,
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=3,
        )
        ax.scatter(
            row.proxy_shuffled_abs_rho_mean,
            idx,
            s=40,
            marker="D",
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            1.025,
            idx,
            f"{row.original_seed_p_lt_05_share:.0%}->{row.proxy_shuffled_seed_p_lt_05_share:.0%}",
            transform=label_transform,
            ha="left",
            va="center",
            fontsize=5.6,
            color=color,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(rows["stratifier_label"])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 0.78)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.2, labelbottom=show_xlabel)
    for sep in np.arange(len(rows) - 1) + 0.5:
        ax.axhline(sep, color=GRID_COLOR, linewidth=0.45, zorder=0)
    ax.axvline(0.0, color="#243B53", linewidth=0.65, zorder=0.5)
    ax.text(0.0, 1.04, "Economics" if domain == "economics" else "Energy", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.6, fontweight="semibold", color=color)
    if show_section_title:
        ax.set_title("A. Stratifier alignment collapses after proxy shuffle", loc="left", pad=20)
        ax.text(1.025, 1.04, r"seed $p<.05$", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.5, color="#4A5568")
    if show_xlabel:
        ax.set_xlabel(r"Mean absolute Spearman alignment, $|\rho|$", labelpad=0.5)
    else:
        ax.tick_params(axis="x", labelbottom=False)


def _draw_r2_panel(ax: plt.Axes, summary: pd.DataFrame) -> None:
    r2_rows = summary.drop_duplicates("domain").sort_values("domain")
    y_lookup = {"economics": 1.0, "energy": 0.0}
    for row in r2_rows.itertuples():
        y = y_lookup.get(row.domain, 0.0)
        delta_r2 = float(row.proxy_shuffled_test_r2_mean - row.original_test_r2_mean)
        color = DOMAIN_COLOR.get(row.domain, "#777777")
        ax.hlines(y, 0.0, delta_r2, color=color, linewidth=1.7, alpha=0.35, zorder=2)
        ax.scatter(delta_r2, y, s=42, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax.text(
            delta_r2 + 0.00016,
            y,
            f"{delta_r2:+.4f}",
            ha="left",
            va="center",
            fontsize=5.95,
            color=color,
        )

    ax.axvline(0.0, color="#243B53", linewidth=0.65, linestyle=(0, (3, 3)), zorder=1)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["Energy", "Economics"])
    ax.set_xlim(-0.0010, 0.0012)
    ax.set_xlabel(r"$\Delta R^2$ (shuffled - original)")
    ax.set_title(r"B. Forecast performance is effectively unchanged", loc="left", pad=2)
    ax.tick_params(axis="y", length=0)
    max_delta = float(r2_rows["delta_test_r2_mean"].abs().max())
    ax.text(
        0.02,
        0.92,
        rf"all $|\Delta R^2| < {max_delta:.4f}$",
        ha="left",
        va="top",
        fontsize=5.8,
        color="#4A5568",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.86),
    )
    ax.axhline(0.5, color=GRID_COLOR, linewidth=0.45, zorder=0)


def build_figure(proxy_root: Path, output_dir: Path, formats: list[str]) -> list[Path]:
    _setup_style()
    summary = _load_summary(proxy_root)

    fig = plt.figure(figsize=(3.52, 4.7))
    grid = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.18, 1.18, 1.0], hspace=0.50)
    ax_economics = fig.add_subplot(grid[0, 0])
    ax_energy = fig.add_subplot(grid[1, 0], sharex=ax_economics)
    ax_r2 = fig.add_subplot(grid[2, 0])

    _draw_alignment_domain_panel(ax_economics, _domain_rows(summary, "economics"), "economics", show_section_title=True, show_xlabel=False)
    _draw_alignment_domain_panel(ax_energy, _domain_rows(summary, "energy"), "energy", show_section_title=False, show_xlabel=True)
    _draw_r2_panel(ax_r2, summary)
    fig.subplots_adjust(left=0.33, right=0.86, top=0.95, bottom=0.08)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        path = output_dir / f"proxy_shuffle_l2_single_column.{fmt}"
        fig.savefig(path, dpi=340 if fmt == "png" else None, bbox_inches="tight", pad_inches=0.08)
        saved.append(path)
    plt.close(fig)
    return saved


def main() -> int:
    args = parse_args()
    proxy_root = args.proxy_root or _default_proxy_root()
    output_dir = args.output_dir or proxy_root / "figures"
    saved = build_figure(proxy_root, output_dir, args.formats)
    print("Generated:")
    for path in saved:
        print(" ", path)

    if args.copy_to_paper_img:
        target_dir = REPO / "paper_draft" / "img"
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in saved:
            dest = target_dir / path.name
            shutil.copy2(path, dest)
            print(" copied:", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())