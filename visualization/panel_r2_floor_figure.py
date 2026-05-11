"""Render the "R^2 floor" diagnostic figure.

Conveys that on real-world panel data every method (classical, RNN,
Transformer, ablations, and AC-GATE) collapses to a narrow band around
zero test R^2.  Used in the paper to reframe `R^2 ~ 0` as a property of
the regime rather than a weakness of any single model.

Outputs
-------
outputs/paper_assets/complete_20seed_20260426/compact_figures/panel_r2_floor.{png,svg}
paper_draft/img/panel_r2_floor.{png,svg}  (when --copy-to-paper-img)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
LOCKED_PLAN = "complete_20seed_20260426"
TFT_PLAN = "complete_20seed"

METHOD_ORDER = [
    "Grouped ARDL",
    "Plain LSTM",
    "TFT",
    "GA-Net",
    "No AC Encoder",
    "Uniform Lag",
    "No Recon Regularization",
    "CMDL",
]
METHOD_LABEL = {
    "Grouped ARDL": "ARDL",
    "Plain LSTM": "Plain LSTM",
    "TFT": "TFT",
    "GA-Net": "GA-Net",
    "No AC Encoder": "No-AC",
    "Uniform Lag": "Uniform-Lag",
    "No Recon Regularization": "No-Recon",
    "CMDL": "AC-GATE",
}
DOMAIN_STYLE = {
    "economics": {
        "label": "Economics",
        "marker": "o",
        "color": "#2563A6",
        "ci_color": "#1D4F82",
        "offset": -0.18,
    },
    "energy": {
        "label": "Energy",
        "marker": "D",
        "color": "#C65A2E",
        "ci_color": "#A94924",
        "offset": 0.18,
    },
}


def _load_locked_r2(domain: str) -> pd.DataFrame:
    path = REPO / "outputs" / f"notebook_{domain}" / LOCKED_PLAN / "comparison" / f"{domain}_comparison.csv"
    df = pd.read_csv(path)
    return df[["display_name", "test_r2"]].rename(columns={"display_name": "method", "test_r2": "r2"}).copy()


def _load_tft_r2(domain: str) -> pd.DataFrame:
    base = REPO / "outputs" / f"notebook_{domain}" / TFT_PLAN / "tft"
    rows = []
    if base.exists():
        for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            summary = run_dir / "summary.json"
            if not summary.exists():
                continue
            with summary.open() as h:
                data = json.load(h)
            r2 = (data.get("metrics", {}).get("test", {}) or {}).get("r2")
            if r2 is None:
                continue
            rows.append({"method": "TFT", "r2": float(r2)})
    return pd.DataFrame(rows)


def _load_ganet_r2(domain: str) -> pd.DataFrame:
    base = REPO / "outputs" / f"notebook_{domain}" / TFT_PLAN / "ganet"
    rows = []
    if base.exists():
        for run_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            summary = run_dir / "summary.json"
            if not summary.exists():
                continue
            with summary.open() as h:
                data = json.load(h)
            r2 = (data.get("metrics", {}).get("test", {}) or {}).get("r2")
            if r2 is None:
                continue
            rows.append({"method": "GA-Net", "r2": float(r2)})
    return pd.DataFrame(rows)


def _domain_frame(domain: str) -> pd.DataFrame:
    return pd.concat(
        [_load_locked_r2(domain), _load_tft_r2(domain), _load_ganet_r2(domain)],
        ignore_index=True,
    )


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "axes.grid": False,
        }
    )


def _draw_merged_panel(ax: plt.Axes, frames: dict[str, pd.DataFrame]) -> None:
    combined = pd.concat(frames.values(), ignore_index=True)
    methods_present = [m for m in METHOD_ORDER if m in combined["method"].unique()]
    row_lookup = {method: row_idx for row_idx, method in enumerate(methods_present)}
    rng = np.random.default_rng(7)
    x_min = min(-0.25, float(combined["r2"].min()) - 0.05)
    x_max = max(0.75, float(combined["r2"].max()) + 0.10)

    ax.axvspan(-0.05, 0.10, color="#EDF2F7", zorder=0)
    ax.axvline(0.0, color="#243B53", linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)

    for domain, frame in frames.items():
        style = DOMAIN_STYLE[domain]
        for method in methods_present:
            sub = frame.loc[frame["method"] == method, "r2"].dropna().to_numpy(dtype=float)
            if sub.size == 0:
                continue
            y = row_lookup[method] + float(style["offset"])
            y_jitter = rng.uniform(-0.055, 0.055, size=sub.size)
            ax.scatter(
                sub,
                np.full_like(sub, y, dtype=float) + y_jitter,
                s=13,
                marker=str(style["marker"]),
                color=str(style["color"]),
                alpha=0.42,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            mean = float(sub.mean())
            if sub.size > 1:
                se = float(sub.std(ddof=1) / np.sqrt(sub.size))
                ci_half_width = 1.96 * se
                ci_low = mean - ci_half_width
                ci_high = mean + ci_half_width
                ax.scatter(
                    mean,
                    y,
                    marker=str(style["marker"]),
                    s=62,
                    facecolor="none",
                    edgecolor=str(style["color"]),
                    linewidth=1.8,
                    zorder=5.2,
                )
                ax.hlines(y, ci_low, ci_high, color="white", linewidth=3.2, zorder=5.45)
                ax.hlines(y, ci_low, ci_high, color=str(style["ci_color"]), linewidth=1.55, zorder=5.6)
                ax.vlines(
                    [ci_low, ci_high],
                    y - 0.045,
                    y + 0.045,
                    color=str(style["ci_color"]),
                    linewidth=1.05,
                    zorder=5.6,
                )
            else:
                ax.plot(
                    mean,
                    y,
                    marker=str(style["marker"]),
                    color=str(style["color"]),
                    markerfacecolor="none",
                    markeredgecolor=str(style["color"]),
                    markeredgewidth=1.35,
                    markersize=6.0,
                    zorder=5,
                )

    ax.set_yticks(range(len(methods_present)))
    ax.set_yticklabels([METHOD_LABEL[m] for m in methods_present])
    ax.invert_yaxis()
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Test $R^2$ (20 seeds)")
    ax.tick_params(axis="x", length=2.5)
    ax.tick_params(axis="y", length=0)
    for y in range(len(methods_present) - 1):
        ax.axhline(y + 0.5, color="#E4E8EE", linewidth=0.45, zorder=0.5)


def build_figure(output_dir: Path, formats: list[str]) -> list[Path]:
    _setup_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.9, 3.45))
    frames = {domain: _domain_frame(domain) for domain in ("economics", "energy")}
    _draw_merged_panel(ax, frames)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor=DOMAIN_STYLE["economics"]["color"], markeredgewidth=1.35, markersize=5.8, label="Economics"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="none",
                   markeredgecolor=DOMAIN_STYLE["energy"]["color"], markeredgewidth=1.35, markersize=5.8, label="Energy"),
        plt.Line2D([0], [0], color="#243B53", linewidth=1.8, label="95% CI"),
        plt.Rectangle((0, 0), 1, 1, color="#EDF2F7", label="Noise band $[-0.05, 0.10]$"),
        plt.Line2D([0], [0], color="#243B53", linestyle=(0, (3, 3)), linewidth=0.8, label="$R^2{=}0$"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.035), fontsize=7.2)

    fig.subplots_adjust(left=0.18, right=0.985, top=0.95, bottom=0.22)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        out = output_dir / f"panel_r2_floor.{fmt}"
        fig.savefig(out, dpi=320 if fmt == "png" else None, bbox_inches="tight", pad_inches=0.15)
        saved.append(out)
    plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "outputs" / "paper_assets" / LOCKED_PLAN / "compact_figures")
    p.add_argument("--formats", nargs="+", default=["png", "svg"])
    p.add_argument("--copy-to-paper-img", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    saved = build_figure(args.output_dir, args.formats)
    print("Generated:")
    for s in saved:
        print(" ", s)
    if args.copy_to_paper_img:
        target_dir = REPO / "paper_draft" / "img"
        target_dir.mkdir(parents=True, exist_ok=True)
        for s in saved:
            dest = target_dir / s.name
            shutil.copy2(s, dest)
            print(" copied:", dest)


if __name__ == "__main__":
    sys.exit(main())
