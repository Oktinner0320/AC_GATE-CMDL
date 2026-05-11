"""Build a standalone introduction use-case box image.

This script does not touch the manuscript or existing figure builders. It only
generates a compact PNG/SVG card that can be dropped into the Introduction as a
visual "application scenario" box.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO / "paper_draft" / "img"
DEFAULT_BASENAME = "intro_use_case_box"

TITLE = "Use case: lag auditing in country-level panels"
INPUTS = [
    "Temporal panel data",
    "Entity proxies",
]
OUTPUTS = [
    "effective lag k*",
    "stratifier\nalignment",
    "verdict\nmatrix",
]

NAVY = "#24426B"
TEXT = "#2A3B4D"
INPUT_FILL = "#EAF2FB"
CORE_FILL = "#FFF4E1"
OUTPUT_FILL = "#EEF8F1"
RULE = "#D9E4EF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Introduction use-case box image.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default=DEFAULT_BASENAME)
    parser.add_argument("--formats", nargs="+", default=["png", "svg"])
    return parser.parse_args()


def _add_round_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    edgecolor: str = NAVY,
    linewidth: float = 1.4,
    radius: float = 0.025,
    pad: float = 0.012,
    zorder: float = 2.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _add_straight_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    shrink_a: float = 4.0,
    shrink_b: float = 8.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color=NAVY,
        connectionstyle="arc3,rad=0.0",
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        zorder=2.6,
    )
    ax.add_patch(arrow)


def _add_elbow_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, elbow_x: float) -> None:
    start_x, start_y = start
    end_x, end_y = end
    ax.plot([start_x, elbow_x], [start_y, start_y], color=NAVY, linewidth=1.8, zorder=2.4)
    ax.plot([elbow_x, elbow_x], [start_y, end_y], color=NAVY, linewidth=1.8, zorder=2.4)
    arrow = FancyArrowPatch((elbow_x, end_y), (end_x, end_y), arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=NAVY, connectionstyle="arc3,rad=0.0", zorder=2.6)
    ax.add_patch(arrow)


def build_figure(output_dir: Path, basename: str, formats: list[str]) -> list[Path]:
    fig = plt.figure(figsize=(7.0, 3.3))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    input_boxes = [(0.07, 0.60, 0.22, 0.12), (0.07, 0.27, 0.22, 0.12)]
    for title, (x, y, w, h) in zip(INPUTS, input_boxes):
        _add_round_box(ax, x, y, w, h, facecolor=INPUT_FILL, edgecolor=NAVY, linewidth=1.3, radius=0.024, zorder=2.1)
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=10.0, fontweight="bold", color=NAVY, zorder=3)

    core_x, core_y, core_w, core_h = 0.355, 0.18, 0.29, 0.56
    divider_y = 0.47
    _add_round_box(ax, core_x, core_y, core_w, core_h, facecolor=CORE_FILL, edgecolor=NAVY, linewidth=1.6, radius=0.03, zorder=1.9)
    ax.plot([core_x + 0.02, core_x + core_w - 0.02], [divider_y, divider_y], color=RULE, linewidth=1.2, zorder=2.2)
    ax.text(core_x + core_w / 2, 0.64, "Lag recovery layer", ha="center", va="center", fontsize=10.5, fontweight="bold", color=NAVY, zorder=3)
    ax.text(core_x + core_w / 2, 0.56, "recover non-degenerate k*", ha="center", va="center", fontsize=8.4, color=TEXT, zorder=3)
    ax.text(core_x + core_w / 2, 0.35, "Audit evidence layer", ha="center", va="center", fontsize=10.3, fontweight="bold", color=NAVY, zorder=3)
    ax.text(
        core_x + core_w / 2,
        0.25,
        "check stratifier alignment\nassemble verdict evidence",
        ha="center",
        va="center",
        fontsize=8.6,
        color=TEXT,
        linespacing=1.18,
        zorder=3,
    )

    output_w = 0.19
    output_h = 0.078
    output_step = 0.17
    bus_y = core_y + core_h / 2
    output_centers = [bus_y + output_step, bus_y, bus_y - output_step]
    output_boxes = [(0.77, center_y - output_h / 2, output_w, output_h) for center_y in output_centers]
    for label, (x, y, w, h) in zip(OUTPUTS, output_boxes):
        _add_round_box(ax, x, y, w, h, facecolor=OUTPUT_FILL, edgecolor=NAVY, linewidth=1.3, radius=0.024, pad=0.004, zorder=2.1)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.4, fontweight="bold", color=NAVY, linespacing=1.08, zorder=3)

    _add_straight_arrow(ax, (0.302, 0.66), (core_x - 0.014, 0.66), shrink_a=0.0, shrink_b=0.0)
    _add_straight_arrow(ax, (0.302, 0.33), (core_x - 0.014, 0.33), shrink_a=0.0, shrink_b=0.0)

    bus_x = 0.703
    ax.plot([core_x + core_w + 0.012, bus_x], [bus_y, bus_y], color=NAVY, linewidth=1.8, zorder=2.4)
    ax.plot([bus_x, bus_x], [output_centers[-1], output_centers[0]], color=NAVY, linewidth=1.8, zorder=2.4)
    ax.add_patch(Circle((bus_x, bus_y), radius=0.0065, facecolor=NAVY, edgecolor=NAVY, zorder=2.7))
    for center_y in output_centers:
        _add_straight_arrow(ax, (bus_x, center_y), (0.758, center_y), shrink_a=0.0, shrink_b=0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{basename}.{fmt}"
        fig.savefig(path, dpi=320 if fmt == "png" else None, bbox_inches="tight", pad_inches=0.01)
        saved.append(path)
    plt.close(fig)
    return saved


def main() -> int:
    args = parse_args()
    saved = build_figure(args.output_dir.resolve(), args.basename, list(args.formats))
    print("Generated:")
    for path in saved:
        print(" ", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())