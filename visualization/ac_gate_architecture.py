"""Standalone AC-GATE architecture figure generator.

Run directly from the repository root, for example:

    python visualization/ac_gate_architecture.py

The script writes both PDF and PNG outputs by default and does not depend on
the experiment pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEFAULT_OUTPUT_DIR = Path("outputs") / "paper_assets"
DEFAULT_BASENAME = "ac_gate_architecture"

COLORS = {
    "fill": "#F7F8FA",
    "proxy": "#EAF2FB",
    "gate": "#EAF6EE",
    "aggregate": "#FFF3D8",
    "backbone": "#F1ECF8",
    "output": "#FCECEC",
    "edge": "#25313A",
    "arrow": "#2F5D7C",
    "blocked": "#B23A48",
    "muted": "#5B6470",
}


def _box(
    ax: plt.Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    title: str,
    subtitle: str | None = None,
    facecolor: str = COLORS["fill"],
    edgecolor: str = COLORS["edge"],
    linewidth: float = 0.85,
    title_size: float = 7.3,
    title_weight: str = "bold",
    subtitle_size: float = 5.5,
    subtitle_color: str = COLORS["muted"],
) -> None:
    width, height = size
    left = center[0] - width / 2.0
    bottom = center[1] - height / 2.0
    patch = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.014",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        center[0],
        center[1] + (0.18 * height if subtitle else 0.0),
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight=title_weight,
        color=edgecolor,
    )
    if subtitle:
        ax.text(
            center[0],
            center[1] - (0.22 * height),
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=subtitle_color,
        )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["arrow"],
    linestyle: str = "-",
    linewidth: float = 0.8,
    mutation_scale: float = 7.0,
    connectionstyle: str = "arc3,rad=0.0",
    shrinkA: float = 4.0,
    shrinkB: float = 4.0,
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
    )
    ax.add_patch(patch)


def _label(ax: plt.Axes, xy: tuple[float, float], text: str, color: str = COLORS["muted"], size: float = 5.8) -> None:
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=size, color=color)


def _elbow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    elbow_x: float,
    color: str = COLORS["arrow"],
    linewidth: float = 0.8,
    start_gap: float = 0.010,
    end_gap: float = 0.012,
) -> None:
    start_out = (start[0] + start_gap, start[1])
    end_out = (end[0] - end_gap, end[1])
    mid_1 = (elbow_x, start_out[1])
    mid_2 = (elbow_x, end_out[1])
    ax.plot([start_out[0], mid_1[0]], [start_out[1], mid_1[1]], color=color, linewidth=linewidth)
    ax.plot([mid_1[0], mid_2[0]], [mid_1[1], mid_2[1]], color=color, linewidth=linewidth)
    _arrow(ax, mid_2, end_out, color=color, linewidth=linewidth, shrinkA=0.0, shrinkB=0.0)


def _save(fig: plt.Figure, output_dir: Path, basename: str, formats: Iterable[str], dpi: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        normalized = fmt.lower().lstrip(".")
        target = output_dir / f"{basename}.{normalized}"
        save_kwargs = {"bbox_inches": "tight"}
        if normalized in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(target, **save_kwargs)
        written.append(target)
    return written


def draw_ac_gate_architecture() -> plt.Figure:
    """Draw the AC-GATE architecture as a self-contained matplotlib figure."""

    fig, ax = plt.subplots(figsize=(3.58, 3.02))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)

    left_x = 0.28
    right_x = 0.73
    box_h = 0.132
    nodes = {
        "p": (left_x, 0.91),
        "encoder": (left_x, 0.70),
        "z": (left_x, 0.49),
        "gate": (left_x, 0.28),
        "omega": (left_x, 0.08),
        "lags": (right_x, 0.86),
        "context": (right_x, 0.65),
        "lstm": (right_x, 0.44),
        "yhat": (right_x, 0.23),
    }
    sizes = {
        "p": (0.210, box_h),
        "encoder": (0.260, box_h),
        "z": (0.290, box_h),
        "gate": (0.260, box_h),
        "omega": (0.340, box_h),
        "lags": (0.290, box_h),
        "context": (0.390, box_h),
        "lstm": (0.250, box_h),
        "yhat": (0.265, box_h),
    }

    _box(ax, nodes["p"], sizes["p"], r"$p_i$", None, COLORS["proxy"], title_size=8.2, title_weight="bold")
    _box(ax, nodes["encoder"], sizes["encoder"], "AC encoder", None, COLORS["proxy"])
    _box(
        ax,
        nodes["z"],
        sizes["z"],
        r"$z_i = f_{\phi}(p_i)$",
        None,
        COLORS["proxy"],
        title_size=6.9,
        title_weight="bold",
    )
    _box(ax, nodes["gate"], sizes["gate"], "Lag gate", None, COLORS["gate"])
    _box(
        ax,
        nodes["omega"],
        sizes["omega"],
        r"$\omega_{i,k} = \mathrm{Softmax}_{k}(\cdot)$",
        None,
        COLORS["gate"],
        title_size=6.1,
        title_weight="bold",
    )
    _box(ax, nodes["lags"], sizes["lags"], r"$X_{i,t-1:t-K}$", None, COLORS["aggregate"], title_size=7.8, title_weight="bold")
    _box(
        ax,
        nodes["context"],
        sizes["context"],
        r"$c_{i,t} = \sum_k \omega_{i,k}X_{i,t-k}$",
        None,
        COLORS["aggregate"],
        title_size=5.9,
        title_weight="bold",
    )
    _box(ax, nodes["lstm"], sizes["lstm"], "LSTM", None, COLORS["backbone"])
    _box(ax, nodes["yhat"], sizes["yhat"], r"$\widehat{Y}_{i,t+1}$", None, COLORS["output"], title_size=7.9, title_weight="bold")

    left_order = ["p", "encoder", "z", "gate", "omega"]
    for upper, lower in zip(left_order[:-1], left_order[1:]):
        start = (nodes[upper][0], nodes[upper][1] - box_h / 2.0)
        end = (nodes[lower][0], nodes[lower][1] + box_h / 2.0)
        _arrow(ax, start, end)

    right_order = ["lags", "context", "lstm", "yhat"]
    for upper, lower in zip(right_order[:-1], right_order[1:]):
        start = (nodes[upper][0], nodes[upper][1] - box_h / 2.0)
        end = (nodes[lower][0], nodes[lower][1] + box_h / 2.0)
        _arrow(ax, start, end)

    _elbow_arrow(
        ax,
        (nodes["omega"][0] + sizes["omega"][0] / 2.0, nodes["omega"][1]),
        (nodes["context"][0] - sizes["context"][0] / 2.0, nodes["context"][1]),
        elbow_x=0.49,
    )

    ax.text(
        nodes["lstm"][0],
        nodes["lstm"][1] - 0.040,
        r"no direct $X_{i,t}$ input",
        ha="center",
        va="center",
        fontsize=4.6,
        color=COLORS["blocked"],
    )

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the AC-GATE architecture figure.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", type=str, default=DEFAULT_BASENAME)
    parser.add_argument("--formats", nargs="+", default=["svg", "png"], help="Output formats, e.g. svg png pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Display the figure after saving it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig = draw_ac_gate_architecture()
    written = _save(fig, args.output_dir, args.basename, args.formats, args.dpi)
    for path in written:
        print(path)
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()