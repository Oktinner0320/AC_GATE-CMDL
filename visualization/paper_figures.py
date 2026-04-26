"""Paper-facing figure builders for CMDL reporting artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _finalize_figure(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(target, dpi=200, bbox_inches="tight")
    return fig


def plot_workflow_overview(save_path: str | Path | None = None) -> plt.Figure:
    """Render the reporting workflow used by the venue-specific paper variants."""

    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.axis("off")

    boxes = [
        (0.04, "Data and Splits", "balanced panel\ntrain/val/test windows\nproxy metadata"),
        (0.25, "CMDL and Baselines", "CMDL\nPlain LSTM\nGrouped ARDL\nablations"),
        (0.46, "Diagnostics", "significance\nstratified k*\nablation guard\nseed stability"),
        (0.67, "Paper Artifacts", "main tables\nCI summaries\nverdict matrix\ncase-study sources"),
        (0.88, "Venue Packaging", "ICDM formalization\nCIKM workflow framing\nECML case-study framing"),
    ]

    for x_pos, title, body in boxes:
        patch = FancyBboxPatch(
            (x_pos - 0.085, 0.2),
            0.17,
            0.55,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.4,
            edgecolor="#1F3A5F",
            facecolor="#F4F7FB",
        )
        ax.add_patch(patch)
        ax.text(x_pos, 0.63, title, ha="center", va="center", fontsize=12, fontweight="bold", color="#1F3A5F")
        ax.text(x_pos, 0.43, body, ha="center", va="center", fontsize=10, color="#22313F")

    for left, right in zip(boxes[:-1], boxes[1:]):
        arrow = FancyArrowPatch(
            (left[0] + 0.095, 0.475),
            (right[0] - 0.095, 0.475),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=1.6,
            color="#2E6F95",
        )
        ax.add_patch(arrow)

    ax.set_title("Shared Reporting Workflow for ICDM, CIKM, and ECML Variants", fontsize=14, fontweight="bold")
    return _finalize_figure(fig, save_path)


def plot_seed_distribution(
    frame: pd.DataFrame,
    value_col: str,
    category_col: str,
    title: str,
    ylabel: str,
    save_path: str | Path | None = None,
    facet_col: str | None = None,
    order: Sequence[str] | None = None,
    palette: str = "Set2",
    rotate_labels: bool = True,
) -> plt.Figure:
    """Draw box-and-strip seed distributions, optionally faceted by one column."""

    if frame.empty:
        raise ValueError("plot_seed_distribution requires a non-empty frame")
    if value_col not in frame.columns or category_col not in frame.columns:
        raise ValueError(f"Missing required columns for seed plot: {value_col}, {category_col}")

    plot_frame = frame.copy()
    plot_frame[value_col] = pd.to_numeric(plot_frame[value_col], errors="coerce")
    plot_frame = plot_frame.dropna(subset=[value_col, category_col])
    if plot_frame.empty:
        raise ValueError(f"No valid rows available for {value_col}")

    if order is None:
        order = list(dict.fromkeys(plot_frame[category_col].tolist()))

    if facet_col is not None and facet_col in plot_frame.columns:
        facet_values = [value for value in plot_frame[facet_col].dropna().unique().tolist()]
        n_panels = max(1, len(facet_values))
        fig, axes = plt.subplots(1, n_panels, figsize=(6.3 * n_panels, 5.2), sharey=True)
        axes_array = np.atleast_1d(axes)
        for axis, facet_value in zip(axes_array, facet_values):
            facet_frame = plot_frame.loc[plot_frame[facet_col] == facet_value].copy()
            sns.boxplot(
                data=facet_frame,
                x=category_col,
                y=value_col,
                hue=category_col,
                order=order,
                palette=palette,
                ax=axis,
                fliersize=0,
                dodge=False,
                legend=False,
            )
            sns.stripplot(
                data=facet_frame,
                x=category_col,
                y=value_col,
                order=order,
                ax=axis,
                color="#243B53",
                alpha=0.7,
                size=4.5,
                jitter=0.18,
            )
            axis.set_title(str(facet_value))
            axis.set_xlabel("")
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.2)
            if rotate_labels:
                axis.tick_params(axis="x", rotation=20)
    else:
        fig, axis = plt.subplots(figsize=(8.6, 5.2))
        sns.boxplot(
            data=plot_frame,
            x=category_col,
            y=value_col,
            hue=category_col,
            order=order,
            palette=palette,
            ax=axis,
            fliersize=0,
            dodge=False,
            legend=False,
        )
        sns.stripplot(
            data=plot_frame,
            x=category_col,
            y=value_col,
            order=order,
            ax=axis,
            color="#243B53",
            alpha=0.7,
            size=4.5,
            jitter=0.18,
        )
        axis.set_xlabel("")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
        if rotate_labels:
            axis.tick_params(axis="x", rotation=20)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return _finalize_figure(fig, save_path)


def plot_stratified_seed_distribution(
    frame: pd.DataFrame,
    title: str,
    ylabel: str,
    save_path: str | Path | None = None,
    stratifier_order: Sequence[str] | None = None,
) -> plt.Figure:
    """Plot per-seed absolute stratified correlations for CMDL."""

    if frame.empty:
        raise ValueError("plot_stratified_seed_distribution requires a non-empty frame")
    required_columns = {"stratifier", "spearman_rho"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required_columns)}")

    plot_frame = frame.copy()
    plot_frame["abs_spearman_rho"] = pd.to_numeric(plot_frame["spearman_rho"], errors="coerce").abs()
    plot_frame = plot_frame.dropna(subset=["stratifier", "abs_spearman_rho"])
    if plot_frame.empty:
        raise ValueError("No valid per-seed stratified rows were available")

    if stratifier_order is None:
        ordering = (
            plot_frame.groupby("stratifier", dropna=False)["abs_spearman_rho"].mean().sort_values(ascending=False).index.tolist()
        )
    else:
        ordering = list(stratifier_order)

    fig, axis = plt.subplots(figsize=(8.8, 5.0))
    sns.boxplot(
        data=plot_frame,
        x="stratifier",
        y="abs_spearman_rho",
        hue="stratifier",
        order=ordering,
        ax=axis,
        palette="Blues",
        fliersize=0,
        dodge=False,
        legend=False,
    )
    sns.stripplot(
        data=plot_frame,
        x="stratifier",
        y="abs_spearman_rho",
        order=ordering,
        ax=axis,
        color="#243B53",
        alpha=0.7,
        size=4.5,
        jitter=0.18,
    )
    axis.set_xlabel("")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    axis.set_title(title, fontsize=13, fontweight="bold")
    return _finalize_figure(fig, save_path)


def _annotation_subset(frame: pd.DataFrame, x_col: str, count: int) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    sorted_frame = frame.sort_values(x_col, na_position="last")
    half = max(1, count // 2)
    subset = pd.concat([sorted_frame.head(half), sorted_frame.tail(half)], ignore_index=False)
    return subset.drop_duplicates()


def plot_case_study_scatter(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str | Path | None = None,
    annotate_top_n: int = 12,
) -> plt.Figure:
    """Plot per-entity k* against one stratifier and annotate extreme entities."""

    if frame.empty:
        raise ValueError("plot_case_study_scatter requires a non-empty frame")
    required_columns = {x_col, y_col, label_col}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required_columns)}")

    plot_frame = frame.copy()
    plot_frame[x_col] = pd.to_numeric(plot_frame[x_col], errors="coerce")
    plot_frame[y_col] = pd.to_numeric(plot_frame[y_col], errors="coerce")
    plot_frame = plot_frame.dropna(subset=[x_col, y_col])
    if plot_frame.empty:
        raise ValueError("No valid rows available for case-study scatter")

    fig, axis = plt.subplots(figsize=(8.2, 6.0))
    axis.scatter(plot_frame[x_col], plot_frame[y_col], s=42, alpha=0.82, color="#2E6F95", edgecolor="white", linewidth=0.4)

    if plot_frame[x_col].nunique() > 1:
        coefficients = np.polyfit(plot_frame[x_col], plot_frame[y_col], deg=1)
        x_values = np.linspace(plot_frame[x_col].min(), plot_frame[x_col].max(), 100)
        y_values = np.polyval(coefficients, x_values)
        axis.plot(x_values, y_values, linestyle="--", linewidth=1.3, color="#D1495B")

    for _, row in _annotation_subset(plot_frame, x_col=x_col, count=annotate_top_n).iterrows():
        axis.annotate(
            str(row[label_col]),
            xy=(row[x_col], row[y_col]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.2,
            color="#12263A",
        )

    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
    return _finalize_figure(fig, save_path)


__all__ = [
    "plot_case_study_scatter",
    "plot_seed_distribution",
    "plot_stratified_seed_distribution",
    "plot_workflow_overview",
]