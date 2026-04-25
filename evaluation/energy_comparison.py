"""Utilities for consolidating energy CMDL, baseline, and ablation outputs.

统一读取 energy CMDL、plain LSTM baseline 与 ablation 的 summary.json，
并生成可直接在 notebook 中展示的对比表。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from evaluation.economics_comparison import (
    build_economics_comparison,
    build_grouped_ardl_lag_trend_table as _build_grouped_ardl_lag_trend_table,
    build_interpretability_table as _build_interpretability_table,
    build_mechanism_result_log as _build_mechanism_result_log,
    build_mechanism_summary_table as _build_mechanism_summary_table,
    build_per_proxy_audit_summary_table as _build_per_proxy_audit_summary_table,
    build_per_proxy_alignment_table as _build_per_proxy_alignment_table,
    build_task_table as _build_task_table,
)


def build_energy_comparison(
    cmdl_root: Path | str | None = None,
    baseline_root: Path | str | None = None,
    ablation_root: Path | str | None = None,
    grouped_ardl_root: Path | str | None = None,
) -> pd.DataFrame:
    """Combine energy CMDL, baseline, and ablation runs into one table."""

    return build_economics_comparison(
        cmdl_root=cmdl_root,
        baseline_root=baseline_root,
        ablation_root=ablation_root,
        grouped_ardl_root=grouped_ardl_root,
    )


def build_task_table(comparison_frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Build the forecast-comparison table used in the energy notebook."""

    return _build_task_table(comparison_frame, split=split)


def build_interpretability_table(comparison_frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Build the lag/proxy diagnostics table used in the energy notebook."""

    return _build_interpretability_table(comparison_frame, split=split)


def build_per_proxy_alignment_table(
    comparison_frame: pd.DataFrame,
    split: str = "test",
    display_names: list[str] | tuple[str, ...] | None = ("CMDL",),
) -> pd.DataFrame:
    """Build a long-form per-proxy adjusted-rho table for anchor audits."""

    return _build_per_proxy_alignment_table(
        comparison_frame,
        split=split,
        display_names=display_names,
    )


def build_per_proxy_audit_summary_table(
    comparison_frame: pd.DataFrame,
    split: str = "test",
    display_names: list[str] | tuple[str, ...] | None = ("CMDL",),
) -> pd.DataFrame:
    """Aggregate named per-proxy mechanism alignment across seeds."""

    return _build_per_proxy_audit_summary_table(
        comparison_frame,
        split=split,
        display_names=display_names,
    )


def build_grouped_ardl_lag_trend_table(comparison_frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Build low/mid/high grouped-ARDL lag trend rows for mechanism audits."""

    return _build_grouped_ardl_lag_trend_table(comparison_frame, split=split)


def build_mechanism_summary_table(comparison_frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Aggregate task and mechanism diagnostics across seeds."""

    return _build_mechanism_summary_table(comparison_frame, split=split)


def build_mechanism_result_log(comparison_frame: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """Build a compact mechanism-first yes/partial/no result log."""

    return _build_mechanism_result_log(comparison_frame, split=split)


__all__ = [
    "build_energy_comparison",
    "build_grouped_ardl_lag_trend_table",
    "build_interpretability_table",
    "build_mechanism_result_log",
    "build_mechanism_summary_table",
    "build_per_proxy_audit_summary_table",
    "build_per_proxy_alignment_table",
    "build_task_table",
]
