"""Paired significance tests across seeds for comparison tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _as_list(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _group_items(frame: pd.DataFrame, group_cols: list[str]):
    if not group_cols:
        yield {}, frame
        return
    grouped = frame.groupby(group_cols, dropna=False, sort=True)
    for key, group in grouped:
        values = key if isinstance(key, tuple) else (key,)
        yield dict(zip(group_cols, values)), group


def paired_wilcoxon(
    per_seed: pd.DataFrame,
    metric: str,
    method_col: str = "model",
    seed_col: str = "seed",
    reference: str = "CMDL",
    group_cols: Sequence[str] | str | None = None,
    min_pairs: int = 5,
    greater_is_better: bool = True,
) -> pd.DataFrame:
    """Return paired Wilcoxon signed-rank tests comparing methods to a reference.

    Rows are paired by ``seed_col`` within each optional group. Duplicate rows for
    the same seed/method are averaged before testing, which keeps notebook-built
    long tables usable when an upstream join creates repeated records.
    """

    resolved_group_cols = [column for column in _as_list(group_cols) if column in per_seed.columns]
    columns = [
        *resolved_group_cols,
        "metric",
        "reference",
        "method",
        "n_pairs",
        "reference_mean",
        "method_mean",
        "mean_diff",
        "median_diff",
        "wilcoxon_statistic",
        "wilcoxon_p",
        "greater_is_better",
        "reference_better_mean",
    ]
    if per_seed.empty:
        return pd.DataFrame(columns=columns)

    required = [metric, method_col, seed_col]
    missing = [column for column in required if column not in per_seed.columns]
    if missing:
        raise ValueError(f"Missing required columns for paired Wilcoxon: {missing}")
    if reference not in set(per_seed[method_col].dropna()):
        raise ValueError(f"reference {reference!r} not in {sorted(set(per_seed[method_col].dropna()))}")

    rows: list[dict[str, Any]] = []
    for group_values, group in _group_items(per_seed, resolved_group_cols):
        working = group.loc[:, [seed_col, method_col, metric]].copy()
        working[metric] = pd.to_numeric(working[metric], errors="coerce")
        working = working.dropna(subset=[seed_col, method_col, metric])
        if working.empty:
            continue

        pivot = working.pivot_table(
            index=seed_col,
            columns=method_col,
            values=metric,
            aggfunc="mean",
        )
        if reference not in pivot.columns:
            continue

        for method in pivot.columns:
            if method == reference:
                continue
            joined = pivot.loc[:, [reference, method]].dropna()
            row: dict[str, Any] = {
                **group_values,
                "metric": metric,
                "reference": reference,
                "method": method,
                "n_pairs": int(len(joined)),
                "reference_mean": np.nan,
                "method_mean": np.nan,
                "mean_diff": np.nan,
                "median_diff": np.nan,
                "wilcoxon_statistic": np.nan,
                "wilcoxon_p": np.nan,
                "greater_is_better": bool(greater_is_better),
                "reference_better_mean": np.nan,
            }
            if len(joined) >= min_pairs:
                diff = joined[reference] - joined[method]
                row["reference_mean"] = float(joined[reference].mean())
                row["method_mean"] = float(joined[method].mean())
                row["mean_diff"] = float(diff.mean())
                row["median_diff"] = float(np.median(diff))
                row["reference_better_mean"] = bool(row["mean_diff"] > 0.0) if greater_is_better else bool(row["mean_diff"] < 0.0)
                try:
                    test_result = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                    row["wilcoxon_statistic"] = float(test_result.statistic)
                    row["wilcoxon_p"] = float(test_result.pvalue)
                except ValueError:
                    row["wilcoxon_statistic"] = np.nan
                    row["wilcoxon_p"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows, columns=columns)


__all__ = ["paired_wilcoxon"]