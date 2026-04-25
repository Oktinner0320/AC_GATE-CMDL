"""Grouped distributed-lag baseline utilities.

This module exposes the anchor-quantile grouped ARDL-style OLS baseline while
keeping the implementation shared with the simple panel calibration helpers.
"""

from baselines.panel_ols import evaluate_grouped_ardl_baseline


__all__ = ["evaluate_grouped_ardl_baseline"]