"""Helpers shared by the GA-Net baseline runners.

The GA-Net baseline reuses the LSTM-baseline runners verbatim by temporarily
swapping the ``PlainLSTMBaseline`` symbol referenced inside each runner module
with :class:`GANetBaseline`. The runner code only relies on the forward
signature and the ``y_pred`` field of the output dataclass, both of which the
GA-Net baseline matches. After the runner returns we restore the original
symbol so later baseline runs in the same process are unaffected.

This mirrors :mod:`experiments._tft_runner_utils` to keep the integration
pattern identical across baselines.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator

from baselines.ganet_baseline import GANetBaseline


_GANET_MODEL_TAG = "ganet"


@contextmanager
def patch_runner_model(runner_module: ModuleType) -> Iterator[None]:
    """Temporarily replace ``PlainLSTMBaseline`` in ``runner_module`` with GA-Net.

    Also rewrites the model-tag literal inside ``summarize_run`` so that
    summaries persisted to disk record ``"model": "ganet"``.
    """

    original_model = runner_module.PlainLSTMBaseline
    original_summarize = runner_module.summarize_run

    def tagged_summarize(*args, **kwargs):
        summary = original_summarize(*args, **kwargs)
        if isinstance(summary, dict):
            summary["model"] = _GANET_MODEL_TAG
        return summary

    runner_module.PlainLSTMBaseline = GANetBaseline  # type: ignore[attr-defined]
    runner_module.summarize_run = tagged_summarize  # type: ignore[attr-defined]
    try:
        yield
    finally:
        runner_module.PlainLSTMBaseline = original_model  # type: ignore[attr-defined]
        runner_module.summarize_run = original_summarize  # type: ignore[attr-defined]


__all__ = ["patch_runner_model", "GANetBaseline"]
