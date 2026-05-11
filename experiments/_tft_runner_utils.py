"""Helpers shared by the TFT-baseline runners.

The TFT baseline reuses the LSTM-baseline runners verbatim by temporarily
swapping the ``PlainLSTMBaseline`` symbol referenced inside each runner module
with :class:`TFTBaseline`. The runner code only relies on the forward signature
and the ``y_pred`` field of the output dataclass, both of which the TFT
baseline matches. After the runner returns we restore the original symbol so
later LSTM runs in the same process are unaffected.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator

from baselines.tft_baseline import TFTBaseline


_TFT_MODEL_TAG = "tft"


@contextmanager
def patch_runner_model(runner_module: ModuleType) -> Iterator[None]:
    """Temporarily replace ``PlainLSTMBaseline`` in ``runner_module`` with TFT.

    Also rewrites the ``"plain_lstm"`` model-tag literal inside ``summarize_run``
    so that summaries persisted to disk record ``"model": "tft"``.
    """

    original_model = runner_module.PlainLSTMBaseline
    original_summarize = runner_module.summarize_run

    def tagged_summarize(*args, **kwargs):
        summary = original_summarize(*args, **kwargs)
        if isinstance(summary, dict):
            summary["model"] = _TFT_MODEL_TAG
        return summary

    runner_module.PlainLSTMBaseline = TFTBaseline  # type: ignore[attr-defined]
    runner_module.summarize_run = tagged_summarize  # type: ignore[attr-defined]
    try:
        yield
    finally:
        runner_module.PlainLSTMBaseline = original_model  # type: ignore[attr-defined]
        runner_module.summarize_run = original_summarize  # type: ignore[attr-defined]


__all__ = ["patch_runner_model", "TFTBaseline"]
