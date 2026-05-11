"""Synthetic TFT-style baseline runner.

Thin wrapper around :mod:`experiments.run_lstm_baseline` that swaps the model
class to :class:`TFTBaseline` while reusing the same data, training loop,
warm-up handling and post-hoc lag-occlusion analysis. Output directories are
expected to differ from the LSTM baseline so existing artifacts are untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments import run_lstm_baseline as lstm_runner
from experiments._tft_runner_utils import patch_runner_model


def run_experiment(args: argparse.Namespace, experiment_name: str, scenario: str):
    """Run one synthetic TFT-baseline experiment matching the LSTM-runner API."""

    with patch_runner_model(lstm_runner):
        return lstm_runner.run_experiment(args=args, experiment_name=experiment_name, scenario=scenario)


def main() -> None:
    args = lstm_runner.parse_args()
    if args.scenario in {"all", "linear"}:
        run_experiment(args, experiment_name="TFT_linear", scenario="linear")
    if args.scenario in {"all", "nonlinear"}:
        run_experiment(args, experiment_name="TFT_nonlinear", scenario="nonlinear")


if __name__ == "__main__":
    main()
