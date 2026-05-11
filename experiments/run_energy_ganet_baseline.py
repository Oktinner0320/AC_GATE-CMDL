"""Energy-domain GA-Net baseline runner.

Thin wrapper around :mod:`experiments.run_energy_lstm_baseline` that swaps
the model class to :class:`GANetBaseline` while reusing the loader, temporal
splits, training loop and post-hoc lag-occlusion diagnostics already validated
for the LSTM baseline. Mirrors the TFT runner layout for parity.
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

from experiments import run_energy_lstm_baseline as lstm_runner
from experiments._ganet_runner_utils import patch_runner_model


def run_experiment(args: argparse.Namespace):
    with patch_runner_model(lstm_runner):
        return lstm_runner.run_experiment(args)


def main() -> None:
    args = lstm_runner.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
