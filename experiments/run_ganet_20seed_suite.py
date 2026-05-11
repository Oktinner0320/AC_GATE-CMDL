"""Run only the GA-Net 20-seed sweeps for the three domains.

This is a focused driver that mirrors the per-domain layout used by
:mod:`experiments.run_complete_20seed_suite` but only schedules the GA-Net
runners. It is intended as a one-off helper to add GA-Net into the
``complete_20seed`` plan without re-running CMDL/LSTM/TFT/ablation jobs that
already have summaries in their respective ``complete_20seed`` directories.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments import run_economics_ganet_baseline
from experiments import run_energy_ganet_baseline
from experiments import run_ganet_baseline as synthetic_ganet
from experiments.run_complete_20seed_suite import (
    SCENARIOS,
    SEEDS,
    economics_common_args,
    energy_common_args,
    run_task,
    synthetic_common_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the GA-Net 20-seed sweeps.")
    parser.add_argument(
        "--domain",
        choices=["all", "synthetic", "economics", "energy"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    return parser.parse_args()


def cleanup() -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_synthetic(force: bool, seeds: list[int]) -> None:
    root = WORKSPACE_ROOT / "outputs" / "notebook_synthetic" / "complete_20seed"
    ganet_dir = root / "ganet"
    ganet_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for scenario in SCENARIOS:
            name = f"ganet_{scenario}_seed{seed}"
            args = Namespace(**synthetic_common_args(ganet_dir), seed=seed)
            run_task(
                f"synthetic GA-Net {scenario} seed {seed}",
                ganet_dir / name,
                force,
                lambda args=args, name=name, scenario=scenario: synthetic_ganet.run_experiment(
                    args, name, scenario
                ),
            )
            cleanup()


def run_economics(force: bool, seeds: list[int]) -> None:
    root = WORKSPACE_ROOT / "outputs" / "notebook_economics" / "complete_20seed"
    ganet_dir = root / "ganet"
    ganet_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        name = f"economics_ganet_seed{seed}"
        args = Namespace(**economics_common_args(ganet_dir), seed=seed, experiment_name=name)
        run_task(
            f"economics GA-Net seed {seed}",
            ganet_dir / name,
            force,
            lambda args=args: run_economics_ganet_baseline.run_experiment(args),
        )
        cleanup()


def run_energy(force: bool, seeds: list[int]) -> None:
    root = WORKSPACE_ROOT / "outputs" / "notebook_energy" / "complete_20seed"
    ganet_dir = root / "ganet"
    ganet_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        name = f"energy_ganet_seed{seed}"
        args = Namespace(
            **energy_common_args(ganet_dir),
            seed=seed,
            seeds=None,
            experiment_name=name,
        )
        run_task(
            f"energy GA-Net seed {seed}",
            ganet_dir / name,
            force,
            lambda args=args: run_energy_ganet_baseline.run_experiment(args),
        )
        cleanup()


def main() -> None:
    args = parse_args()
    seeds = sorted({int(seed) for seed in (args.seeds or SEEDS)})
    if args.domain in {"all", "synthetic"}:
        run_synthetic(args.force, seeds)
    if args.domain in {"all", "economics"}:
        run_economics(args.force, seeds)
    if args.domain in {"all", "energy"}:
        run_energy(args.force, seeds)
    print("GA-Net 20-seed sweep finished.", flush=True)


if __name__ == "__main__":
    main()
