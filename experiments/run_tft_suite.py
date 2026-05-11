"""TFT-only runner for the 20-seed comparison.

Runs ONLY the TFT-style baseline on synthetic (linear + nonlinear), economics,
and energy across 20 seeds. Reuses the common-arg helpers from the main suite
but skips every other model so existing CMDL / Plain LSTM / ablation /
Grouped-ARDL artifacts are never touched. Resumable: tasks whose
``summary.json`` already exists are skipped unless ``--force`` is supplied.
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

from experiments import run_economics_tft_baseline
from experiments import run_energy_tft_baseline
from experiments import run_tft_baseline as synthetic_tft
from experiments.run_complete_20seed_suite import (
    SCENARIOS,
    SEEDS,
    economics_common_args,
    energy_common_args,
    summary_exists,
    synthetic_common_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the TFT baseline across 20 seeds.")
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


def run_one(label: str, run_dir: Path, force: bool, callback) -> None:
    if not force and summary_exists(run_dir):
        print(f"[skip] {label}: {run_dir}", flush=True)
        return
    print(f"[run] {label}: {run_dir}", flush=True)
    callback()
    cleanup()
    if not summary_exists(run_dir):
        raise RuntimeError(f"Expected summary.json was not created for {label}: {run_dir}")


def run_synthetic_tft(force: bool, seeds: list[int]) -> None:
    root = WORKSPACE_ROOT / "outputs" / "notebook_synthetic" / "complete_20seed" / "tft"
    root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for scenario in SCENARIOS:
            name = f"tft_{scenario}_seed{seed}"
            args = Namespace(**synthetic_common_args(root), seed=seed)
            run_one(
                f"synthetic TFT {scenario} seed {seed}",
                root / name,
                force,
                lambda args=args, name=name, scenario=scenario: synthetic_tft.run_experiment(args, name, scenario),
            )


def run_economics_tft(force: bool, seeds: list[int]) -> None:
    root = WORKSPACE_ROOT / "outputs" / "notebook_economics" / "complete_20seed" / "tft"
    root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        name = f"economics_tft_seed{seed}"
        args = Namespace(**economics_common_args(root), seed=seed, experiment_name=name)
        run_one(
            f"economics TFT seed {seed}",
            root / name,
            force,
            lambda args=args: run_economics_tft_baseline.run_experiment(args),
        )


def run_energy_tft(force: bool, seeds: list[int]) -> None:
    root = WORKSPACE_ROOT / "outputs" / "notebook_energy" / "complete_20seed" / "tft"
    root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        name = f"energy_tft_seed{seed}"
        args = Namespace(**energy_common_args(root), seed=seed, seeds=None, experiment_name=name)
        run_one(
            f"energy TFT seed {seed}",
            root / name,
            force,
            lambda args=args: run_energy_tft_baseline.run_experiment(args),
        )


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            f"torch={torch.__version__}, executable={sys.executable}. "
            "Activate the GPU-enabled env (e.g. `conda activate PTenv`) before launching."
        )
    print(
        f"[gpu] torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={torch.cuda.get_device_name(0)}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    _require_cuda()
    seeds = sorted({int(s) for s in (args.seeds or SEEDS)})
    invalid = [s for s in seeds if s not in SEEDS]
    if invalid:
        raise ValueError(f"Seeds outside the supported 20-seed range: {invalid}")
    if args.domain in {"all", "synthetic"}:
        run_synthetic_tft(args.force, seeds)
    if args.domain in {"all", "economics"}:
        run_economics_tft(args.force, seeds)
    if args.domain in {"all", "energy"}:
        run_energy_tft(args.force, seeds)
    print("TFT-only suite finished.", flush=True)


if __name__ == "__main__":
    main()
