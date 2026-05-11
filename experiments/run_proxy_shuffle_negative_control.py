"""Run real-data proxy-shuffle negative-control AC-GATE experiments.

This runner writes to an isolated negative-control output tree and never uses
the locked notebook result directories as its destination.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from argparse import Namespace
from datetime import date
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiments import run_economics, run_energy
from experiments._proxy_shuffle_control import apply_proxy_shuffle_to_setup, ensure_safe_negative_control_output_root
from experiments.run_complete_20seed_suite import SEEDS, economics_common_args, energy_common_args


def default_output_root(smoke: bool = False) -> Path:
    label = "proxy_shuffle_smoke" if smoke else "proxy_shuffle_20seed"
    return WORKSPACE_ROOT / "outputs" / "negative_controls" / f"{label}_{date.today():%Y%m%d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run proxy-shuffle negative-control AC-GATE experiments.")
    parser.add_argument("--domain", choices=["all", "economics", "energy"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--proxy-perm-seed-offset", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true", help="Run one-epoch smoke checks in an isolated smoke output tree.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def summary_exists(run_dir: Path) -> bool:
    return (run_dir / "summary.json").exists()


def cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _run_task(label: str, run_dir: Path, force: bool, dry_run: bool, callback) -> None:
    if not force and summary_exists(run_dir):
        print(f"[skip] {label}: {run_dir}", flush=True)
        return
    if dry_run:
        print(f"[dry-run] {label}: {run_dir}", flush=True)
        return
    print(f"[run] {label}: {run_dir}", flush=True)
    callback()
    cleanup()
    if not summary_exists(run_dir):
        raise RuntimeError(f"Expected summary.json was not created for {label}: {run_dir}")


def _proxy_transform(training_seed: int, proxy_perm_seed: int):
    def transform(setup):
        return apply_proxy_shuffle_to_setup(
            setup=setup,
            training_seed=training_seed,
            proxy_perm_seed=proxy_perm_seed,
        )

    return transform


def _run_economics(output_root: Path, seeds: list[int], force: bool, dry_run: bool, seed_offset: int, smoke: bool) -> None:
    cmdl_dir = output_root / "economics" / "cmdl"
    for seed in seeds:
        name = f"economics_proxy_shuffle_seed{seed}"
        proxy_perm_seed = int(seed_offset + seed)
        common_args = economics_common_args(cmdl_dir)
        common_args["smoke"] = bool(smoke)
        args = Namespace(
            **common_args,
            seed=seed,
            experiment_name=name,
            negative_control="proxy_shuffle",
            proxy_perm_seed=proxy_perm_seed,
        )
        _run_task(
            f"economics proxy-shuffle AC-GATE seed {seed}",
            cmdl_dir / name,
            force,
            dry_run,
            lambda args=args, seed=seed, proxy_perm_seed=proxy_perm_seed: run_economics.run_experiment(
                args,
                setup_transform=_proxy_transform(seed, proxy_perm_seed),
            ),
        )


def _run_energy(output_root: Path, seeds: list[int], force: bool, dry_run: bool, seed_offset: int, smoke: bool) -> None:
    cmdl_dir = output_root / "energy" / "cmdl"
    for seed in seeds:
        name = f"energy_proxy_shuffle_seed{seed}"
        proxy_perm_seed = int(seed_offset + seed)
        common_args = energy_common_args(cmdl_dir)
        common_args["smoke"] = bool(smoke)
        args = Namespace(
            **common_args,
            seed=seed,
            seeds=None,
            experiment_name=name,
            negative_control="proxy_shuffle",
            proxy_perm_seed=proxy_perm_seed,
        )
        _run_task(
            f"energy proxy-shuffle AC-GATE seed {seed}",
            cmdl_dir / name,
            force,
            dry_run,
            lambda args=args, seed=seed, proxy_perm_seed=proxy_perm_seed: run_energy.run_experiment(
                args,
                setup_transform=_proxy_transform(seed, proxy_perm_seed),
            ),
        )


def main() -> None:
    args = parse_args()
    seeds = sorted({int(seed) for seed in (args.seeds or SEEDS)})
    invalid_seeds = [seed for seed in seeds if seed not in SEEDS]
    if invalid_seeds:
        raise ValueError(f"Seeds outside the supported complete-suite range: {invalid_seeds}")

    output_root = ensure_safe_negative_control_output_root(args.output_root or default_output_root(args.smoke))
    print(f"Negative-control output root: {output_root}", flush=True)
    if args.domain in {"all", "economics"}:
        _run_economics(output_root, seeds, args.force, args.dry_run, args.proxy_perm_seed_offset, args.smoke)
    if args.domain in {"all", "energy"}:
        _run_energy(output_root, seeds, args.force, args.dry_run, args.proxy_perm_seed_offset, args.smoke)


if __name__ == "__main__":
    main()