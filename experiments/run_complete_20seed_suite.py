"""Run the complete 20-seed notebook experiment suite.

This script mirrors the three compact result notebooks and writes artifacts to
outputs/notebook_*/complete_20seed. It is resumable: a run whose summary.json is
already present is skipped unless --force is supplied.
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

from experiments import run_ablation as synthetic_ablation
from experiments import run_economics
from experiments import run_economics_ablation
from experiments import run_economics_grouped_ardl
from experiments import run_economics_lstm_baseline
from experiments import run_energy
from experiments import run_energy_ablation
from experiments import run_energy_grouped_ardl
from experiments import run_energy_lstm_baseline
from experiments import run_lstm_baseline as synthetic_lstm
from experiments import run_synthetic


SEEDS = list(range(20))
SCENARIOS = ["linear", "nonlinear"]
SYNTHETIC_VARIANTS = ["no_ac_encoder", "uniform_lag", "no_recon_regularization"]
REALDATA_VARIANTS = ["no_ac_encoder", "uniform_lag", "no_recon_regularization"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete 20-seed CMDL notebook experiments.")
    parser.add_argument(
        "--domain",
        choices=["all", "synthetic", "economics", "energy"],
        default="all",
        help="Limit execution to one domain.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run tasks even when summary.json exists.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional seed subset to run, for parallel partitions.",
    )
    return parser.parse_args()


def summary_exists(run_dir: Path) -> bool:
    return (run_dir / "summary.json").exists()


def should_run(run_dir: Path, force: bool) -> bool:
    return force or not summary_exists(run_dir)


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


def run_task(label: str, run_dir: Path, force: bool, callback) -> None:
    if not should_run(run_dir, force):
        print(f"[skip] {label}: {run_dir}", flush=True)
        return
    print(f"[run] {label}: {run_dir}", flush=True)
    callback()
    cleanup()
    if not summary_exists(run_dir):
        raise RuntimeError(f"Expected summary.json was not created for {label}: {run_dir}")


def synthetic_common_args(output_root: Path) -> dict[str, object]:
    return {
        "scenario": "all",
        "lr": 1e-3,
        "lambda_r": 0.1,
        "temperature": 1.0,
        "omega_transform": "softmax",
        "lambda_omega_entropy": 0.0,
        "omega_entropy_min": None,
        "omega_entropy_max": None,
        "lambda_z_anchor": 0.0,
        "z_anchor_target_sign": 1.0,
        "lag_bias_strength": 1.0,
        "grad_clip": 1.0,
        "val_fraction": 0.2,
        "device": "auto",
        "disable_mlflow": True,
        "epochs": 200,
        "patience": 20,
        "log_every": 10,
        "output_dir": str(output_root),
    }


def economics_common_args(output_root: Path) -> dict[str, object]:
    return {
        "csv_path": str(WORKSPACE_ROOT / "data" / "economics" / "processed" / "economics_cleaned_long_v2.csv"),
        "year_start": 1980,
        "year_end": 2023,
        "train_end_year": 2007,
        "val_end_year": 2013,
        "target_column": "ctfp",
        "feature_bundle": "effective_labor_aware",
        "recon_loss_mode": "anchor_weighted",
        "anchor_recon_weight": 2.0,
        "reconstruction_detach": False,
        "grad_clip_mode": "split",
        "max_missing_share": 0.15,
        "lr": 1e-3,
        "lambda_r": 0.1,
        "temperature": 1.0,
        "omega_transform": "softmax",
        "lambda_omega_entropy": 0.0,
        "omega_entropy_min": None,
        "omega_entropy_max": None,
        "lambda_z_anchor": 0.0,
        "z_anchor_target_sign": 1.0,
        "lag_bias_strength": 1.0,
        "grad_clip": 1.0,
        "device": "auto",
        "disable_mlflow": True,
        "smoke": False,
        "epochs": 120,
        "patience": 20,
        "log_every": 10,
        "output_dir": str(output_root),
    }


def energy_common_args(output_root: Path) -> dict[str, object]:
    return {
        "csv_path": str(WORKSPACE_ROOT / "data" / "energy" / "raw" / "energy_wgi_merged.csv"),
        "year_start": 1996,
        "year_end": 2023,
        "train_end_year": 2011,
        "val_end_year": 2017,
        "treatment_column": "renewables_share_energy",
        "target_column": "co2_per_unit_energy",
        "feature_bundle": "minimal",
        "grad_clip_mode": "global",
        "max_missing_share": 0.15,
        "lr": 1e-3,
        "lambda_r": 0.1,
        "temperature": 1.0,
        "omega_transform": "softmax",
        "lambda_omega_entropy": 0.0,
        "omega_entropy_min": None,
        "omega_entropy_max": None,
        "lambda_z_anchor": 0.0,
        "z_anchor_target_sign": 1.0,
        "lag_bias_strength": 1.0,
        "grad_clip": 1.0,
        "device": "auto",
        "disable_mlflow": True,
        "smoke": False,
        "epochs": 120,
        "patience": 20,
        "log_every": 10,
        "output_dir": str(output_root),
    }


def run_synthetic_suite(force: bool, seeds: list[int]) -> None:
    root = WORKSPACE_ROOT / "outputs" / "notebook_synthetic" / "complete_20seed"
    cmdl_dir = root / "cmdl"
    lstm_dir = root / "plain_lstm"
    ablation_dir = root / "ablation"
    for path in [cmdl_dir, lstm_dir, ablation_dir, root / "comparison", root / "comparison_plots"]:
        path.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        for scenario in SCENARIOS:
            name = f"cmdl_{scenario}_seed{seed}"
            args = Namespace(**synthetic_common_args(cmdl_dir), seed=seed)
            run_task(
                f"synthetic CMDL {scenario} seed {seed}",
                cmdl_dir / name,
                force,
                lambda args=args, name=name, scenario=scenario: run_synthetic.run_experiment(args, name, scenario),
            )

    for seed in seeds:
        for scenario in SCENARIOS:
            name = f"plain_lstm_{scenario}_seed{seed}"
            args = Namespace(**synthetic_common_args(lstm_dir), seed=seed)
            run_task(
                f"synthetic Plain LSTM {scenario} seed {seed}",
                lstm_dir / name,
                force,
                lambda args=args, name=name, scenario=scenario: synthetic_lstm.run_experiment(args, name, scenario),
            )

    ablation_args = Namespace(**synthetic_common_args(ablation_dir), variant="all", seeds=seeds)
    for seed in seeds:
        for scenario in SCENARIOS:
            for variant in SYNTHETIC_VARIANTS:
                name = f"{variant}_{scenario}_seed{seed}"
                run_task(
                    f"synthetic ablation {variant} {scenario} seed {seed}",
                    ablation_dir / name,
                    force,
                    lambda variant=variant, scenario=scenario, seed=seed: synthetic_ablation.run_variant(
                        ablation_args,
                        variant,
                        scenario,
                        seed,
                    ),
                )


def run_economics_suite(force: bool, seeds: list[int]) -> None:
    root = WORKSPACE_ROOT / "outputs" / "notebook_economics" / "complete_20seed"
    cmdl_dir = root / "cmdl"
    lstm_dir = root / "plain_lstm"
    grouped_dir = root / "grouped_ardl"
    ablation_dir = root / "ablation"
    for path in [cmdl_dir, lstm_dir, grouped_dir, ablation_dir, root / "comparison", root / "comparison_plots"]:
        path.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        name = f"economics_cmdl_seed{seed}"
        args = Namespace(**economics_common_args(cmdl_dir), seed=seed, experiment_name=name)
        run_task(
            f"economics CMDL seed {seed}",
            cmdl_dir / name,
            force,
            lambda args=args: run_economics.run_experiment(args),
        )

    for seed in seeds:
        name = f"economics_lstm_seed{seed}"
        args = Namespace(**economics_common_args(lstm_dir), seed=seed, experiment_name=name)
        run_task(
            f"economics Plain LSTM seed {seed}",
            lstm_dir / name,
            force,
            lambda args=args: run_economics_lstm_baseline.run_experiment(args),
        )

    for seed in seeds:
        name = f"economics_grouped_ardl_seed{seed}"
        args = Namespace(**economics_common_args(grouped_dir), seed=seed, experiment_name=name)
        run_task(
            f"economics Grouped ARDL seed {seed}",
            grouped_dir / name,
            force,
            lambda args=args: run_economics_grouped_ardl.run_experiment(args),
        )

    ablation_args = Namespace(
        **economics_common_args(ablation_dir),
        variant="all",
        seeds=seeds,
        experiment_prefix="economics_ablation",
    )
    for seed in seeds:
        for variant in REALDATA_VARIANTS:
            name = f"economics_ablation_{variant}_seed{seed}"
            run_task(
                f"economics ablation {variant} seed {seed}",
                ablation_dir / name,
                force,
                lambda variant=variant, seed=seed: run_economics_ablation.run_variant(ablation_args, variant, seed),
            )


def run_energy_suite(force: bool, seeds: list[int]) -> None:
    root = WORKSPACE_ROOT / "outputs" / "notebook_energy" / "complete_20seed"
    cmdl_dir = root / "cmdl"
    lstm_dir = root / "plain_lstm"
    grouped_dir = root / "grouped_ardl"
    ablation_dir = root / "ablation"
    for path in [cmdl_dir, lstm_dir, grouped_dir, ablation_dir, root / "comparison", root / "comparison_plots"]:
        path.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        name = f"energy_cmdl_seed{seed}"
        args = Namespace(**energy_common_args(cmdl_dir), seed=seed, seeds=None, experiment_name=name)
        run_task(
            f"energy CMDL seed {seed}",
            cmdl_dir / name,
            force,
            lambda args=args: run_energy.run_experiment(args),
        )

    for seed in seeds:
        name = f"energy_lstm_seed{seed}"
        args = Namespace(**energy_common_args(lstm_dir), seed=seed, seeds=None, experiment_name=name)
        run_task(
            f"energy Plain LSTM seed {seed}",
            lstm_dir / name,
            force,
            lambda args=args: run_energy_lstm_baseline.run_experiment(args),
        )

    for seed in seeds:
        name = f"energy_grouped_ardl_seed{seed}"
        args = Namespace(**energy_common_args(grouped_dir), seed=seed, experiment_name=name)
        run_task(
            f"energy Grouped ARDL seed {seed}",
            grouped_dir / name,
            force,
            lambda args=args: run_energy_grouped_ardl.run_experiment(args),
        )

    ablation_args = Namespace(
        **energy_common_args(ablation_dir),
        variant="all",
        seeds=seeds,
        experiment_prefix="energy_ablation",
    )
    for seed in seeds:
        for variant in REALDATA_VARIANTS:
            name = f"energy_ablation_{variant}_seed{seed}"
            run_task(
                f"energy ablation {variant} seed {seed}",
                ablation_dir / name,
                force,
                lambda variant=variant, seed=seed: run_energy_ablation.run_variant(ablation_args, variant, seed),
            )


def main() -> None:
    args = parse_args()
    seeds = sorted({int(seed) for seed in (args.seeds or SEEDS)})
    invalid_seeds = [seed for seed in seeds if seed not in SEEDS]
    if invalid_seeds:
        raise ValueError(f"Seeds outside the supported complete suite range: {invalid_seeds}")
    if args.domain in {"all", "synthetic"}:
        run_synthetic_suite(args.force, seeds)
    if args.domain in {"all", "economics"}:
        run_economics_suite(args.force, seeds)
    if args.domain in {"all", "energy"}:
        run_energy_suite(args.force, seeds)
    print("Complete 20-seed suite finished.", flush=True)


if __name__ == "__main__":
    main()