"""Run the isolated Informal RQ experiment suite."""

from __future__ import annotations

import argparse
import gc
import sys
from argparse import Namespace
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
RQ_RES_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = RQ_RES_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(RQ_RES_ROOT) not in sys.path:
    sys.path.insert(0, str(RQ_RES_ROOT))

from informal_acgate.aggregate import aggregate_summaries
from informal_acgate.runner import run_experiment


DEFAULT_OUTPUT_DIR = RQ_RES_ROOT / "outputs" / "informal_acgate"
DEFAULT_SEEDS = list(range(20))
ABLATION_VARIANTS = ["no_ac_encoder", "uniform_lag", "no_recon_regularization"]
SCENARIO_CHOICES = ("overlap_region_proxy", "fullspan_region_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated Informal RQ experiments under RQ_res.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="overlap_region_proxy")
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument("--stats-end-year", type=int, default=None)
    parser.add_argument("--train-end-year", type=int, default=None)
    parser.add_argument("--val-end-year", type=int, default=None)
    parser.add_argument("--max-lag", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--smoke", action="store_true", help="Run one epoch per task.")
    parser.add_argument("--force", action="store_true", help="Re-run tasks even when summary.json exists.")
    return parser.parse_args()


def common_args(args: argparse.Namespace, seed: int, output_dir: Path) -> dict[str, object]:
    if args.scenario == "fullspan_region_proxy":
        year_start = 2006 if args.year_start is None else args.year_start
        year_end = 2023 if args.year_end is None else args.year_end
        stats_end_year = 2018 if args.stats_end_year is None else args.stats_end_year
        train_end_year = 2018 if args.train_end_year is None else args.train_end_year
        val_end_year = 2020 if args.val_end_year is None else args.val_end_year
        max_lag = 3 if args.max_lag is None else args.max_lag
    else:
        year_start = args.year_start
        year_end = args.year_end
        stats_end_year = args.stats_end_year
        train_end_year = 2021 if args.train_end_year is None else args.train_end_year
        val_end_year = 2022 if args.val_end_year is None else args.val_end_year
        max_lag = 2 if args.max_lag is None else args.max_lag

    return {
        "csv_path": None,
        "year_start": year_start,
        "year_end": year_end,
        "stats_end_year": stats_end_year,
        "train_end_year": train_end_year,
        "val_end_year": val_end_year,
        "missing_policy": "error",
        "seed": seed,
        "max_lag": max_lag,
        "d_model": 32,
        "lstm_layers": 1,
        "dropout": 0.05,
        "lr": 1e-3,
        "epochs": 1 if args.smoke else args.epochs,
        "patience": 1 if args.smoke else args.patience,
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
        "grad_clip_mode": "global",
        "recon_loss_mode": "all",
        "anchor_recon_weight": 1.0,
        "reconstruction_detach": True,
        "output_dir": str(output_dir),
        "device": args.device,
        "log_every": 10,
        "smoke": args.smoke,
    }


def cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def should_run(run_dir: Path, force: bool) -> bool:
    return force or not (run_dir / "summary.json").exists()


def run_task(label: str, task_args: Namespace, force: bool) -> None:
    run_dir = Path(task_args.output_dir).resolve() / str(task_args.experiment_name)
    if not should_run(run_dir, force):
        print(f"[skip] {label}: {run_dir}", flush=True)
        return
    print(f"[run] {label}: {run_dir}", flush=True)
    run_experiment(task_args)
    cleanup()
    if not (run_dir / "summary.json").exists():
        raise RuntimeError(f"Expected summary.json was not created for {label}: {run_dir}")


def build_tasks(args: argparse.Namespace, output_dir: Path) -> list[tuple[str, Namespace]]:
    tasks: list[tuple[str, Namespace]] = []
    for seed in args.seeds:
        base = common_args(args, seed, output_dir)
        if args.scenario == "fullspan_region_proxy":
            tasks.extend(
                [
                    (
                        f"single full-span income region-proxy CMDL seed {seed}",
                        Namespace(
                            **base,
                            feature_bundle="single_fullspan_region_proxy",
                            model="cmdl",
                            ablation="none",
                            experiment_name=f"single_fullspan_region_proxy_cmdl_seed{seed}",
                        ),
                    ),
                    (
                        f"single full-span income region-proxy plain LSTM seed {seed}",
                        Namespace(
                            **base,
                            feature_bundle="single_fullspan_region_proxy",
                            model="plain_lstm",
                            ablation="none",
                            experiment_name=f"single_fullspan_region_proxy_plain_lstm_seed{seed}",
                        ),
                    ),
                ]
            )
            for variant in ABLATION_VARIANTS:
                tasks.append(
                    (
                        f"single full-span income ablation {variant} seed {seed}",
                        Namespace(
                            **base,
                            feature_bundle="single_fullspan_region_proxy",
                            model="cmdl",
                            ablation=variant,
                            experiment_name=f"single_fullspan_region_proxy_{variant}_seed{seed}",
                        ),
                    )
                )
            continue

        tasks.extend(
            [
                (
                    f"single-feature region-proxy CMDL seed {seed}",
                    Namespace(
                        **base,
                        feature_bundle="single_overlap_region_proxy",
                        model="cmdl",
                        ablation="none",
                        experiment_name=f"single_feature_region_proxy_cmdl_seed{seed}",
                    ),
                ),
                (
                    f"multiseq overlap region-proxy CMDL seed {seed}",
                    Namespace(
                        **base,
                        feature_bundle="multiseq_overlap_region_proxy",
                        model="cmdl",
                        ablation="none",
                        experiment_name=f"multiseq_overlap_region_proxy_cmdl_seed{seed}",
                    ),
                ),
                (
                    f"multiseq overlap region-proxy plain LSTM seed {seed}",
                    Namespace(
                        **base,
                        feature_bundle="multiseq_overlap_region_proxy",
                        model="plain_lstm",
                        ablation="none",
                        experiment_name=f"multiseq_overlap_region_proxy_plain_lstm_seed{seed}",
                    ),
                ),
            ]
        )
        for variant in ABLATION_VARIANTS:
            tasks.append(
                (
                    f"multiseq overlap ablation {variant} seed {seed}",
                    Namespace(
                        **base,
                        feature_bundle="multiseq_overlap_region_proxy",
                        model="cmdl",
                        ablation=variant,
                        experiment_name=f"multiseq_overlap_region_proxy_{variant}_seed{seed}",
                    ),
                )
            )
    return tasks


def main() -> None:
    args = parse_args()
    if args.scenario == "fullspan_region_proxy":
        suite_name = "smoke_fullspan_region_proxy" if args.smoke else "suite_fullspan_region_proxy"
    else:
        suite_name = "smoke_region_proxy" if args.smoke else "suite_region_proxy"
    output_dir = Path(args.output_dir).resolve() / suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, task_args in build_tasks(args, output_dir):
        run_task(label, task_args, args.force)
    comparison = aggregate_summaries(output_dir=output_dir, output_csv=output_dir / "comparison.csv")
    print(f"Suite complete. Aggregated {len(comparison)} runs into {output_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
