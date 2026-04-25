"""Run the grouped ARDL-style baseline on the energy panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from baselines.grouped_ardl import evaluate_grouped_ardl_baseline
from config.cmdl_config import CMDLConfig
from data.energy.energy_loader import (
    DEFAULT_ENERGY_FEATURE_BUNDLE,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TREATMENT_COLUMN,
    DEFAULT_YEAR_END,
    DEFAULT_YEAR_START,
    SUPPORTED_ENERGY_FEATURE_BUNDLES,
    build_temporal_splits,
    load_energy_panel,
)
from evaluation.realdata_diagnostics import proxy_metadata_payload


def parse_args() -> argparse.Namespace:
    defaults = CMDLConfig.from_domain("energy")
    parser = argparse.ArgumentParser(description="Run grouped ARDL energy baseline.")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--year-start", type=int, default=DEFAULT_YEAR_START)
    parser.add_argument("--year-end", type=int, default=DEFAULT_YEAR_END)
    parser.add_argument("--train-end-year", type=int, default=2011)
    parser.add_argument("--val-end-year", type=int, default=2017)
    parser.add_argument("--treatment-column", type=str, default=DEFAULT_TREATMENT_COLUMN)
    parser.add_argument("--target-column", type=str, default=DEFAULT_TARGET_COLUMN)
    parser.add_argument(
        "--feature-bundle",
        type=str,
        choices=list(SUPPORTED_ENERGY_FEATURE_BUNDLES),
        default=DEFAULT_ENERGY_FEATURE_BUNDLE,
    )
    parser.add_argument("--max-missing-share", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", type=str, default="outputs/step5/energy_grouped_ardl")
    parser.add_argument("--experiment-name", type=str, default="E3_energy_grouped_ardl")
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)


def _strip_baseline_prefix(metrics: dict[str, float]) -> dict[str, float]:
    prefix = "baseline_grouped_ardl_"
    return {key.removeprefix(prefix): float(value) for key, value in metrics.items() if key.startswith(prefix)}


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    panel = load_energy_panel(
        csv_path=args.csv_path,
        treatment_column=args.treatment_column,
        target_column=args.target_column,
        feature_bundle=args.feature_bundle,
        year_start=args.year_start,
        year_end=args.year_end,
        stats_end_year=args.train_end_year,
        max_missing_share=args.max_missing_share,
    )
    cfg = CMDLConfig.from_domain(
        "energy",
        seed=args.seed,
        n_entities=panel.X_it.shape[0],
        seq_length=panel.X_it.shape[1],
        seq_features=panel.X_it.shape[2],
        n_proxies=panel.p_i.shape[1],
        static_dim=panel.s_i.shape[1],
    )
    train_panel, val_panel, test_panel = build_temporal_splits(
        panel=panel,
        max_lag=cfg.max_lag,
        train_end_year=args.train_end_year,
        val_end_year=args.val_end_year,
    )
    metrics = {
        "train": _strip_baseline_prefix(evaluate_grouped_ardl_baseline(train_panel, train_panel, cfg.max_lag)),
        "val": _strip_baseline_prefix(evaluate_grouped_ardl_baseline(train_panel, val_panel, cfg.max_lag)),
        "test": _strip_baseline_prefix(evaluate_grouped_ardl_baseline(train_panel, test_panel, cfg.max_lag)),
    }
    summary = {
        "experiment": args.experiment_name,
        "model": "grouped_ardl",
        "tracking_backend": "json",
        "device": "cpu",
        "best_epoch": 0,
        "best_val_task_loss": metrics["val"].get("mse"),
        "config": cfg.to_dict(),
        "data": {
            "source_path": panel.metadata["source_path"],
            "treatment_column": panel.metadata["treatment_column"],
            "target_column": panel.metadata["target_column"],
            "feature_bundle": panel.metadata["feature_bundle"],
            "seq_feature_columns": list(panel.metadata["seq_feature_columns"]),
            "proxy_columns": list(panel.metadata["proxy_columns"]),
            **proxy_metadata_payload(panel.metadata, cfg.n_proxies),
            "static_columns": list(panel.metadata["static_columns"]),
            "stats_end_year": int(panel.metadata["stats_end_year"]),
            "year_start": int(args.year_start),
            "year_end": int(args.year_end),
            "train_end_year": int(args.train_end_year),
            "val_end_year": int(args.val_end_year),
            "n_entities": int(cfg.n_entities),
            "full_seq_length": int(cfg.seq_length),
            "train_years": list(train_panel.metadata["years"]),
            "val_years": list(val_panel.metadata["years"]),
            "test_years": list(test_panel.metadata["years"]),
        },
        "metrics": metrics,
    }
    run_dir = Path(args.output_dir).resolve() / args.experiment_name
    save_json(run_dir / "args.json", vars(args))
    save_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    summary = run_experiment(parse_args())
    test_metrics = summary["metrics"]["test"]
    print(
        "Energy grouped ARDL complete. "
        f"test_r2={test_metrics.get('r2'):.4f} "
        f"best_lag_mean={test_metrics.get('best_lag_mean')}"
    )


if __name__ == "__main__":
    main()