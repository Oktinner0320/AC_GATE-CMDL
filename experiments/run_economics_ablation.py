"""Economics-domain ablation runner for the three core CMDL variants.

该脚本复用 economics 真实数据训练与评估链，只替换模型或损失的关键部件，
用于衡量 AC encoder、lag gate 和 reconstruction regularization 的贡献。
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import pandas as pd
import torch

from config.cmdl_config import CMDLConfig
from data.economics.economics_loader import DEFAULT_ECONOMICS_FEATURE_BUNDLE, SUPPORTED_ECONOMICS_FEATURE_BUNDLES
from experiments.run_ablation import NoACEncoderCMDLModel, UniformLagCMDLModel
from experiments.run_economics import (
    GRAD_CLIP_MODE_CHOICES,
    RECON_LOSS_MODE_CHOICES,
    evaluate,
    finish_mlflow,
    log_mlflow_metrics,
    prefix_metrics,
    refit_proxy_reconstructor,
    save_json,
    save_predictions,
    setup_experiment,
    summarize_run,
    train_one_epoch,
    try_start_mlflow,
)
from model.cmdl_model import CMDLModel
from model.loss import DomainAgnosticLoss


VARIANT_CHOICES = ["all", "no_ac_encoder", "uniform_lag", "no_recon_regularization"]


def _ablation_diagnostics(variant: str, matched_init_to_full_cmdl: bool) -> dict[str, Any]:
    """Describe whether one ablation run is a clean causal comparison to full CMDL."""

    same_architecture_as_full_cmdl = variant == "no_recon_regularization"
    if variant == "no_recon_regularization" and matched_init_to_full_cmdl:
        causal_ablation_validity = "matched_init_effective_lambda_only"
    elif same_architecture_as_full_cmdl:
        causal_ablation_validity = "same_architecture_unmatched_init"
    else:
        causal_ablation_validity = "architecture_changed"

    return {
        "same_architecture_as_full_cmdl": same_architecture_as_full_cmdl,
        "matched_init_to_full_cmdl": matched_init_to_full_cmdl,
        "causal_ablation_validity": causal_ablation_validity,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the economics ablation suite.

    解析 economics ablation 运行所需的命令行参数。
    """

    economics_defaults = CMDLConfig.from_domain("economics")

    parser = argparse.ArgumentParser(description="Run the economics-domain CMDL ablation suite.")
    parser.add_argument("--variant", choices=VARIANT_CHOICES, default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[economics_defaults.seed])
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument("--year-start", type=int, default=1980)
    parser.add_argument("--year-end", type=int, default=2023)
    parser.add_argument("--train-end-year", type=int, default=2007)
    parser.add_argument("--val-end-year", type=int, default=2013)
    parser.add_argument("--target-column", type=str, default="ctfp")
    parser.add_argument(
        "--feature-bundle",
        type=str,
        choices=list(SUPPORTED_ECONOMICS_FEATURE_BUNDLES),
        default=DEFAULT_ECONOMICS_FEATURE_BUNDLE,
    )
    parser.add_argument("--max-missing-share", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lambda-r", dest="lambda_r", type=float, default=economics_defaults.lambda_r)
    parser.add_argument("--temperature", type=float, default=economics_defaults.temperature)
    parser.add_argument("--omega-transform", choices=["softmax", "sparsemax"], default=economics_defaults.omega_transform)
    parser.add_argument("--lambda-omega-entropy", type=float, default=economics_defaults.lambda_omega_entropy)
    parser.add_argument("--omega-entropy-min", type=float, default=economics_defaults.omega_entropy_min)
    parser.add_argument("--omega-entropy-max", type=float, default=economics_defaults.omega_entropy_max)
    parser.add_argument("--lambda-z-anchor", type=float, default=economics_defaults.lambda_z_anchor)
    parser.add_argument("--z-anchor-target-sign", type=float, default=economics_defaults.z_anchor_target_sign)
    parser.add_argument("--lag-bias-strength", type=float, default=economics_defaults.lag_bias_strength)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-clip-mode", choices=GRAD_CLIP_MODE_CHOICES, default="global")
    parser.add_argument("--recon-loss-mode", choices=RECON_LOSS_MODE_CHOICES, default="all")
    parser.add_argument("--anchor-recon-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-detach", dest="reconstruction_detach", action="store_true", default=True)
    parser.add_argument("--no-reconstruction-detach", dest="reconstruction_detach", action="store_false")
    parser.add_argument("--output-dir", type=str, default="outputs/step5/economics_ablation")
    parser.add_argument("--experiment-prefix", type=str, default="E4_economics_ablation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="Run one epoch per variant instead of a full suite.")
    return parser.parse_args()


def selected_variants(variant: str) -> list[str]:
    """Expand the CLI variant selector into concrete ablation variants.

    将命令行的 variant 选择器展开为具体变体列表。
    """

    if variant == "all":
        return [choice for choice in VARIANT_CHOICES if choice != "all"]
    return [variant]


def build_variant_model(variant: str, cfg: CMDLConfig) -> tuple[CMDLModel, float]:
    """Construct the requested ablation model and its effective lambda_r.

    构造指定 ablation 变体，并返回应使用的 reconstruction loss 权重。
    """

    if variant == "no_ac_encoder":
        return NoACEncoderCMDLModel(cfg), 0.0
    if variant == "uniform_lag":
        return UniformLagCMDLModel(cfg), cfg.lambda_r
    if variant == "no_recon_regularization":
        return CMDLModel(cfg), 0.0
    raise ValueError(f"Unsupported ablation variant: {variant}")


def prepare_variant_setup(
    args: argparse.Namespace,
    variant: str,
    seed: int,
) -> tuple[Any, argparse.Namespace, float, dict[str, Any]]:
    """Prepare one economics ablation setup by reusing the standard runner.

    复用标准 economics runner，准备单个 ablation 变体的运行时对象。
    """

    variant_args = copy.deepcopy(args)
    variant_args.seed = int(seed)
    variant_args.experiment_name = f"{args.experiment_prefix}_{variant}_seed{seed}"
    setup = setup_experiment(variant_args)
    if variant == "no_recon_regularization":
        model = setup.model
        effective_lambda_r = 0.0
        matched_init_to_full_cmdl = True
    else:
        model, effective_lambda_r = build_variant_model(variant, setup.cfg)
        matched_init_to_full_cmdl = False
    setup.model = model.to(setup.device)
    setup.criterion = DomainAgnosticLoss(
        lambda_r=effective_lambda_r,
        warmup_steps=setup.cfg.max_lag,
        recon_loss_mode=getattr(variant_args, "recon_loss_mode", "all"),
        anchor_proxy_index=int(setup.full_panel.metadata.get("anchor_proxy_index", 0)),
        anchor_recon_weight=float(getattr(variant_args, "anchor_recon_weight", 1.0)),
        lambda_omega_entropy=getattr(variant_args, "lambda_omega_entropy", 0.0),
        omega_entropy_min=getattr(variant_args, "omega_entropy_min", None),
        omega_entropy_max=getattr(variant_args, "omega_entropy_max", None),
        lambda_z_anchor=getattr(variant_args, "lambda_z_anchor", 0.0),
        z_anchor_target_sign=getattr(variant_args, "z_anchor_target_sign", 1.0),
    )
    setup.optimizer = torch.optim.Adam(setup.model.parameters(), lr=variant_args.lr)
    return setup, variant_args, effective_lambda_r, _ablation_diagnostics(variant, matched_init_to_full_cmdl)


def run_variant(args: argparse.Namespace, variant: str, seed: int) -> dict[str, Any]:
    """Train and evaluate one economics ablation variant for one seed.

    训练并评估单个 economics ablation 变体与随机种子。
    """

    setup, variant_args, effective_lambda_r, ablation_diagnostics = prepare_variant_setup(
        args=args,
        variant=variant,
        seed=seed,
    )
    save_json(setup.run_dir / "args.json", vars(variant_args))

    tracking_backend = try_start_mlflow(
        args=variant_args,
        output_root=Path(args.output_dir).resolve(),
        run_name=variant_args.experiment_name,
        params={
            "experiment": variant_args.experiment_name,
            "model": "cmdl_ablation",
            "variant": variant,
            "target_column": variant_args.target_column,
            "feature_bundle": getattr(variant_args, "feature_bundle", DEFAULT_ECONOMICS_FEATURE_BUNDLE),
            "seed": variant_args.seed,
            "lr": variant_args.lr,
            "epochs": variant_args.epochs,
            "patience": variant_args.patience,
            "lambda_r": variant_args.lambda_r,
            "effective_lambda_r": effective_lambda_r,
            "matched_init_to_full_cmdl": ablation_diagnostics["matched_init_to_full_cmdl"],
            "causal_ablation_validity": ablation_diagnostics["causal_ablation_validity"],
            "temperature": variant_args.temperature,
            "omega_transform": getattr(variant_args, "omega_transform", "softmax"),
            "lambda_omega_entropy": getattr(variant_args, "lambda_omega_entropy", 0.0),
            "omega_entropy_min": getattr(variant_args, "omega_entropy_min", None),
            "omega_entropy_max": getattr(variant_args, "omega_entropy_max", None),
            "lambda_z_anchor": getattr(variant_args, "lambda_z_anchor", 0.0),
            "z_anchor_target_sign": getattr(variant_args, "z_anchor_target_sign", 1.0),
            "lag_bias_strength": variant_args.lag_bias_strength,
            "grad_clip": variant_args.grad_clip,
            "grad_clip_mode": getattr(variant_args, "grad_clip_mode", "global"),
            "recon_loss_mode": getattr(variant_args, "recon_loss_mode", "all"),
            "anchor_recon_weight": getattr(variant_args, "anchor_recon_weight", 1.0),
            "reconstruction_detach": getattr(variant_args, "reconstruction_detach", True),
            "year_start": variant_args.year_start,
            "year_end": variant_args.year_end,
            "train_end_year": variant_args.train_end_year,
            "val_end_year": variant_args.val_end_year,
            "max_missing_share": variant_args.max_missing_share,
        },
    )

    history: list[dict[str, float]] = []
    best_epoch = 0
    best_val_task_loss = float("inf")
    patience_counter = 0
    best_state = copy.deepcopy(setup.model.state_dict())

    try:
        for epoch in range(1, variant_args.epochs + 1):
            train_metrics = train_one_epoch(
                model=setup.model,
                criterion=setup.criterion,
                optimizer=setup.optimizer,
                panel=setup.train_panel,
                grad_clip=variant_args.grad_clip,
                grad_clip_mode=getattr(variant_args, "grad_clip_mode", "global"),
            )
            val_metrics, _ = evaluate(setup.model, setup.criterion, setup.val_panel)

            epoch_record: dict[str, float] = {"epoch": float(epoch)}
            epoch_record.update(prefix_metrics("train", train_metrics))
            epoch_record.update(prefix_metrics("val", val_metrics))
            history.append(epoch_record)
            log_mlflow_metrics({key: value for key, value in epoch_record.items() if key != "epoch"}, step=epoch)

            if epoch == 1 or (variant_args.log_every > 0 and epoch % variant_args.log_every == 0):
                print(
                    f"[{variant_args.experiment_name}] epoch={epoch:03d} "
                    f"train_total={train_metrics['total_loss']:.4f} "
                    f"val_task={val_metrics['task_loss']:.4f} "
                    f"val_r2={val_metrics['r2']:.4f} "
                    f"val_proxy_r2={val_metrics['proxy_recon_r2']:.4f}"
                )

            if val_metrics["task_loss"] < best_val_task_loss - 1e-8:
                best_val_task_loss = float(val_metrics["task_loss"])
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(setup.model.state_dict())
                torch.save(
                    {
                        "experiment": variant_args.experiment_name,
                        "model": "cmdl_ablation",
                        "variant": variant,
                        "best_epoch": best_epoch,
                        "best_val_task_loss": best_val_task_loss,
                        "effective_lambda_r": float(effective_lambda_r),
                        "ablation_diagnostics": dict(ablation_diagnostics),
                        "config": setup.cfg.to_dict(),
                        "model_state_dict": best_state,
                    },
                    setup.checkpoint_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= variant_args.patience:
                    print(f"[{variant_args.experiment_name}] early stopping at epoch {epoch}")
                    break

        setup.model.load_state_dict(best_state)
        proxy_refit_result = refit_proxy_reconstructor(setup.model, setup.train_panel)
        if not proxy_refit_result.applied:
            print(
                f"[{variant_args.experiment_name}] proxy refit skipped: "
                f"{proxy_refit_result.reason} (rank={proxy_refit_result.design_rank}/"
                f"{proxy_refit_result.design_columns})"
            )
        torch.save(
            {
                "experiment": variant_args.experiment_name,
                "model": "cmdl_ablation",
                "variant": variant,
                "best_epoch": best_epoch,
                "best_val_task_loss": best_val_task_loss,
                "effective_lambda_r": float(effective_lambda_r),
                "ablation_diagnostics": dict(ablation_diagnostics),
                "config": setup.cfg.to_dict(),
                "proxy_head_refit": proxy_refit_result.applied,
                "proxy_refit": proxy_refit_result.to_dict(),
                "model_state_dict": copy.deepcopy(setup.model.state_dict()),
            },
            setup.checkpoint_path,
        )

        train_metrics, _ = evaluate(
            setup.model,
            setup.criterion,
            setup.train_panel,
            proxy_refit_result=proxy_refit_result,
        )
        val_metrics, _ = evaluate(
            setup.model,
            setup.criterion,
            setup.val_panel,
            proxy_refit_result=proxy_refit_result,
        )
        test_metrics, outputs = evaluate(
            setup.model,
            setup.criterion,
            setup.test_panel,
            include_outputs=True,
            proxy_refit_result=proxy_refit_result,
        )
        if outputs is None:
            raise RuntimeError("Expected test outputs from economics ablation evaluation")

        pd.DataFrame(history).to_csv(setup.history_csv_path, index=False)
        save_json(setup.history_json_path, history)
        save_predictions(outputs, setup.predictions_path)

        summary = summarize_run(
            setup=setup,
            args=variant_args,
            tracking_backend=tracking_backend,
            best_epoch=best_epoch,
            best_val_task_loss=best_val_task_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            proxy_refit_result=proxy_refit_result,
        )
        summary["model"] = "cmdl_ablation"
        summary["variant"] = variant
        summary["effective_lambda_r"] = float(effective_lambda_r)
        summary.setdefault("diagnostics", {})["ablation"] = dict(ablation_diagnostics)
        save_json(setup.summary_path, summary)
        return summary
    finally:
        finish_mlflow(
            [
                setup.run_dir / "args.json",
                setup.history_json_path,
                setup.history_csv_path,
                setup.summary_path,
                setup.predictions_path,
                setup.checkpoint_path,
            ]
        )


def aggregate_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate repeated-seed economics ablation results into mean/std tables.

    将多随机种子的 economics ablation 结果聚合为 mean/std 表。
    """

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    group_columns = [column for column in ["variant", "target_column"] if column in frame.columns]
    non_numeric_columns = {
        "experiment",
        "model",
        "variant",
        "target_column",
        "seed",
        "run_dir",
        "source_path",
        "proxy_refit_status",
        "proxy_refit_reason",
        "causal_ablation_validity",
    }
    numeric_columns = [
        column
        for column in frame.columns
        if column not in non_numeric_columns and pd.api.types.is_numeric_dtype(frame[column])
    ]
    aggregated = frame.groupby(group_columns, as_index=False)[numeric_columns].agg(["mean", "std"])
    aggregated.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in aggregated.columns.to_flat_index()
    ]
    return aggregated


def run_suite(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the requested economics ablation suite and collect summary tables.

    按参数执行 economics ablation，并输出逐 run 与聚合汇总结果。
    """

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for variant in selected_variants(args.variant):
        for seed in args.seeds:
            summary = run_variant(args=args, variant=variant, seed=seed)
            summary_rows.append(
                {
                    "experiment": summary["experiment"],
                    "model": summary.get("model", "cmdl_ablation"),
                    "variant": variant,
                    "target_column": summary["data"]["target_column"],
                    "source_path": summary["data"]["source_path"],
                    "seed": summary["config"]["seed"],
                    "effective_lambda_r": summary.get("effective_lambda_r"),
                    "matched_init_to_full_cmdl": summary.get("diagnostics", {})
                    .get("ablation", {})
                    .get("matched_init_to_full_cmdl"),
                    "causal_ablation_validity": summary.get("diagnostics", {})
                    .get("ablation", {})
                    .get("causal_ablation_validity"),
                    "best_epoch": summary["best_epoch"],
                    "best_val_task_loss": summary["best_val_task_loss"],
                    "proxy_refit_status": summary.get("diagnostics", {}).get("proxy_refit", {}).get("status"),
                    "proxy_refit_applied": summary.get("diagnostics", {}).get("proxy_refit", {}).get("applied"),
                    "proxy_metric_interpretable": summary.get("diagnostics", {})
                    .get("proxy_refit", {})
                    .get("metrics_interpretable"),
                    "proxy_refit_reason": summary.get("diagnostics", {}).get("proxy_refit", {}).get("reason"),
                    "grad_clip_mode": summary.get("diagnostics", {})
                    .get("training_controls", {})
                    .get("grad_clip_mode"),
                    "run_dir": str(output_root / summary["experiment"]),
                    **prefix_metrics("train", summary["metrics"]["train"]),
                    **prefix_metrics("val", summary["metrics"]["val"]),
                    **prefix_metrics("test", summary["metrics"]["test"]),
                }
            )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_root / "ablation_results.csv", index=False)
    save_json(output_root / "ablation_results.json", summary_rows)

    aggregated = aggregate_results(summary_rows)
    if not aggregated.empty:
        aggregated.to_csv(output_root / "ablation_results_aggregated.csv", index=False)

    return summary_frame, aggregated


def main() -> None:
    """Execute the requested economics ablation runs and print summary tables.

    按参数执行 economics ablation，并打印逐 run 与聚合摘要。
    """

    args = parse_args()
    summary_frame, aggregated = run_suite(args)

    print("Economics ablation experiments complete.")
    if not summary_frame.empty:
        print(summary_frame.to_string(index=False))
    if not aggregated.empty:
        print("\nAggregated across seeds:")
        print(aggregated.to_string(index=False))


if __name__ == "__main__":
    main()