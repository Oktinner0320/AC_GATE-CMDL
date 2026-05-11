"""Synthetic plain-LSTM baseline runner for Step 4.5 comparisons.

该脚本复用当前 Step 4 的合成数据、切分与日志习惯，
并额外提供基于 lag occlusion 的 post-hoc 有效滞后解释。
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from baselines.lstm_baseline import PlainLSTMBaseline, PlainLSTMBaselineOutput
from config.cmdl_config import CMDLConfig
from data.synthetic.generate import SyntheticPanel, generate_cmdl_synthetic
from evaluation.kstar_eval import evaluate_kstar, evaluate_omega_distribution
from evaluation.metrics import compute_mae, compute_mse, compute_r2
from experiments._checkpoint_io import save_torch_checkpoint
from experiments._runtime_meta import attach_runtime_metadata, start_runtime_timer
from experiments.run_synthetic import (
    finish_mlflow,
    log_mlflow_metrics,
    move_panel_to_device,
    save_json,
    select_device,
    set_seed,
    split_entity_indices,
    subset_panel,
    try_start_mlflow,
)
from visualization.kstar_distribution import plot_kstar_scatter
from visualization.omega_heatmap import plot_omega_heatmap


@dataclass(slots=True)
class BaselineExperimentSetup:
    """Bundle runtime objects required by one baseline experiment.

    汇总单次 plain-LSTM baseline 运行所需的对象。
    """

    cfg: CMDLConfig
    full_panel: SyntheticPanel
    train_panel: SyntheticPanel
    val_panel: SyntheticPanel
    model: PlainLSTMBaseline
    optimizer: torch.optim.Optimizer
    device: torch.device
    run_dir: Path
    checkpoint_path: Path
    history_json_path: Path
    history_csv_path: Path
    summary_path: Path
    predictions_path: Path


@dataclass(slots=True)
class BaselineExperimentResult:
    """Store final artifacts and metrics returned by a baseline run.

    保存单次 baseline 运行的最终指标和输出。
    """

    experiment_name: str
    scenario: str
    run_dir: Path
    best_epoch: int
    best_val_task_loss: float
    tracking_backend: str
    history: list[dict[str, float]]
    metrics: dict[str, float]
    outputs: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the synthetic plain-LSTM baseline.

    解析 synthetic plain-LSTM baseline 的命令行参数。
    """

    synthetic_defaults = CMDLConfig.from_domain("synthetic")

    parser = argparse.ArgumentParser(description="Run synthetic plain-LSTM baseline experiments.")
    parser.add_argument(
        "--scenario",
        choices=["all", "linear", "nonlinear"],
        default="all",
        help="all runs both synthetic scenarios, otherwise run one scenario only.",
    )
    parser.add_argument("--seed", type=int, default=synthetic_defaults.seed)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="outputs/step4_lstm_baseline")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def aligned_targets(y_true: torch.Tensor, warmup_steps: int) -> torch.Tensor:
    """Trim warm-up steps from targets to match the model outputs.

    去除 warm-up 时间步，使目标张量与 baseline 输出对齐。
    """

    if y_true.dim() != 2:
        raise ValueError(f"Expected y_true with shape [B, T], got {tuple(y_true.shape)}")
    if y_true.size(1) <= warmup_steps:
        raise ValueError("y_true length must be greater than warmup_steps")
    return y_true[:, warmup_steps:]


def setup_experiment(
    args: argparse.Namespace,
    experiment_name: str,
    scenario: str,
) -> BaselineExperimentSetup:
    """Prepare data, model, optimizer, and output layout for one baseline run.

    为单次 baseline 实验准备数据、模型、优化器和输出目录。
    """

    cfg = CMDLConfig.from_domain(
        "synthetic",
        seed=args.seed,
        scenario=scenario,
    )
    set_seed(cfg.seed)

    full_panel_cpu = generate_cmdl_synthetic(cfg)
    train_indices, val_indices = split_entity_indices(cfg.n_entities, args.val_fraction, cfg.seed)
    device = select_device(args.device)

    full_panel = move_panel_to_device(full_panel_cpu, device)
    train_panel = move_panel_to_device(subset_panel(full_panel_cpu, train_indices), device)
    val_panel = move_panel_to_device(subset_panel(full_panel_cpu, val_indices), device)

    model = PlainLSTMBaseline(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = Path(args.output_dir).resolve() / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return BaselineExperimentSetup(
        cfg=cfg,
        full_panel=full_panel,
        train_panel=train_panel,
        val_panel=val_panel,
        model=model,
        optimizer=optimizer,
        device=device,
        run_dir=run_dir,
        checkpoint_path=run_dir / "best_model.pt",
        history_json_path=run_dir / "history.json",
        history_csv_path=run_dir / "history.csv",
        summary_path=run_dir / "summary.json",
        predictions_path=run_dir / "predictions.csv",
    )


def compute_posthoc_lag_profile(
    model: PlainLSTMBaseline,
    panel: SyntheticPanel,
    base_output: PlainLSTMBaselineOutput | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-entity effective lag profiles by lag-wise occlusion.

    用逐 lag 的输入遮挡构造 baseline 的 post-hoc 有效滞后分布，
    并据此得到 pseudo-k*。该分布是解释结果，不是训练得到的原生 omega。
    """

    aligned_y = aligned_targets(panel.Y_it, model.cfg.max_lag)
    model.eval()
    with torch.no_grad():
        if base_output is None:
            base_output = model(
                entity_ids=panel.entity_ids,
                X_it=panel.X_it,
                s_i=panel.s_i,
            )

        base_entity_errors = torch.mean((base_output.y_pred - aligned_y) ** 2, dim=1)
        lag_deltas: list[torch.Tensor] = []

        for lag_index in range(1, model.cfg.max_lag + 1):
            occluded_inputs = panel.X_it.clone()
            start_index = model.cfg.max_lag - lag_index
            end_index = panel.X_it.size(1) - lag_index
            occluded_inputs[:, start_index:end_index, :] = 0.0

            occluded_output = model(
                entity_ids=panel.entity_ids,
                X_it=occluded_inputs,
                s_i=panel.s_i,
            )
            occluded_entity_errors = torch.mean((occluded_output.y_pred - aligned_y) ** 2, dim=1)
            lag_deltas.append((occluded_entity_errors - base_entity_errors).unsqueeze(1))

        raw_profile = torch.cat(lag_deltas, dim=1)
        positive_profile = raw_profile.clamp(min=0.0)
        profile_sums = positive_profile.sum(dim=1, keepdim=True)
        normalized_profile = positive_profile / profile_sums.clamp_min(1e-8)
        uniform_profile = torch.full_like(normalized_profile, 1.0 / model.cfg.max_lag)
        lag_profile = torch.where(profile_sums > 0.0, normalized_profile, uniform_profile)
        lag_indices = torch.arange(
            1,
            model.cfg.max_lag + 1,
            device=lag_profile.device,
            dtype=lag_profile.dtype,
        )
        pseudo_k_star = torch.sum(lag_profile * lag_indices.unsqueeze(0), dim=1)

    return lag_profile, pseudo_k_star


def train_one_epoch(
    model: PlainLSTMBaseline,
    optimizer: torch.optim.Optimizer,
    panel: SyntheticPanel,
    grad_clip: float,
) -> dict[str, float]:
    """Run one optimization epoch on the full synthetic training panel.

    在完整合成训练面板上执行一轮 plain-LSTM baseline 参数更新。
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, s_i=panel.s_i)
    target = aligned_targets(panel.Y_it, model.cfg.max_lag)
    task_loss = F.mse_loss(output.y_pred, target)
    if not torch.isfinite(task_loss):
        raise FloatingPointError("Encountered a non-finite task loss during baseline training")

    task_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    return {
        "task_loss": float(task_loss.item()),
        "grad_norm": float(grad_norm),
    }


def evaluate(
    model: PlainLSTMBaseline,
    panel: SyntheticPanel,
    include_outputs: bool = False,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    """Evaluate one synthetic panel and optionally return serialized outputs.

    评估一个 synthetic panel，并在需要时返回用于可视化和落盘的输出。
    """

    model.eval()
    with torch.no_grad():
        output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, s_i=panel.s_i)
        target = aligned_targets(panel.Y_it, model.cfg.max_lag)
        task_loss = F.mse_loss(output.y_pred, target)

    lag_profile, pseudo_k_star = compute_posthoc_lag_profile(model, panel, base_output=output)

    metrics = {
        "task_loss": float(task_loss.item()),
        "task_mse": compute_mse(output.y_pred, target),
        "task_mae": compute_mae(output.y_pred, target),
        "task_r2": compute_r2(output.y_pred, target),
    }
    metrics.update(
        {
            f"posthoc_kstar_{key}": value
            for key, value in evaluate_kstar(pseudo_k_star, panel.kstar_true).items()
        }
    )
    metrics.update(
        {
            f"posthoc_profile_{key}": value
            for key, value in evaluate_omega_distribution(lag_profile, panel.kstar_true).items()
        }
    )

    if not include_outputs:
        return metrics, None

    outputs = {
        "entity_ids": panel.entity_ids.detach().cpu().numpy(),
        "y_pred": output.y_pred.detach().cpu().numpy(),
        "y_true": target.detach().cpu().numpy(),
        "lag_profile": lag_profile.detach().cpu().numpy(),
        "posthoc_k_star": pseudo_k_star.detach().cpu().numpy(),
        "kstar_true": panel.kstar_true.detach().cpu().numpy(),
        "z_true": panel.z_true.detach().cpu().numpy(),
    }
    return metrics, outputs


def save_predictions(outputs: dict[str, np.ndarray], path: Path) -> None:
    """Persist entity-level baseline predictions and lag profiles to CSV.

    保存 baseline 的实体级预测结果和 post-hoc lag profile。
    """

    lag_profile = outputs["lag_profile"]
    dataframe = pd.DataFrame(
        {
            "entity_id": outputs["entity_ids"].astype(int),
            "kstar_true": outputs["kstar_true"],
            "posthoc_kstar_pred": outputs["posthoc_k_star"],
            "lag_profile_peak": np.argmax(lag_profile, axis=1) + 1,
        }
    )
    for lag_index in range(lag_profile.shape[1]):
        dataframe[f"lag_profile_{lag_index + 1}"] = lag_profile[:, lag_index]

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def summarize_run(
    setup: BaselineExperimentSetup,
    experiment_name: str,
    tracking_backend: str,
    best_epoch: int,
    best_val_task_loss: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Build the JSON summary for one completed baseline run.

    生成单次 baseline 运行的 JSON 汇总结果。
    """

    return {
        "experiment": experiment_name,
        "model": "plain_lstm",
        "scenario": setup.cfg.scenario,
        "tracking_backend": tracking_backend,
        "device": str(setup.device),
        "best_epoch": int(best_epoch),
        "best_val_task_loss": float(best_val_task_loss),
        "posthoc_lag_method": "lag_occlusion",
        "config": setup.cfg.to_dict(),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }


def run_experiment(
    args: argparse.Namespace,
    experiment_name: str,
    scenario: str,
) -> BaselineExperimentResult:
    """Train, evaluate, visualize, and log one synthetic plain-LSTM baseline run.

    完成单次 synthetic plain-LSTM baseline 的训练、评估、可视化和日志记录。
    """

    run_started_at = start_runtime_timer()
    setup = setup_experiment(args, experiment_name, scenario)
    save_json(setup.run_dir / "args.json", vars(args))

    tracking_backend = try_start_mlflow(
        args=args,
        output_root=Path(args.output_dir).resolve(),
        run_name=experiment_name,
        params={
            "experiment": experiment_name,
            "model": "plain_lstm",
            "scenario": scenario,
            "seed": args.seed,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
            "val_fraction": args.val_fraction,
        },
    )

    history: list[dict[str, float]] = []
    best_epoch = 0
    best_val_task_loss = float("inf")
    best_state = copy.deepcopy(setup.model.state_dict())
    patience_counter = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model=setup.model,
                optimizer=setup.optimizer,
                panel=setup.train_panel,
                grad_clip=args.grad_clip,
            )
            val_metrics, _ = evaluate(setup.model, setup.val_panel)

            epoch_record: dict[str, float] = {
                "epoch": float(epoch),
                "train_task_loss": float(train_metrics["task_loss"]),
                "train_grad_norm": float(train_metrics["grad_norm"]),
                "val_task_loss": float(val_metrics["task_loss"]),
                "val_task_mae": float(val_metrics["task_mae"]),
                "val_task_r2": float(val_metrics["task_r2"]),
            }
            history.append(epoch_record)
            log_mlflow_metrics({key: value for key, value in epoch_record.items() if key != "epoch"}, step=epoch)

            if epoch == 1 or (args.log_every > 0 and epoch % args.log_every == 0):
                print(
                    f"[{experiment_name}] epoch={epoch:03d} "
                    f"train_task={train_metrics['task_loss']:.4f} "
                    f"val_task={val_metrics['task_loss']:.4f} "
                    f"val_mae={val_metrics['task_mae']:.4f} "
                    f"val_r2={val_metrics['task_r2']:.4f}"
                )

            if val_metrics["task_loss"] < best_val_task_loss - 1e-8:
                best_val_task_loss = float(val_metrics["task_loss"])
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(setup.model.state_dict())
                save_torch_checkpoint(
                    {
                        "experiment": experiment_name,
                        "model": "plain_lstm",
                        "best_epoch": best_epoch,
                        "best_val_task_loss": best_val_task_loss,
                        "config": setup.cfg.to_dict(),
                        "model_state_dict": best_state,
                    },
                    setup.checkpoint_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[{experiment_name}] early stopping at epoch {epoch}")
                    break

        setup.model.load_state_dict(best_state)
        final_metrics, outputs = evaluate(setup.model, setup.full_panel, include_outputs=True)
        if outputs is None:
            raise RuntimeError("Expected outputs from final baseline evaluation")

        pd.DataFrame(history).to_csv(setup.history_csv_path, index=False)
        save_json(setup.history_json_path, history)
        save_predictions(outputs, setup.predictions_path)

        lag_ax = plot_omega_heatmap(
            outputs["lag_profile"],
            outputs["z_true"],
            outputs["kstar_true"],
            setup.run_dir / "lag_profile_heatmap.png",
        )
        kstar_ax = plot_kstar_scatter(
            outputs["posthoc_k_star"],
            outputs["kstar_true"],
            setup.run_dir / "posthoc_kstar_scatter.png",
            z_values=outputs["z_true"],
        )
        lag_ax.figure.clf()
        kstar_ax.figure.clf()

        summary = summarize_run(
            setup=setup,
            experiment_name=experiment_name,
            tracking_backend=tracking_backend,
            best_epoch=best_epoch,
            best_val_task_loss=best_val_task_loss,
            metrics=final_metrics,
        )
        attach_runtime_metadata(summary, setup.device, started_at=run_started_at)
        save_json(setup.summary_path, summary)

        return BaselineExperimentResult(
            experiment_name=experiment_name,
            scenario=scenario,
            run_dir=setup.run_dir,
            best_epoch=best_epoch,
            best_val_task_loss=best_val_task_loss,
            tracking_backend=tracking_backend,
            history=history,
            metrics=final_metrics,
            outputs=outputs,
        )
    finally:
        finish_mlflow(
            [
                setup.run_dir / "args.json",
                setup.history_json_path,
                setup.history_csv_path,
                setup.summary_path,
                setup.predictions_path,
                setup.checkpoint_path,
                setup.run_dir / "lag_profile_heatmap.png",
                setup.run_dir / "posthoc_kstar_scatter.png",
            ]
        )


def run_linear(args: argparse.Namespace) -> BaselineExperimentResult:
    """Run the linear synthetic plain-LSTM baseline.

    运行线性 synthetic plain-LSTM baseline。
    """

    return run_experiment(args=args, experiment_name="LSTM_linear", scenario="linear")


def run_nonlinear(args: argparse.Namespace) -> BaselineExperimentResult:
    """Run the nonlinear synthetic plain-LSTM baseline.

    运行非线性 synthetic plain-LSTM baseline。
    """

    return run_experiment(args=args, experiment_name="LSTM_nonlinear", scenario="nonlinear")


def main() -> None:
    """Execute the requested plain-LSTM baseline runs and collect a summary table.

    按参数执行 plain-LSTM baseline，并输出汇总表。
    """

    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    if args.scenario in {"all", "linear"}:
        linear_result = run_linear(args)
        summary_rows.append(
            {
                "experiment": linear_result.experiment_name,
                "scenario": linear_result.scenario,
                "best_epoch": linear_result.best_epoch,
                "best_val_task_loss": linear_result.best_val_task_loss,
                **linear_result.metrics,
            }
        )

    if args.scenario in {"all", "nonlinear"}:
        nonlinear_result = run_nonlinear(args)
        summary_rows.append(
            {
                "experiment": nonlinear_result.experiment_name,
                "scenario": nonlinear_result.scenario,
                "best_epoch": nonlinear_result.best_epoch,
                "best_val_task_loss": nonlinear_result.best_val_task_loss,
                **nonlinear_result.metrics,
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_root / "lstm_baseline_results.csv", index=False)
    save_json(output_root / "lstm_baseline_results.json", summary_rows)

    print("Synthetic plain-LSTM baseline experiments complete.")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()