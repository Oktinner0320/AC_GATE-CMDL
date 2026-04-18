"""Step 4 synthetic experiment entrypoint for E1a, E1b, and E1c.

Step 4 合成数据实验入口脚本，用于运行 E1a、E1b 与 E1c。
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    mlflow = importlib.import_module("mlflow")
except ImportError:
    mlflow = None

import numpy as np
import pandas as pd
import torch

from config.cmdl_config import CMDLConfig
from data.synthetic.generate import SyntheticPanel, generate_cmdl_synthetic
from evaluation.kstar_eval import evaluate_kstar, evaluate_omega_distribution, evaluate_z_identification
from model.cmdl_model import CMDLModel
from model.loss import DomainAgnosticLoss
from visualization.kstar_distribution import plot_kstar_scatter
from visualization.omega_heatmap import plot_omega_heatmap


MLFLOW_EXPERIMENT_NAME = "CMDL-Step4-Synthetic"


@dataclass(slots=True)
class ExperimentSetup:
    """Bundle runtime objects required by a single experiment.

    汇总单次实验运行所需的运行时对象。
    """

    cfg: CMDLConfig
    full_panel: SyntheticPanel
    train_panel: SyntheticPanel
    val_panel: SyntheticPanel
    model: CMDLModel
    criterion: DomainAgnosticLoss
    optimizer: torch.optim.Optimizer
    device: torch.device
    run_dir: Path
    checkpoint_path: Path
    history_json_path: Path
    history_csv_path: Path
    summary_path: Path
    predictions_path: Path


@dataclass(slots=True)
class ExperimentResult:
    """Store the final artifacts and metrics returned by an experiment.

    保存单次实验返回的最终指标与产物。
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
    """Parse command-line arguments for Step 4 runs.

    解析 Step 4 实验运行所需的命令行参数。
    """

    synthetic_defaults = CMDLConfig.from_domain("synthetic")

    parser = argparse.ArgumentParser(description="Run Step 4 synthetic CMDL experiments.")
    parser.add_argument(
        "--scenario",
        choices=["all", "linear", "nonlinear"],
        default="all",
        help="all runs E1a/E1b/E1c, linear runs E1a/E1b, nonlinear runs E1c only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lambda-r", dest="lambda_r", type=float, default=synthetic_defaults.lambda_r)
    parser.add_argument("--temperature", type=float, default=synthetic_defaults.temperature)
    parser.add_argument(
        "--lag-bias-strength",
        type=float,
        default=synthetic_defaults.lag_bias_strength,
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="outputs/step4")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    同步设置 Python、NumPy 与 PyTorch 的随机种子。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    """Resolve the execution device from a user-facing option.

    根据用户配置解析实验运行设备。
    """

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def subset_panel(panel: SyntheticPanel, entity_indices: torch.Tensor) -> SyntheticPanel:
    """Select a subset of entities while preserving panel structure.

    在保留面板结构的前提下抽取实体子集。
    """

    entity_indices = entity_indices.to(dtype=torch.long, device=panel.entity_ids.device)
    return SyntheticPanel(
        X_it=panel.X_it.index_select(0, entity_indices),
        p_i=panel.p_i.index_select(0, entity_indices),
        s_i=panel.s_i.index_select(0, entity_indices),
        Y_it=panel.Y_it.index_select(0, entity_indices),
        z_true=panel.z_true.index_select(0, entity_indices),
        kstar_true=panel.kstar_true.index_select(0, entity_indices),
        entity_ids=panel.entity_ids.index_select(0, entity_indices),
        time_index=panel.time_index.clone(),
        metadata={**panel.metadata, "split_size": int(entity_indices.numel())},
    )


def move_panel_to_device(panel: SyntheticPanel, device: torch.device) -> SyntheticPanel:
    """Move every tensor field of a synthetic panel onto one device.

    将合成面板中的全部张量字段移动到同一设备。
    """

    return SyntheticPanel(
        X_it=panel.X_it.to(device),
        p_i=panel.p_i.to(device),
        s_i=panel.s_i.to(device),
        Y_it=panel.Y_it.to(device),
        z_true=panel.z_true.to(device),
        kstar_true=panel.kstar_true.to(device),
        entity_ids=panel.entity_ids.to(device),
        time_index=panel.time_index.to(device),
        metadata=dict(panel.metadata),
    )


def split_entity_indices(n_entities: int, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a reproducible entity-level train/validation split.

    按实体维度构造可复现的训练集与验证集划分。
    """

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    val_size = int(round(n_entities * val_fraction))
    val_size = min(max(1, val_size), n_entities - 1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_entities, generator=generator)
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    return train_indices, val_indices


def save_json(path: Path, payload: Any) -> None:
    """Write JSON payloads with stable UTF-8 formatting.

    使用稳定的 UTF-8 格式写出 JSON 文件。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Namespace a metric dictionary with a fixed prefix.

    为一组指标统一添加前缀，便于区分 train/val 阶段。
    """

    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def build_sqlite_tracking_uri(database_path: Path) -> str:
    """Build a Windows-safe sqlite tracking URI for local MLflow runs.

    为本地 MLflow 运行构造兼容 Windows 的 sqlite tracking URI。
    """

    return f"sqlite:///{database_path.resolve().as_posix()}"


def ensure_mlflow_experiment(tracking_uri: str, artifact_root: Path) -> None:
    """Create the MLflow experiment once and pin its artifact directory.

    创建 MLflow 实验并固定 artifact 目录，避免依赖已弃用的文件元数据后端。
    """

    if mlflow is None:
        return

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=artifact_root.resolve().as_uri(),
        )


def try_start_mlflow(
    args: argparse.Namespace,
    output_root: Path,
    run_name: str,
    params: dict[str, Any],
) -> str:
    """Start MLflow logging or gracefully fall back to JSON logging.

    启动 MLflow 记录；若不可用则平滑回退到 JSON 记录。
    """

    if args.disable_mlflow or mlflow is None:
        return "json"

    try:
        # Switching the tracking backend only changes metadata persistence.
        # 切换 tracking backend 只影响实验元数据落盘方式，不会改变训练数值结果。
        tracking_dir = (output_root / "mlflow").resolve()
        tracking_dir.mkdir(parents=True, exist_ok=True)
        artifact_root = tracking_dir / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        tracking_uri = build_sqlite_tracking_uri(tracking_dir / "mlflow.db")

        if mlflow.active_run() is not None:
            mlflow.end_run(status="KILLED")
        ensure_mlflow_experiment(tracking_uri=tracking_uri, artifact_root=artifact_root)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(params)
        return "mlflow"
    except Exception as error:
        print(f"MLflow unavailable, falling back to JSON logging: {error}")
        try:
            if mlflow is not None and mlflow.active_run() is not None:
                mlflow.end_run(status="FAILED")
        except Exception:
            pass
        return "json"


def log_mlflow_metrics(metrics: dict[str, float], step: int) -> None:
    """Log scalar metrics to the active MLflow run when available.

    若存在活动 MLflow run，则写入当前步骤的标量指标。
    """

    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)


def finish_mlflow(artifact_paths: list[Path]) -> None:
    """Upload final artifacts and close the active MLflow run.

    上传最终产物并结束当前活动的 MLflow run。
    """

    if mlflow is None or mlflow.active_run() is None:
        return

    for artifact_path in artifact_paths:
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path))
    mlflow.end_run(status="FINISHED")


def setup_experiment(
    args: argparse.Namespace,
    experiment_name: str,
    scenario: str,
) -> ExperimentSetup:
    """Prepare data, model, optimizer, and file layout for one experiment.

    为单次实验准备数据、模型、优化器以及输出目录结构。
    """

    cfg = CMDLConfig.from_domain(
        "synthetic",
        seed=args.seed,
        scenario=scenario,
        lambda_r=args.lambda_r,
        temperature=args.temperature,
        lag_bias_strength=args.lag_bias_strength,
    )
    set_seed(cfg.seed)

    full_panel_cpu = generate_cmdl_synthetic(cfg)
    train_indices, val_indices = split_entity_indices(cfg.n_entities, args.val_fraction, cfg.seed)
    device = select_device(args.device)

    full_panel = move_panel_to_device(full_panel_cpu, device)
    train_panel = move_panel_to_device(subset_panel(full_panel_cpu, train_indices), device)
    val_panel = move_panel_to_device(subset_panel(full_panel_cpu, val_indices), device)

    model = CMDLModel(cfg).to(device)
    criterion = DomainAgnosticLoss(lambda_r=cfg.lambda_r, warmup_steps=cfg.max_lag)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = Path(args.output_dir).resolve() / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return ExperimentSetup(
        cfg=cfg,
        full_panel=full_panel,
        train_panel=train_panel,
        val_panel=val_panel,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        run_dir=run_dir,
        checkpoint_path=run_dir / "best_model.pt",
        history_json_path=run_dir / "history.json",
        history_csv_path=run_dir / "history.csv",
        summary_path=run_dir / "summary.json",
        predictions_path=run_dir / "predictions.csv",
    )


def refit_proxy_reconstructor(model: CMDLModel, panel: SyntheticPanel) -> None:
    """Refit the linear proxy head on frozen latent scores from one panel split.

    在固定 encoder / lag gate / backbone 的前提下，基于一个面板切分上的 z_i 与 p_i
    对线性 proxy 重构头做一次闭式最小二乘重拟合。
    """

    reconstructor = model.ac_encoder.proxy_reconstructor
    if not isinstance(reconstructor, torch.nn.Linear):
        raise TypeError("proxy_reconstructor must be an nn.Linear layer")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(
            entity_ids=panel.entity_ids,
            X_it=panel.X_it,
            p_i=panel.p_i,
            s_i=panel.s_i,
        )
        z_values = output.z_i.detach()
        proxy_targets = panel.p_i.detach()
        if z_values.dim() != 2 or z_values.size(-1) != 1:
            raise ValueError(f"Expected z_i with shape [N, 1], got {tuple(z_values.shape)}")
        if proxy_targets.dim() != 2:
            raise ValueError(f"Expected p_i with shape [N, M], got {tuple(proxy_targets.shape)}")

        # 在线性头上直接求最小二乘解，避免 6 参数小头在 Adam 下长期跟不上冻结后的 z 分布。
        # Solve least squares directly on the linear head so this tiny 6-parameter layer does not rely on slow Adam updates.
        design_matrix = torch.cat([z_values, torch.ones_like(z_values)], dim=1).to(dtype=torch.float64)
        target_matrix = proxy_targets.to(dtype=torch.float64)
        solution = torch.linalg.lstsq(design_matrix, target_matrix).solution.to(dtype=reconstructor.weight.dtype)
        reconstructor.weight.copy_(solution[0:1].transpose(0, 1))
        reconstructor.bias.copy_(solution[1])

    if was_training:
        model.train()


def train_one_epoch(
    model: CMDLModel,
    criterion: DomainAgnosticLoss,
    optimizer: torch.optim.Optimizer,
    panel: SyntheticPanel,
    grad_clip: float,
) -> dict[str, float]:
    """Run one optimization step over the full synthetic training panel.

    在完整合成训练面板上执行一轮参数更新。
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, p_i=panel.p_i, s_i=panel.s_i)
    losses = criterion(output.y_pred, panel.Y_it, output.p_hat_i, panel.p_i)
    if not torch.isfinite(losses.total_loss):
        raise FloatingPointError("Encountered a non-finite total loss during training")

    losses.total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    return {
        "total_loss": float(losses.total_loss.item()),
        "task_loss": float(losses.task_loss.item()),
        "recon_loss": float(losses.recon_loss.item()),
        "grad_norm": float(grad_norm),
    }


def evaluate(
    model: CMDLModel,
    criterion: DomainAgnosticLoss,
    panel: SyntheticPanel,
    include_outputs: bool = False,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    """Evaluate one panel split and optionally return model outputs.

    评估一个面板子集，并在需要时返回可视化所需的模型输出。
    """

    model.eval()
    with torch.no_grad():
        output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, p_i=panel.p_i, s_i=panel.s_i)
        losses = criterion(output.y_pred, panel.Y_it, output.p_hat_i, panel.p_i)

    metrics = {
        "total_loss": float(losses.total_loss.item()),
        "task_loss": float(losses.task_loss.item()),
        "recon_loss": float(losses.recon_loss.item()),
    }
    metrics.update({f"kstar_{key}": value for key, value in evaluate_kstar(output.k_star, panel.kstar_true).items()})
    metrics.update(evaluate_z_identification(output.z_i, panel.z_true, output.p_hat_i, panel.p_i))
    metrics.update(
        {
            f"omega_{key}": value
            for key, value in evaluate_omega_distribution(output.omega, panel.kstar_true).items()
        }
    )

    if not include_outputs:
        return metrics, None

    aligned_y_true = panel.Y_it[:, model.cfg.max_lag :]
    outputs = {
        "entity_ids": panel.entity_ids.detach().cpu().numpy(),
        "y_pred": output.y_pred.detach().cpu().numpy(),
        "y_true": aligned_y_true.detach().cpu().numpy(),
        "omega": output.omega.detach().cpu().numpy(),
        "k_star": output.k_star.detach().cpu().numpy(),
        "kstar_true": panel.kstar_true.detach().cpu().numpy(),
        "z_pred": output.z_i.detach().cpu().numpy().reshape(-1),
        "z_true": panel.z_true.detach().cpu().numpy().reshape(-1),
        "p_pred": output.p_hat_i.detach().cpu().numpy(),
        "p_true": panel.p_i.detach().cpu().numpy(),
    }
    return metrics, outputs


def save_predictions(outputs: dict[str, np.ndarray], path: Path) -> None:
    """Persist entity-level predictions and omega weights to CSV.

    将实体级预测结果与 omega 权重保存为 CSV 文件。
    """

    omega = outputs["omega"]
    dataframe = pd.DataFrame(
        {
            "entity_id": outputs["entity_ids"].astype(int),
            "z_true": outputs["z_true"],
            "z_pred": outputs["z_pred"],
            "kstar_true": outputs["kstar_true"],
            "kstar_pred": outputs["k_star"],
            "omega_peak": np.argmax(omega, axis=1) + 1,
        }
    )

    for proxy_index in range(outputs["p_true"].shape[1]):
        dataframe[f"proxy_{proxy_index + 1}_true"] = outputs["p_true"][:, proxy_index]
        dataframe[f"proxy_{proxy_index + 1}_pred"] = outputs["p_pred"][:, proxy_index]
    for lag_index in range(omega.shape[1]):
        dataframe[f"omega_{lag_index + 1}"] = omega[:, lag_index]

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def summarize_run(
    setup: ExperimentSetup,
    experiment_name: str,
    tracking_backend: str,
    best_epoch: int,
    best_val_task_loss: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Build the JSON summary for one completed experiment.

    生成单次实验完成后的 JSON 汇总结果。
    """

    return {
        "experiment": experiment_name,
        "scenario": setup.cfg.scenario,
        "tracking_backend": tracking_backend,
        "device": str(setup.device),
        "best_epoch": int(best_epoch),
        "best_val_task_loss": float(best_val_task_loss),
        "config": setup.cfg.to_dict(),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }


def run_experiment(
    args: argparse.Namespace,
    experiment_name: str,
    scenario: str,
) -> ExperimentResult:
    """Train, evaluate, visualize, and log one Step 4 experiment.

    完成单次 Step 4 实验的训练、评估、可视化与日志记录。
    """

    setup = setup_experiment(args, experiment_name, scenario)
    save_json(setup.run_dir / "args.json", vars(args))

    tracking_backend = try_start_mlflow(
        args=args,
        output_root=Path(args.output_dir).resolve(),
        run_name=experiment_name,
        params={
            "experiment": experiment_name,
            "scenario": scenario,
            "seed": args.seed,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "lambda_r": args.lambda_r,
            "temperature": args.temperature,
            "lag_bias_strength": args.lag_bias_strength,
            "grad_clip": args.grad_clip,
            "val_fraction": args.val_fraction,
        },
    )

    history: list[dict[str, float]] = []
    best_epoch = 0
    best_val_task_loss = float("inf")
    best_val_score = float("inf")
    best_state = copy.deepcopy(setup.model.state_dict())
    patience_counter = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model=setup.model,
                criterion=setup.criterion,
                optimizer=setup.optimizer,
                panel=setup.train_panel,
                grad_clip=args.grad_clip,
            )
            val_metrics, _ = evaluate(setup.model, setup.criterion, setup.val_panel)

            epoch_record: dict[str, float] = {"epoch": float(epoch)}
            epoch_record.update(prefix_metrics("train", train_metrics))
            epoch_record.update(prefix_metrics("val", val_metrics))
            history.append(epoch_record)
            log_mlflow_metrics({key: value for key, value in epoch_record.items() if key != "epoch"}, step=epoch)

            if epoch == 1 or (args.log_every > 0 and epoch % args.log_every == 0):
                print(
                    f"[{experiment_name}] epoch={epoch:03d} "
                    f"train_total={train_metrics['total_loss']:.4f} "
                    f"val_task={val_metrics['task_loss']:.4f} "
                    f"val_kstar_mae={val_metrics['kstar_mae']:.4f} "
                    f"val_proxy_r2={val_metrics['proxy_recon_r2']:.4f}"
                )

            # 使用复合指标做模型选择，避免只优化 task_loss 而忽略 k* 恢复质量。
            # Use a composite score for model selection so k* recovery is not ignored.
            val_score = val_metrics["task_loss"] + val_metrics["kstar_mae"]
            if val_score < best_val_score - 1e-8:
                best_val_score = float(val_score)
                best_val_task_loss = float(val_metrics["task_loss"])
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(setup.model.state_dict())
                torch.save(
                    {
                        "experiment": experiment_name,
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
        refit_proxy_reconstructor(setup.model, setup.train_panel)
        # 覆盖最佳 checkpoint，确保落盘模型与最终 summary / predictions 使用的是同一组参数。
        # Overwrite the best checkpoint so the serialized model matches the final summary and predictions.
        torch.save(
            {
                "experiment": experiment_name,
                "best_epoch": best_epoch,
                "best_val_task_loss": best_val_task_loss,
                "config": setup.cfg.to_dict(),
                "proxy_head_refit": True,
                "model_state_dict": copy.deepcopy(setup.model.state_dict()),
            },
            setup.checkpoint_path,
        )

        final_metrics, outputs = evaluate(setup.model, setup.criterion, setup.full_panel, include_outputs=True)
        if outputs is None:
            raise RuntimeError("Expected outputs from final evaluation")

        pd.DataFrame(history).to_csv(setup.history_csv_path, index=False)
        save_json(setup.history_json_path, history)
        save_predictions(outputs, setup.predictions_path)

        omega_ax = plot_omega_heatmap(
            outputs["omega"],
            outputs["z_true"],
            outputs["kstar_true"],
            setup.run_dir / "omega_heatmap.png",
        )
        kstar_ax = plot_kstar_scatter(
            outputs["k_star"],
            outputs["kstar_true"],
            setup.run_dir / "kstar_scatter.png",
            z_values=outputs["z_true"],
        )
        # Release figure objects explicitly so repeated runs do not accumulate GUI state.
        # 显式释放图对象，避免重复实验时持续累积绘图状态。
        omega_ax.figure.clf()
        kstar_ax.figure.clf()

        summary = summarize_run(
            setup=setup,
            experiment_name=experiment_name,
            tracking_backend=tracking_backend,
            best_epoch=best_epoch,
            best_val_task_loss=best_val_task_loss,
            metrics=final_metrics,
        )
        save_json(setup.summary_path, summary)

        return ExperimentResult(
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
                setup.run_dir / "omega_heatmap.png",
                setup.run_dir / "kstar_scatter.png",
            ]
        )


def run_e1a(args: argparse.Namespace) -> ExperimentResult:
    """Run the linear synthetic mechanism verification experiment.

    运行线性场景下的机制验证实验 E1a。
    """

    return run_experiment(args=args, experiment_name="E1a_linear", scenario="linear")


def run_e1b(args: argparse.Namespace, linear_result: ExperimentResult) -> dict[str, Any]:
    """Export z-identification diagnostics from the trained linear experiment.

    基于已训练的线性实验导出 z 可识别性诊断结果。
    """

    run_dir = Path(args.output_dir).resolve() / "E1b_identification"
    run_dir.mkdir(parents=True, exist_ok=True)

    outputs = linear_result.outputs
    # E1b reuses E1a outputs and therefore does not retrain the model.
    # E1b 直接复用 E1a 的输出，因此不会再次训练模型。
    z_table = pd.DataFrame(
        {
            "entity_id": outputs["entity_ids"].astype(int),
            "z_true": outputs["z_true"],
            "z_pred": outputs["z_pred"],
        }
    )
    for proxy_index in range(outputs["p_true"].shape[1]):
        z_table[f"proxy_{proxy_index + 1}_true"] = outputs["p_true"][:, proxy_index]
        z_table[f"proxy_{proxy_index + 1}_pred"] = outputs["p_pred"][:, proxy_index]

    z_table.to_csv(run_dir / "z_identification.csv", index=False)

    summary = {
        "experiment": "E1b_identification",
        "scenario": linear_result.scenario,
        "source_run": linear_result.experiment_name,
        "metrics": {
            "z_spearman_rho": float(linear_result.metrics["z_spearman_rho"]),
            "z_spearman_p": float(linear_result.metrics["z_spearman_p"]),
            "proxy_recon_r2": float(linear_result.metrics["proxy_recon_r2"]),
        },
    }
    save_json(run_dir / "summary.json", summary)
    return summary


def run_e1c(args: argparse.Namespace) -> ExperimentResult:
    """Run the nonlinear synthetic mechanism verification experiment.

    运行非线性场景下的机制验证实验 E1c。
    """

    return run_experiment(args=args, experiment_name="E1c_nonlinear", scenario="nonlinear")


def main() -> None:
    """Execute the requested Step 4 experiments and collect a summary table.

    按参数执行 Step 4 实验，并汇总结果表。
    """

    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    if args.scenario in {"all", "linear"}:
        linear_result = run_e1a(args)
        summary_rows.append(
            {
                "experiment": linear_result.experiment_name,
                "scenario": linear_result.scenario,
                "best_epoch": linear_result.best_epoch,
                "best_val_task_loss": linear_result.best_val_task_loss,
                **linear_result.metrics,
            }
        )
        e1b_summary = run_e1b(args, linear_result)
        summary_rows.append(
            {
                "experiment": e1b_summary["experiment"],
                "scenario": e1b_summary["scenario"],
                **e1b_summary["metrics"],
            }
        )

    if args.scenario in {"all", "nonlinear"}:
        nonlinear_result = run_e1c(args)
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
    summary_frame.to_csv(output_root / "step4_results.csv", index=False)
    save_json(output_root / "step4_results.json", summary_rows)

    print("Step 4 synthetic experiments complete.")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()