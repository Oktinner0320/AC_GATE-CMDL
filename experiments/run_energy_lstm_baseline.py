"""Energy-domain plain LSTM baseline runner.

该脚本在 energy 真实数据上复用当前 plain LSTM baseline，
并与 AC-GATE 使用同一份 loader、同一组时间切分和同一套日志落盘习惯。
"""

from __future__ import annotations

import argparse
import copy
import importlib
import os
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
import torch.nn.functional as F

from baselines.panel_ols import add_forecast_calibration
from baselines.lstm_baseline import PlainLSTMBaseline, PlainLSTMBaselineOutput
from config.cmdl_config import CMDLConfig
from data.energy.energy_loader import (
    DEFAULT_ENERGY_FEATURE_BUNDLE,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TREATMENT_COLUMN,
    DEFAULT_YEAR_END,
    DEFAULT_YEAR_START,
    EnergyPanel,
    SUPPORTED_ENERGY_FEATURE_BUNDLES,
    build_temporal_splits,
    get_prediction_years,
    load_energy_panel,
)
from evaluation.metrics import compute_mae, compute_mse, compute_r2
from evaluation.realdata_diagnostics import build_realdata_diagnostics, proxy_metadata_payload
from experiments._runtime_meta import attach_runtime_metadata, start_runtime_timer
from experiments.run_energy import (
    build_sqlite_tracking_uri,
    move_panel_to_device,
    prefix_metrics,
    save_json,
    select_device,
    set_seed,
)


MLFLOW_EXPERIMENT_NAME = "CMDL-Step5-Energy-Baseline"
_METRIC_EPS = 1e-10


@dataclass(slots=True)
class EnergyBaselineSetup:
    """Bundle runtime objects for one energy baseline run.

    energy 域 plain LSTM baseline 单次实验所需对象集合。
    """

    cfg: CMDLConfig
    full_panel: EnergyPanel
    train_panel: EnergyPanel
    val_panel: EnergyPanel
    test_panel: EnergyPanel
    model: PlainLSTMBaseline
    optimizer: torch.optim.Optimizer
    device: torch.device
    run_dir: Path
    checkpoint_path: Path
    history_json_path: Path
    history_csv_path: Path
    summary_path: Path
    predictions_path: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the energy real-data plain-LSTM baseline.

    解析 energy 真实数据 plain-LSTM baseline 的命令行参数。
    """

    energy_defaults = CMDLConfig.from_domain("energy")

    parser = argparse.ArgumentParser(description="Run the energy-domain plain LSTM baseline.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Local merged energy+WGI CSV path. Uses the cached merged CSV by default.",
    )
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
    parser.add_argument("--seed", type=int, default=energy_defaults.seed)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="outputs/step5/energy_lstm_baseline")
    parser.add_argument("--experiment-name", type=str, default="E3_energy_lstm_baseline")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="Run a one-epoch smoke check instead of a full baseline run.")
    return parser.parse_args()


def ensure_mlflow_experiment(tracking_uri: str, artifact_root: Path) -> None:
    """Create the MLflow experiment if it does not already exist.

    若实验不存在，则创建 energy baseline 的 MLflow 实验。
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

    启动 MLflow；若不可用则平滑回退到 JSON 记录。
    """

    if args.disable_mlflow or mlflow is None:
        return "json"

    try:
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
    """Log scalar metrics to the active MLflow run when available."""

    if mlflow is None or mlflow.active_run() is None:
        return
    mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)


def finish_mlflow(artifact_paths: list[Path]) -> None:
    """Upload final artifacts and close the active MLflow run."""

    if mlflow is None or mlflow.active_run() is None:
        return

    for artifact_path in artifact_paths:
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path))
    mlflow.end_run(status="FINISHED")


def aligned_targets(y_true: torch.Tensor, warmup_steps: int) -> torch.Tensor:
    """Trim warm-up steps from targets to match the baseline outputs."""

    if y_true.dim() != 2:
        raise ValueError(f"Expected y_true with shape [B, T], got {tuple(y_true.shape)}")
    if y_true.size(1) <= warmup_steps:
        raise ValueError("y_true length must be greater than warmup_steps")
    return y_true[:, warmup_steps:]


def _tensor_population_std(tensor: torch.Tensor) -> float:
    values = tensor.detach().to(dtype=torch.float64).reshape(-1)
    if values.numel() <= 1:
        return 0.0
    return float(values.std(unbiased=False).item())


def _mean_proxy_signal(panel: EnergyPanel) -> torch.Tensor:
    if panel.p_i.dim() != 2 or panel.p_i.size(1) < 1:
        raise ValueError(f"Expected p_i with shape [N, M], got {tuple(panel.p_i.shape)}")
    return panel.p_i.mean(dim=1)


def compute_posthoc_lag_profile(
    model: PlainLSTMBaseline,
    panel: EnergyPanel,
    base_output: PlainLSTMBaselineOutput | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-entity effective lag profiles by lag-wise occlusion.

    用逐 lag 的输入遮挡构造 energy baseline 的 post-hoc 有效滞后分布，
    并据此得到 pseudo-k*。该分布是解释结果，不是训练得到的原生 omega。
    """

    target = aligned_targets(panel.Y_it, model.cfg.max_lag)
    model.eval()
    with torch.no_grad():
        if base_output is None:
            base_output = model(
                entity_ids=panel.entity_ids,
                X_it=panel.X_it,
                s_i=panel.s_i,
            )

        base_entity_errors = torch.mean((base_output.y_pred - target) ** 2, dim=1)
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
            occluded_entity_errors = torch.mean((occluded_output.y_pred - target) ** 2, dim=1)
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


def setup_experiment(args: argparse.Namespace) -> EnergyBaselineSetup:
    """Prepare data, model, optimizer, and output layout for the energy baseline."""

    if args.smoke:
        args.epochs = 1
        args.patience = 1
    feature_bundle = getattr(args, "feature_bundle", DEFAULT_ENERGY_FEATURE_BUNDLE)

    set_seed(args.seed)
    full_panel_cpu = load_energy_panel(
        csv_path=args.csv_path,
        treatment_column=args.treatment_column,
        target_column=args.target_column,
        feature_bundle=feature_bundle,
        year_start=args.year_start,
        year_end=args.year_end,
        stats_end_year=args.train_end_year,
        max_missing_share=args.max_missing_share,
    )

    cfg = CMDLConfig.from_domain(
        "energy",
        seed=args.seed,
        n_entities=full_panel_cpu.X_it.shape[0],
        seq_length=full_panel_cpu.X_it.shape[1],
        seq_features=full_panel_cpu.X_it.shape[2],
        n_proxies=full_panel_cpu.p_i.shape[1],
        static_dim=full_panel_cpu.s_i.shape[1],
    )

    train_panel_cpu, val_panel_cpu, test_panel_cpu = build_temporal_splits(
        panel=full_panel_cpu,
        max_lag=cfg.max_lag,
        train_end_year=args.train_end_year,
        val_end_year=args.val_end_year,
    )
    device = select_device(args.device)

    full_panel = move_panel_to_device(full_panel_cpu, device)
    train_panel = move_panel_to_device(train_panel_cpu, device)
    val_panel = move_panel_to_device(val_panel_cpu, device)
    test_panel = move_panel_to_device(test_panel_cpu, device)

    model = PlainLSTMBaseline(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_dir = Path(args.output_dir).resolve() / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    return EnergyBaselineSetup(
        cfg=cfg,
        full_panel=full_panel,
        train_panel=train_panel,
        val_panel=val_panel,
        test_panel=test_panel,
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


def train_one_epoch(
    model: PlainLSTMBaseline,
    optimizer: torch.optim.Optimizer,
    panel: EnergyPanel,
    grad_clip: float,
) -> dict[str, float]:
    """Run one optimization epoch on the energy training split."""

    model.train()
    optimizer.zero_grad(set_to_none=True)

    output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, s_i=panel.s_i)
    target = aligned_targets(panel.Y_it, model.cfg.max_lag)
    task_loss = F.mse_loss(output.y_pred, target)
    if not torch.isfinite(task_loss):
        raise FloatingPointError("Encountered a non-finite task loss during energy baseline training")

    task_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()

    return {
        "task_loss": float(task_loss.item()),
        "grad_norm": float(grad_norm),
    }


def evaluate(
    model: PlainLSTMBaseline,
    panel: EnergyPanel,
    include_outputs: bool = False,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    """Evaluate one energy split and optionally return serialized outputs."""

    model.eval()
    with torch.no_grad():
        output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, s_i=panel.s_i)
        target = aligned_targets(panel.Y_it, model.cfg.max_lag)
        task_loss = F.mse_loss(output.y_pred, target)

    lag_profile, pseudo_k_star = compute_posthoc_lag_profile(model, panel, base_output=output)
    lag_profile_np = lag_profile.detach().cpu().numpy()
    diagnostics = build_realdata_diagnostics(
        effective_kstar=pseudo_k_star,
        proxies=panel.p_i,
        metadata=panel.metadata,
        prefix="posthoc_kstar",
        omega=lag_profile,
    )

    metrics = {
        "task_loss": float(task_loss.item()),
        "mse": float(compute_mse(output.y_pred, target)),
        "mae": float(compute_mae(output.y_pred, target)),
        "r2": float(compute_r2(output.y_pred, target)),
        "posthoc_kstar_mean": float(pseudo_k_star.mean().item()),
    }
    metrics.update(diagnostics)
    metrics["lag_profile_entropy_mean"] = metrics["omega_entropy_mean"]
    metrics["kstar_proxy_metric_valid"] = metrics["posthoc_kstar_proxy_metric_valid"]

    if not include_outputs:
        return metrics, None

    outputs = {
        "entity_ids": panel.entity_ids.detach().cpu().numpy(),
        "entity_codes": np.asarray(panel.entity_codes, dtype=object),
        "entity_names": np.asarray(panel.entity_names, dtype=object),
        "years": np.asarray(get_prediction_years(panel, model.cfg.max_lag), dtype=np.int64),
        "y_pred": output.y_pred.detach().cpu().numpy(),
        "y_true": target.detach().cpu().numpy(),
        "lag_profile": lag_profile_np,
        "posthoc_k_star": pseudo_k_star.detach().cpu().numpy(),
        "p_true": panel.p_i.detach().cpu().numpy(),
    }
    return metrics, outputs


def save_predictions(outputs: dict[str, np.ndarray], path: Path) -> None:
    """Persist energy baseline predictions and post-hoc lag diagnostics."""

    rows: list[dict[str, Any]] = []
    years = outputs["years"].astype(int)
    lag_profile = outputs["lag_profile"]

    for entity_index, entity_id in enumerate(outputs["entity_ids"].astype(int)):
        lag_profile_row = lag_profile[entity_index]
        lag_profile_peak = int(np.argmax(lag_profile_row) + 1)
        proxy_true_row = outputs["p_true"][entity_index]
        proxy_signal_true = float(np.mean(proxy_true_row))
        for time_index, year in enumerate(years):
            row = {
                "entity_id": int(entity_id),
                "entity_code": str(outputs["entity_codes"][entity_index]),
                "entity_name": str(outputs["entity_names"][entity_index]),
                "year": int(year),
                "y_true": float(outputs["y_true"][entity_index, time_index]),
                "y_pred": float(outputs["y_pred"][entity_index, time_index]),
                "posthoc_k_star": float(outputs["posthoc_k_star"][entity_index]),
                "lag_profile_peak": lag_profile_peak,
                "proxy_signal_true": proxy_signal_true,
            }
            for proxy_index, proxy_value in enumerate(proxy_true_row, start=1):
                row[f"proxy_{proxy_index}_true"] = float(proxy_value)
            for lag_index, lag_weight in enumerate(lag_profile_row, start=1):
                row[f"lag_profile_{lag_index}"] = float(lag_weight)
            rows.append(row)

    dataframe = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def summarize_run(
    setup: EnergyBaselineSetup,
    args: argparse.Namespace,
    tracking_backend: str,
    best_epoch: int,
    best_val_task_loss: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> dict[str, Any]:
    """Build the JSON summary for one completed energy baseline run."""

    train_metrics = add_forecast_calibration(train_metrics, setup.train_panel, setup.train_panel, setup.cfg.max_lag)
    val_metrics = add_forecast_calibration(val_metrics, setup.train_panel, setup.val_panel, setup.cfg.max_lag)
    test_metrics = add_forecast_calibration(test_metrics, setup.train_panel, setup.test_panel, setup.cfg.max_lag)
    proxy_payload = proxy_metadata_payload(setup.full_panel.metadata, setup.cfg.n_proxies)

    return {
        "experiment": args.experiment_name,
        "model": "plain_lstm",
        "tracking_backend": tracking_backend,
        "device": str(setup.device),
        "best_epoch": int(best_epoch),
        "best_val_task_loss": float(best_val_task_loss),
        "posthoc_lag_method": "lag_occlusion",
        "config": setup.cfg.to_dict(),
        "data": {
            "source_path": setup.full_panel.metadata["source_path"],
            "treatment_column": setup.full_panel.metadata["treatment_column"],
            "target_column": setup.full_panel.metadata["target_column"],
            "feature_bundle": setup.full_panel.metadata["feature_bundle"],
            "seq_feature_columns": list(setup.full_panel.metadata["seq_feature_columns"]),
            "proxy_columns": list(setup.full_panel.metadata["proxy_columns"]),
            **proxy_payload,
            "static_columns": list(setup.full_panel.metadata["static_columns"]),
            "stats_end_year": int(setup.full_panel.metadata["stats_end_year"]),
            "year_start": int(args.year_start),
            "year_end": int(args.year_end),
            "train_end_year": int(args.train_end_year),
            "val_end_year": int(args.val_end_year),
            "n_entities": int(setup.cfg.n_entities),
            "full_seq_length": int(setup.cfg.seq_length),
            "train_years": list(setup.train_panel.metadata["years"]),
            "val_years": list(setup.val_panel.metadata["years"]),
            "test_years": list(setup.test_panel.metadata["years"]),
        },
        "metrics": {
            "train": {key: float(value) for key, value in train_metrics.items()},
            "val": {key: float(value) for key, value in val_metrics.items()},
            "test": {key: float(value) for key, value in test_metrics.items()},
        },
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Train and evaluate the energy real-data plain-LSTM baseline."""

    run_started_at = start_runtime_timer()
    setup = setup_experiment(args)
    save_json(setup.run_dir / "args.json", vars(args))

    tracking_backend = try_start_mlflow(
        args=args,
        output_root=Path(args.output_dir).resolve(),
        run_name=args.experiment_name,
        params={
            "experiment": args.experiment_name,
            "model": "plain_lstm",
            "treatment_column": args.treatment_column,
            "target_column": args.target_column,
            "feature_bundle": getattr(args, "feature_bundle", DEFAULT_ENERGY_FEATURE_BUNDLE),
            "seed": args.seed,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
            "year_start": args.year_start,
            "year_end": args.year_end,
            "train_end_year": args.train_end_year,
            "val_end_year": args.val_end_year,
            "max_missing_share": args.max_missing_share,
        },
    )

    history: list[dict[str, float]] = []
    best_epoch = 0
    best_val_task_loss = float("inf")
    patience_counter = 0
    best_state = copy.deepcopy(setup.model.state_dict())

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
                "val_mae": float(val_metrics["mae"]),
                "val_r2": float(val_metrics["r2"]),
            }
            history.append(epoch_record)
            log_mlflow_metrics({key: value for key, value in epoch_record.items() if key != "epoch"}, step=epoch)

            if epoch == 1 or (args.log_every > 0 and epoch % args.log_every == 0):
                print(
                    f"[{args.experiment_name}] epoch={epoch:03d} "
                    f"train_task={train_metrics['task_loss']:.4f} "
                    f"val_task={val_metrics['task_loss']:.4f} "
                    f"val_mae={val_metrics['mae']:.4f} "
                    f"val_r2={val_metrics['r2']:.4f}"
                )

            if val_metrics["task_loss"] < best_val_task_loss - 1e-8:
                best_val_task_loss = float(val_metrics["task_loss"])
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(setup.model.state_dict())
                torch.save(
                    {
                        "experiment": args.experiment_name,
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
                    print(f"[{args.experiment_name}] early stopping at epoch {epoch}")
                    break

        setup.model.load_state_dict(best_state)
        train_metrics, _ = evaluate(setup.model, setup.train_panel)
        val_metrics, _ = evaluate(setup.model, setup.val_panel)
        test_metrics, outputs = evaluate(setup.model, setup.test_panel, include_outputs=True)
        if outputs is None:
            raise RuntimeError("Expected outputs from energy baseline evaluation")

        pd.DataFrame(history).to_csv(setup.history_csv_path, index=False)
        save_json(setup.history_json_path, history)
        save_predictions(outputs, setup.predictions_path)

        summary = summarize_run(
            setup=setup,
            args=args,
            tracking_backend=tracking_backend,
            best_epoch=best_epoch,
            best_val_task_loss=best_val_task_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
        attach_runtime_metadata(summary, setup.device, started_at=run_started_at)
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


def _resolve_seeds(args: argparse.Namespace) -> list[int]:
    seeds = getattr(args, "seeds", None)
    if seeds is None:
        return [int(args.seed)]
    return [int(seed) for seed in seeds]


def _positive_share(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float((numeric > 0.0).mean())


def aggregate_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate repeated-seed energy plain-LSTM results into mean/std tables."""

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    group_columns = [
        column
        for column in ["model", "treatment_column", "target_column", "feature_bundle"]
        if column in frame.columns
    ]
    non_numeric_columns = {
        "experiment",
        "model",
        "treatment_column",
        "target_column",
        "feature_bundle",
        "seed",
        "run_dir",
        "source_path",
    }
    numeric_columns = [
        column
        for column in frame.columns
        if column not in non_numeric_columns and pd.api.types.is_numeric_dtype(frame[column])
    ]
    grouped = frame.groupby(group_columns, dropna=False)
    aggregated = grouped[numeric_columns].agg(["mean", "std"]).reset_index()
    aggregated.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in aggregated.columns.to_flat_index()
    ]
    n_seeds = grouped["seed"].nunique(dropna=True).reset_index(name="n_seeds")
    aggregated = aggregated.merge(n_seeds, on=group_columns, how="left")
    positive_columns = [
        column
        for column in numeric_columns
        if column.endswith("spearman_adjusted_rho") or column.endswith("adjusted_rho")
    ]
    for column in positive_columns:
        shares = grouped[column].apply(_positive_share).reset_index(name=f"{column}_positive_share")
        aggregated = aggregated.merge(shares, on=group_columns, how="left")
    return aggregated


def run_suite(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute energy plain-LSTM for one or more seeds and write suite summaries."""

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = _resolve_seeds(args)
    base_experiment_name = args.experiment_name
    multi_seed_run = getattr(args, "seeds", None) is not None or len(seeds) > 1
    summary_rows: list[dict[str, Any]] = []
    for seed in seeds:
        run_args = copy.deepcopy(args)
        run_args.seed = int(seed)
        run_args.seeds = None
        if multi_seed_run:
            run_args.experiment_name = f"{base_experiment_name}_seed{seed}"
        summary = run_experiment(run_args)
        summary_rows.append(
            {
                "experiment": summary["experiment"],
                "model": "plain_lstm",
                "treatment_column": summary["data"].get("treatment_column"),
                "target_column": summary["data"]["target_column"],
                "feature_bundle": summary["data"].get("feature_bundle"),
                "source_path": summary["data"]["source_path"],
                "seed": summary["config"]["seed"],
                "best_epoch": summary["best_epoch"],
                "best_val_task_loss": summary["best_val_task_loss"],
                "run_dir": str(output_root / summary["experiment"]),
                **prefix_metrics("train", summary["metrics"]["train"]),
                **prefix_metrics("val", summary["metrics"]["val"]),
                **prefix_metrics("test", summary["metrics"]["test"]),
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_root / "baseline_results.csv", index=False)
    save_json(output_root / "baseline_results.json", summary_rows)

    aggregated = aggregate_results(summary_rows)
    if not aggregated.empty:
        aggregated.to_csv(output_root / "baseline_results_aggregated.csv", index=False)
        aggregated.to_csv(output_root / "baseline_mechanism_summary.csv", index=False)

    return summary_frame, aggregated


def main() -> None:
    """Execute the energy real-data plain-LSTM baseline."""

    args = parse_args()
    if getattr(args, "seeds", None) is not None:
        summary_frame, aggregated = run_suite(args)
        print("Energy plain-LSTM multi-seed baseline complete.")
        if not summary_frame.empty:
            print(summary_frame.to_string(index=False))
        if not aggregated.empty:
            print("\nAggregated across seeds:")
            print(aggregated.to_string(index=False))
        return

    summary = run_experiment(args)
    test_metrics = summary["metrics"]["test"]

    print("Energy plain-LSTM baseline complete.")
    print(
        f"best_epoch={summary['best_epoch']} "
        f"test_mse={test_metrics['mse']:.4f} "
        f"test_mae={test_metrics['mae']:.4f} "
        f"test_r2={test_metrics['r2']:.4f} "
        f"posthoc_kstar_proxy_rho={test_metrics['posthoc_kstar_proxy_spearman_rho']:.4f}"
    )


if __name__ == "__main__":
    main()