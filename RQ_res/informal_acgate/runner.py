"""Isolated Informal AC-GATE experiment runner.

This script intentionally writes outputs under RQ_res by default and leaves the
shared experiments/, data/, model/, and outputs/ trees untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PACKAGE_ROOT = Path(__file__).resolve().parent
RQ_RES_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = RQ_RES_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(RQ_RES_ROOT) not in sys.path:
    sys.path.insert(0, str(RQ_RES_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from baselines.lstm_baseline import PlainLSTMBaseline, PlainLSTMBaselineOutput
from baselines.panel_ols import add_forecast_calibration
from config.cmdl_config import CMDLConfig
from evaluation.metrics import compute_mae, compute_mse, compute_r2
from evaluation.realdata_diagnostics import build_realdata_diagnostics, proxy_metadata_payload
from model.cmdl_model import CMDLModel
from model.loss import DomainAgnosticLoss

from informal_acgate.loader import (
    MISSING_POLICIES,
    SUPPORTED_FEATURE_BUNDLES,
    InformalPanel,
    build_temporal_splits,
    get_prediction_years,
    load_informal_panel,
    move_panel_to_device,
)
from informal_acgate.falsification import PROXY_PERTURBATION_CHOICES, apply_proxy_perturbation


MODEL_CHOICES = ("cmdl", "plain_lstm")
ABLATION_CHOICES = ("none", "no_ac_encoder", "uniform_lag", "no_recon_regularization")
GRAD_CLIP_MODE_CHOICES = ("global", "none", "main_only", "split")
RECON_LOSS_MODE_CHOICES = ("all", "anchor_only", "anchor_weighted")
METRIC_EPS = 1e-10


@dataclass(slots=True)
class ProxyRefitResult:
    """Describe whether the closed-form proxy refit is interpretable."""

    status: str
    applied: bool
    metrics_interpretable: bool
    reason: str | None
    design_rank: int | None
    design_columns: int | None
    latent_std: float | None
    proxy_std: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "applied": self.applied,
            "metrics_interpretable": self.metrics_interpretable,
            "reason": self.reason,
            "design_rank": self.design_rank,
            "design_columns": self.design_columns,
            "latent_std": self.latent_std,
            "proxy_std": self.proxy_std,
        }


@dataclass(slots=True)
class InformalExperimentSetup:
    """Runtime objects for one isolated Informal experiment."""

    cfg: CMDLConfig
    full_panel: InformalPanel
    train_panel: InformalPanel
    val_panel: InformalPanel
    test_panel: InformalPanel
    model: torch.nn.Module
    criterion: DomainAgnosticLoss | None
    optimizer: torch.optim.Optimizer
    device: torch.device
    run_dir: Path
    checkpoint_path: Path
    history_json_path: Path
    history_csv_path: Path
    summary_path: Path
    predictions_path: Path
    audit_path: Path
    effective_lambda_r: float
    grad_clip: float
    grad_clip_mode: str


def parse_args() -> argparse.Namespace:
    defaults = CMDLConfig.from_domain("economics")
    parser = argparse.ArgumentParser(description="Run an isolated Informal AC-GATE RQ experiment.")
    parser.add_argument("--csv-path", type=str, default=None)
    parser.add_argument(
        "--feature-bundle",
        choices=sorted(SUPPORTED_FEATURE_BUNDLES),
        default="multiseq_overlap",
    )
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument("--stats-end-year", type=int, default=None)
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-end-year", type=int, default=2022)
    parser.add_argument("--missing-policy", choices=sorted(MISSING_POLICIES), default="error")
    parser.add_argument("--proxy-perturbation", choices=PROXY_PERTURBATION_CHOICES, default="none")
    parser.add_argument("--proxy-perturbation-seed", type=int, default=None)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="cmdl")
    parser.add_argument("--ablation", choices=ABLATION_CHOICES, default="none")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--max-lag", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lambda-r", dest="lambda_r", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--omega-transform", choices=["softmax", "sparsemax"], default=defaults.omega_transform)
    parser.add_argument("--lambda-omega-entropy", type=float, default=defaults.lambda_omega_entropy)
    parser.add_argument("--omega-entropy-min", type=float, default=defaults.omega_entropy_min)
    parser.add_argument("--omega-entropy-max", type=float, default=defaults.omega_entropy_max)
    parser.add_argument("--lambda-z-anchor", type=float, default=defaults.lambda_z_anchor)
    parser.add_argument("--z-anchor-target-sign", type=float, default=defaults.z_anchor_target_sign)
    parser.add_argument("--lag-bias-strength", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-clip-mode", choices=GRAD_CLIP_MODE_CHOICES, default="global")
    parser.add_argument("--recon-loss-mode", choices=RECON_LOSS_MODE_CHOICES, default="all")
    parser.add_argument("--anchor-recon-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-detach", dest="reconstruction_detach", action="store_true", default=True)
    parser.add_argument("--no-reconstruction-detach", dest="reconstruction_detach", action="store_false")
    parser.add_argument("--output-dir", type=str, default=str(RQ_RES_ROOT / "outputs" / "informal_acgate"))
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="Run one epoch for a fast wiring check.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _experiment_name(args: argparse.Namespace) -> str:
    if args.experiment_name:
        return str(args.experiment_name)
    suffix = f"{args.model}_{args.feature_bundle}_seed{args.seed}"
    if args.ablation != "none":
        suffix = f"cmdl_{args.feature_bundle}_{args.ablation}_seed{args.seed}"
    return f"informal_{suffix}"


def _tensor_population_std(tensor: torch.Tensor) -> float:
    values = tensor.detach().to(dtype=torch.float64).reshape(-1)
    if values.numel() <= 1:
        return 0.0
    return float(values.std(unbiased=False).item())


def _parameter_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().to(dtype=torch.float64)
        squared_norm += float(torch.sum(grad * grad).item())
    return float(np.sqrt(squared_norm))


def _split_main_proxy_parameters(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    main_parameters: list[torch.nn.Parameter] = []
    proxy_parameters: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "proxy_reconstructor" in name:
            proxy_parameters.append(parameter)
        else:
            main_parameters.append(parameter)
    return main_parameters, proxy_parameters


def _clip_gradients(model: torch.nn.Module, grad_clip: float, grad_clip_mode: str) -> dict[str, float]:
    main_parameters, proxy_parameters = _split_main_proxy_parameters(model)
    all_parameters = main_parameters + proxy_parameters
    grad_norm_total = _parameter_grad_norm(all_parameters)
    grad_norm_main = _parameter_grad_norm(main_parameters)
    grad_norm_proxy_head = _parameter_grad_norm(proxy_parameters)
    clip_applied = grad_clip_mode != "none" and grad_clip > 0.0

    if clip_applied:
        if grad_clip_mode == "global":
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=grad_clip)
        elif grad_clip_mode == "main_only":
            torch.nn.utils.clip_grad_norm_(main_parameters, max_norm=grad_clip)
        elif grad_clip_mode == "split":
            torch.nn.utils.clip_grad_norm_(main_parameters, max_norm=grad_clip)
            torch.nn.utils.clip_grad_norm_(proxy_parameters, max_norm=grad_clip)
        else:
            raise ValueError(f"Unsupported grad_clip_mode: {grad_clip_mode}")
    return {
        "grad_norm": grad_norm_total,
        "grad_norm_total": grad_norm_total,
        "grad_norm_main": grad_norm_main,
        "grad_norm_proxy_head": grad_norm_proxy_head,
        "clip_applied": float(clip_applied),
    }


def _aligned_targets(panel: InformalPanel, max_lag: int) -> torch.Tensor:
    if panel.Y_it.dim() != 2 or panel.Y_it.size(1) <= max_lag:
        raise ValueError("panel.Y_it must have shape [N, T] with T > max_lag")
    return panel.Y_it[:, max_lag:]


def _build_proxy_refit_matrices(model: CMDLModel, panel: InformalPanel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, p_i=panel.p_i, s_i=panel.s_i)
    design_matrix = torch.cat([output.z_i.detach(), torch.ones_like(output.z_i.detach())], dim=1).to(
        dtype=torch.float64
    )
    target_matrix = panel.p_i.detach().to(dtype=torch.float64)
    return output.z_i.detach(), design_matrix, target_matrix


def refit_proxy_reconstructor(model: torch.nn.Module, panel: InformalPanel) -> ProxyRefitResult:
    """Refit the linear proxy head on frozen z values when interpretable."""

    if not isinstance(model, CMDLModel):
        return ProxyRefitResult("skipped_non_cmdl", False, False, "non_cmdl_model", None, None, None, None)
    reconstructor = model.ac_encoder.proxy_reconstructor
    if not isinstance(reconstructor, torch.nn.Linear):
        raise TypeError("proxy_reconstructor must be an nn.Linear layer")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        latent_scores, design_matrix, target_matrix = _build_proxy_refit_matrices(model, panel)
        design_rank = int(torch.linalg.matrix_rank(design_matrix).item())
        design_columns = int(design_matrix.shape[1])
        latent_std = _tensor_population_std(latent_scores)
        proxy_std = _tensor_population_std(target_matrix)
        if proxy_std <= METRIC_EPS:
            result = ProxyRefitResult(
                "skipped_constant_proxy",
                False,
                False,
                "constant_proxy",
                design_rank,
                design_columns,
                latent_std,
                proxy_std,
            )
        elif design_rank < design_columns:
            result = ProxyRefitResult(
                "skipped_rank_deficient",
                False,
                False,
                "rank_deficient_design",
                design_rank,
                design_columns,
                latent_std,
                proxy_std,
            )
        else:
            solution = torch.linalg.lstsq(design_matrix, target_matrix).solution.to(dtype=reconstructor.weight.dtype)
            reconstructor.weight.copy_(solution[0:1].transpose(0, 1))
            reconstructor.bias.copy_(solution[1])
            result = ProxyRefitResult("applied_ols", True, True, None, design_rank, design_columns, latent_std, proxy_std)
    if was_training:
        model.train()
    return result


def build_variant_model(variant: str, cfg: CMDLConfig) -> tuple[CMDLModel, float, dict[str, Any]]:
    if variant == "none":
        return CMDLModel(cfg), cfg.lambda_r, {"variant": "none", "causal_ablation_validity": "full_model"}
    if variant == "no_recon_regularization":
        return CMDLModel(cfg), 0.0, {
            "variant": variant,
            "same_architecture_as_full_cmdl": True,
            "causal_ablation_validity": "same_architecture_lambda_zero",
        }

    from experiments.run_ablation import NoACEncoderCMDLModel, UniformLagCMDLModel

    if variant == "no_ac_encoder":
        return NoACEncoderCMDLModel(cfg), 0.0, {
            "variant": variant,
            "same_architecture_as_full_cmdl": False,
            "causal_ablation_validity": "architecture_changed",
        }
    if variant == "uniform_lag":
        return UniformLagCMDLModel(cfg), cfg.lambda_r, {
            "variant": variant,
            "same_architecture_as_full_cmdl": False,
            "causal_ablation_validity": "architecture_changed",
        }
    raise ValueError(f"Unsupported ablation variant: {variant}")


def compute_posthoc_lag_profile(
    model: PlainLSTMBaseline,
    panel: InformalPanel,
    base_output: PlainLSTMBaselineOutput | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate baseline lag profiles by lag-wise occlusion."""

    target = _aligned_targets(panel, model.cfg.max_lag)
    model.eval()
    with torch.no_grad():
        if base_output is None:
            base_output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, s_i=panel.s_i)
        base_entity_errors = torch.mean((base_output.y_pred - target) ** 2, dim=1)
        lag_scores: list[torch.Tensor] = []
        for lag_index in range(1, model.cfg.max_lag + 1):
            occluded_inputs = panel.X_it.clone()
            start_index = model.cfg.max_lag - lag_index
            end_index = panel.X_it.size(1) - lag_index
            occluded_inputs[:, start_index:end_index, :] = 0.0
            occluded_output = model(entity_ids=panel.entity_ids, X_it=occluded_inputs, s_i=panel.s_i)
            occluded_errors = torch.mean((occluded_output.y_pred - target) ** 2, dim=1)
            lag_scores.append(torch.relu(occluded_errors - base_entity_errors))

        score_matrix = torch.stack(lag_scores, dim=1)
        row_sum = score_matrix.sum(dim=1, keepdim=True)
        uniform = torch.full_like(score_matrix, 1.0 / model.cfg.max_lag)
        omega = torch.where(row_sum > METRIC_EPS, score_matrix / row_sum.clamp_min(METRIC_EPS), uniform)
        lag_indices = torch.arange(1, model.cfg.max_lag + 1, dtype=omega.dtype, device=omega.device)
        k_star = torch.sum(omega * lag_indices.unsqueeze(0), dim=1)
    return omega, k_star


def train_one_epoch(setup: InformalExperimentSetup) -> dict[str, float]:
    setup.model.train()
    setup.optimizer.zero_grad(set_to_none=True)
    if isinstance(setup.model, PlainLSTMBaseline):
        output = setup.model(entity_ids=setup.train_panel.entity_ids, X_it=setup.train_panel.X_it, s_i=setup.train_panel.s_i)
        target = _aligned_targets(setup.train_panel, setup.cfg.max_lag)
        loss = F.mse_loss(output.y_pred, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite LSTM loss during Informal training")
        loss.backward()
        grad_metrics = _clip_gradients(setup.model, grad_clip=setup.grad_clip, grad_clip_mode=setup.grad_clip_mode)
        setup.optimizer.step()
        metrics = {
            "total_loss": float(loss.item()),
            "task_loss": float(loss.item()),
            "recon_loss": float("nan"),
            "anchor_recon_loss": float("nan"),
            "omega_entropy_penalty": float("nan"),
            "omega_entropy_band_violation_share": float("nan"),
            "z_anchor_loss": float("nan"),
        }
        metrics.update(grad_metrics)
        return metrics

    if setup.criterion is None:
        raise RuntimeError("CMDL training requires a criterion")
    output = setup.model(
        entity_ids=setup.train_panel.entity_ids,
        X_it=setup.train_panel.X_it,
        p_i=setup.train_panel.p_i,
        s_i=setup.train_panel.s_i,
    )
    losses = setup.criterion(output.y_pred, setup.train_panel.Y_it, output.p_hat_i, setup.train_panel.p_i, output.omega, output.z_i)
    if not torch.isfinite(losses.total_loss):
        raise FloatingPointError("Encountered a non-finite total loss during Informal training")
    losses.total_loss.backward()
    grad_metrics = _clip_gradients(setup.model, grad_clip=setup.grad_clip, grad_clip_mode=setup.grad_clip_mode)
    setup.optimizer.step()
    metrics = {
        "total_loss": float(losses.total_loss.item()),
        "task_loss": float(losses.task_loss.item()),
        "recon_loss": float(losses.recon_loss.item()),
        "anchor_recon_loss": float(losses.anchor_recon_loss.item()),
        "omega_entropy_penalty": float(losses.omega_entropy_penalty.item()),
        "omega_entropy_band_violation_share": float(losses.omega_entropy_band_violation_share.item()),
        "z_anchor_loss": float(losses.z_anchor_loss.item()),
    }
    metrics.update(grad_metrics)
    return metrics


def evaluate(
    model: torch.nn.Module,
    criterion: DomainAgnosticLoss | None,
    panel: InformalPanel,
    include_outputs: bool = False,
    proxy_refit_result: ProxyRefitResult | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    model.eval()
    aligned_y_true = _aligned_targets(panel, model.cfg.max_lag)  # type: ignore[attr-defined]

    if isinstance(model, PlainLSTMBaseline):
        with torch.no_grad():
            output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, s_i=panel.s_i)
            omega, k_star = compute_posthoc_lag_profile(model, panel, base_output=output)
        diagnostics = build_realdata_diagnostics(
            effective_kstar=k_star,
            proxies=panel.p_i,
            metadata=panel.metadata,
            prefix="kstar",
            omega=omega,
        )
        task_loss = float(F.mse_loss(output.y_pred, aligned_y_true).item())
        metrics = {
            "total_loss": task_loss,
            "task_loss": task_loss,
            "recon_loss": float("nan"),
            "anchor_recon_loss": float("nan"),
            "omega_entropy_penalty": float("nan"),
            "omega_entropy_band_violation_share": float("nan"),
            "z_anchor_loss": float("nan"),
            "mse": float(compute_mse(output.y_pred, aligned_y_true)),
            "mae": float(compute_mae(output.y_pred, aligned_y_true)),
            "r2": float(compute_r2(output.y_pred, aligned_y_true)),
            "kstar_mean": float(k_star.mean().item()),
        }
        metrics.update(diagnostics)
        if not include_outputs:
            return metrics, None
        outputs = {
            "entity_ids": panel.entity_ids.detach().cpu().numpy(),
            "entity_codes": np.asarray(panel.entity_codes, dtype=object),
            "entity_names": np.asarray(panel.entity_names, dtype=object),
            "years": np.asarray(get_prediction_years(panel, model.cfg.max_lag), dtype=np.int64),
            "y_pred": output.y_pred.detach().cpu().numpy(),
            "y_true": aligned_y_true.detach().cpu().numpy(),
            "omega": omega.detach().cpu().numpy(),
            "k_star": k_star.detach().cpu().numpy(),
            "p_pred": np.full(tuple(panel.p_i.shape), np.nan, dtype=np.float32),
            "p_true": panel.p_i.detach().cpu().numpy(),
        }
        return metrics, outputs

    if criterion is None:
        raise RuntimeError("CMDL evaluation requires a criterion")
    with torch.no_grad():
        output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, p_i=panel.p_i, s_i=panel.s_i)
        losses = criterion(output.y_pred, panel.Y_it, output.p_hat_i, panel.p_i, output.omega, output.z_i)

    proxy_metric_valid = True if proxy_refit_result is None else proxy_refit_result.metrics_interpretable
    diagnostics = build_realdata_diagnostics(
        effective_kstar=output.k_star,
        proxies=panel.p_i,
        metadata=panel.metadata,
        prefix="kstar",
        omega=output.omega,
        z_values=output.z_i,
        proxy_predictions=output.p_hat_i,
        proxy_metric_valid=proxy_metric_valid,
        lag_gate=getattr(model, "lag_gate", None),
    )
    metrics = {
        "total_loss": float(losses.total_loss.item()),
        "task_loss": float(losses.task_loss.item()),
        "recon_loss": float(losses.recon_loss.item()),
        "anchor_recon_loss": float(losses.anchor_recon_loss.item()),
        "omega_entropy_penalty": float(losses.omega_entropy_penalty.item()),
        "omega_entropy_band_violation_share": float(losses.omega_entropy_band_violation_share.item()),
        "z_anchor_loss": float(losses.z_anchor_loss.item()),
        "mse": float(compute_mse(output.y_pred, aligned_y_true)),
        "mae": float(compute_mae(output.y_pred, aligned_y_true)),
        "r2": float(compute_r2(output.y_pred, aligned_y_true)),
        "kstar_mean": float(output.k_star.mean().item()),
    }
    metrics.update(diagnostics)

    if not include_outputs:
        return metrics, None
    outputs = {
        "entity_ids": panel.entity_ids.detach().cpu().numpy(),
        "entity_codes": np.asarray(panel.entity_codes, dtype=object),
        "entity_names": np.asarray(panel.entity_names, dtype=object),
        "years": np.asarray(get_prediction_years(panel, model.cfg.max_lag), dtype=np.int64),
        "y_pred": output.y_pred.detach().cpu().numpy(),
        "y_true": aligned_y_true.detach().cpu().numpy(),
        "omega": output.omega.detach().cpu().numpy(),
        "k_star": output.k_star.detach().cpu().numpy(),
        "p_pred": output.p_hat_i.detach().cpu().numpy(),
        "p_true": panel.p_i.detach().cpu().numpy(),
    }
    return metrics, outputs


def save_predictions(outputs: dict[str, np.ndarray], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    years = outputs["years"].astype(int)
    omega = outputs["omega"]
    for entity_index, entity_id in enumerate(outputs["entity_ids"].astype(int)):
        omega_row = omega[entity_index]
        omega_peak = int(np.argmax(omega_row) + 1)
        proxy_true_row = outputs["p_true"][entity_index]
        proxy_pred_row = outputs["p_pred"][entity_index]
        for time_index, year in enumerate(years):
            row = {
                "entity_id": int(entity_id),
                "entity_code": str(outputs["entity_codes"][entity_index]),
                "entity_name": str(outputs["entity_names"][entity_index]),
                "year": int(year),
                "y_true": float(outputs["y_true"][entity_index, time_index]),
                "y_pred": float(outputs["y_pred"][entity_index, time_index]),
                "k_star": float(outputs["k_star"][entity_index]),
                "omega_peak": omega_peak,
            }
            for proxy_index, proxy_value in enumerate(proxy_true_row, start=1):
                row[f"proxy_{proxy_index}_true"] = float(proxy_value)
                row[f"proxy_{proxy_index}_pred"] = float(proxy_pred_row[proxy_index - 1])
            for lag_index, lag_weight in enumerate(omega_row, start=1):
                row[f"omega_{lag_index}"] = float(lag_weight)
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def summarize_run(
    setup: InformalExperimentSetup,
    args: argparse.Namespace,
    started_at: str,
    duration_seconds: float,
    best_epoch: int,
    best_val_task_loss: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    proxy_refit_result: ProxyRefitResult | None,
    ablation_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    train_metrics = add_forecast_calibration(train_metrics, setup.train_panel, setup.train_panel, setup.cfg.max_lag)
    val_metrics = add_forecast_calibration(val_metrics, setup.train_panel, setup.val_panel, setup.cfg.max_lag)
    test_metrics = add_forecast_calibration(test_metrics, setup.train_panel, setup.test_panel, setup.cfg.max_lag)
    proxy_payload = proxy_metadata_payload(setup.full_panel.metadata, setup.cfg.n_proxies)
    audit = setup.full_panel.metadata.get("audit", {})

    return {
        "experiment": _experiment_name(args),
        "model": args.model,
        "ablation": args.ablation,
        "best_epoch": int(best_epoch),
        "best_val_task_loss": float(best_val_task_loss),
        "config": setup.cfg.to_dict(),
        "data": {
            "domain": "informal",
            "source_path": setup.full_panel.metadata["source_path"],
            "feature_bundle": setup.full_panel.metadata["feature_bundle"],
            "target_column": setup.full_panel.metadata["target_column"],
            "seq_feature_columns": list(setup.full_panel.metadata["seq_feature_columns"]),
            "proxy_columns": list(setup.full_panel.metadata["proxy_columns"]),
            **proxy_payload,
            "static_columns": list(setup.full_panel.metadata["static_columns"]),
            "stats_end_year": int(setup.full_panel.metadata["stats_end_year"]),
            "year_start": int(setup.full_panel.metadata["year_start"]),
            "year_end": int(setup.full_panel.metadata["year_end"]),
            "train_end_year": int(args.train_end_year),
            "val_end_year": int(args.val_end_year),
            "missing_policy": setup.full_panel.metadata["missing_policy"],
            "n_entities": int(setup.cfg.n_entities),
            "full_seq_length": int(setup.cfg.seq_length),
            "train_years": list(setup.train_panel.metadata["years"]),
            "val_years": list(setup.val_panel.metadata["years"]),
            "test_years": list(setup.test_panel.metadata["years"]),
            "audit": audit,
        },
        "metrics": {
            "train": {key: float(value) for key, value in train_metrics.items()},
            "val": {key: float(value) for key, value in val_metrics.items()},
            "test": {key: float(value) for key, value in test_metrics.items()},
        },
        "diagnostics": {
            "proxy_refit": None if proxy_refit_result is None else proxy_refit_result.to_dict(),
            "ablation": dict(ablation_diagnostics),
            "training_controls": {
                "effective_lambda_r": float(setup.effective_lambda_r),
                "grad_clip_mode": args.grad_clip_mode,
                "recon_loss_mode": args.recon_loss_mode,
                "anchor_recon_weight": float(args.anchor_recon_weight),
                "reconstruction_detach": bool(args.reconstruction_detach),
                "omega_transform": args.omega_transform,
                "lambda_omega_entropy": float(args.lambda_omega_entropy),
                "omega_entropy_min": args.omega_entropy_min,
                "omega_entropy_max": args.omega_entropy_max,
                "lambda_z_anchor": float(args.lambda_z_anchor),
                "z_anchor_target_sign": float(args.z_anchor_target_sign),
                "proxy_perturbation": getattr(args, "proxy_perturbation", "none"),
                "proxy_perturbation_seed": getattr(args, "proxy_perturbation_seed", None),
            },
        },
        "runtime": {
            "started_at": started_at,
            "finished_at": _now_iso(),
            "duration_seconds": float(duration_seconds),
            "device": str(setup.device),
        },
    }


def setup_experiment(args: argparse.Namespace) -> tuple[InformalExperimentSetup, dict[str, Any]]:
    if args.smoke:
        args.epochs = 1
        args.patience = 1
    if args.model == "plain_lstm" and args.ablation != "none":
        raise ValueError("Ablations are only supported for --model cmdl")

    set_seed(int(args.seed))
    stats_end_year = args.train_end_year if args.stats_end_year is None else args.stats_end_year
    full_panel_cpu = load_informal_panel(
        csv_path=args.csv_path,
        feature_bundle=args.feature_bundle,
        year_start=args.year_start,
        year_end=args.year_end,
        stats_end_year=stats_end_year,
        missing_policy=args.missing_policy,
    )
    proxy_perturbation = getattr(args, "proxy_perturbation", "none")
    proxy_perturbation_seed = getattr(args, "proxy_perturbation_seed", None)
    if proxy_perturbation_seed is None:
        proxy_perturbation_seed = int(args.seed)
    full_panel_cpu = apply_proxy_perturbation(
        full_panel_cpu,
        mode=proxy_perturbation,
        seed=proxy_perturbation_seed,
    )
    cfg = CMDLConfig.from_domain(
        "economics",
        seed=args.seed,
        max_lag=args.max_lag,
        d_model=args.d_model,
        n_entities=full_panel_cpu.X_it.shape[0],
        seq_length=full_panel_cpu.X_it.shape[1],
        seq_features=full_panel_cpu.X_it.shape[2],
        n_proxies=full_panel_cpu.p_i.shape[1],
        static_dim=full_panel_cpu.s_i.shape[1],
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        lambda_r=args.lambda_r,
        temperature=args.temperature,
        lag_bias_strength=args.lag_bias_strength,
        omega_transform=args.omega_transform,
        lambda_omega_entropy=args.lambda_omega_entropy,
        omega_entropy_min=args.omega_entropy_min,
        omega_entropy_max=args.omega_entropy_max,
        lambda_z_anchor=args.lambda_z_anchor,
        z_anchor_target_sign=args.z_anchor_target_sign,
        reconstruction_detach=args.reconstruction_detach,
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

    ablation_diagnostics: dict[str, Any] = {"variant": args.ablation}
    if args.model == "plain_lstm":
        model: torch.nn.Module = PlainLSTMBaseline(cfg).to(device)
        effective_lambda_r = float("nan")
        criterion = None
    else:
        model, effective_lambda_r, ablation_diagnostics = build_variant_model(args.ablation, cfg)
        model = model.to(device)
        criterion = DomainAgnosticLoss(
            lambda_r=effective_lambda_r,
            warmup_steps=cfg.max_lag,
            recon_loss_mode=args.recon_loss_mode,
            anchor_proxy_index=int(full_panel_cpu.metadata.get("anchor_proxy_index", 0)),
            anchor_recon_weight=float(args.anchor_recon_weight),
            lambda_omega_entropy=cfg.lambda_omega_entropy,
            omega_entropy_min=cfg.omega_entropy_min,
            omega_entropy_max=cfg.omega_entropy_max,
            lambda_z_anchor=cfg.lambda_z_anchor,
            z_anchor_target_sign=cfg.z_anchor_target_sign,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run_dir = Path(args.output_dir).resolve() / _experiment_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    setup = InformalExperimentSetup(
        cfg=cfg,
        full_panel=full_panel,
        train_panel=train_panel,
        val_panel=val_panel,
        test_panel=test_panel,
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
        audit_path=run_dir / "panel_audit.json",
        effective_lambda_r=float(effective_lambda_r),
        grad_clip=float(args.grad_clip),
        grad_clip_mode=str(args.grad_clip_mode),
    )
    return setup, ablation_diagnostics


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    started_at = _now_iso()
    started_seconds = time.perf_counter()
    setup, ablation_diagnostics = setup_experiment(args)
    save_json(setup.run_dir / "args.json", vars(args))
    save_json(setup.audit_path, setup.full_panel.metadata.get("audit", {}))

    history: list[dict[str, float]] = []
    best_epoch = 0
    best_val_task_loss = float("inf")
    patience_counter = 0
    best_state = copy.deepcopy(setup.model.state_dict())

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(setup)
        val_metrics, _ = evaluate(setup.model, setup.criterion, setup.val_panel)

        epoch_record: dict[str, float] = {"epoch": float(epoch)}
        epoch_record.update(prefix_metrics("train", train_metrics))
        epoch_record.update(prefix_metrics("val", val_metrics))
        history.append(epoch_record)

        if epoch == 1 or (args.log_every > 0 and epoch % args.log_every == 0):
            print(
                f"[{_experiment_name(args)}] epoch={epoch:03d} "
                f"train_total={train_metrics['total_loss']:.4f} "
                f"val_task={val_metrics['task_loss']:.4f} "
                f"val_r2={val_metrics['r2']:.4f}"
            )

        if val_metrics["task_loss"] < best_val_task_loss - 1e-8:
            best_val_task_loss = float(val_metrics["task_loss"])
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(setup.model.state_dict())
            torch.save(
                {
                    "experiment": _experiment_name(args),
                    "model": args.model,
                    "ablation": args.ablation,
                    "best_epoch": best_epoch,
                    "best_val_task_loss": best_val_task_loss,
                    "effective_lambda_r": setup.effective_lambda_r,
                    "config": setup.cfg.to_dict(),
                    "model_state_dict": best_state,
                },
                setup.checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[{_experiment_name(args)}] early stopping at epoch {epoch}")
                break

    setup.model.load_state_dict(best_state)
    proxy_refit_result: ProxyRefitResult | None = None
    if not isinstance(setup.model, PlainLSTMBaseline):
        proxy_refit_result = refit_proxy_reconstructor(setup.model, setup.train_panel)
        if not proxy_refit_result.applied:
            print(f"[{_experiment_name(args)}] proxy refit skipped: {proxy_refit_result.reason}")
        torch.save(
            {
                "experiment": _experiment_name(args),
                "model": args.model,
                "ablation": args.ablation,
                "best_epoch": best_epoch,
                "best_val_task_loss": best_val_task_loss,
                "effective_lambda_r": setup.effective_lambda_r,
                "proxy_refit": proxy_refit_result.to_dict(),
                "config": setup.cfg.to_dict(),
                "model_state_dict": copy.deepcopy(setup.model.state_dict()),
            },
            setup.checkpoint_path,
        )

    train_metrics, _ = evaluate(setup.model, setup.criterion, setup.train_panel, proxy_refit_result=proxy_refit_result)
    val_metrics, _ = evaluate(setup.model, setup.criterion, setup.val_panel, proxy_refit_result=proxy_refit_result)
    test_metrics, outputs = evaluate(
        setup.model,
        setup.criterion,
        setup.test_panel,
        include_outputs=True,
        proxy_refit_result=proxy_refit_result,
    )
    if outputs is None:
        raise RuntimeError("Expected Informal test outputs")

    pd.DataFrame(history).to_csv(setup.history_csv_path, index=False)
    save_json(setup.history_json_path, history)
    save_predictions(outputs, setup.predictions_path)
    summary = summarize_run(
        setup=setup,
        args=args,
        started_at=started_at,
        duration_seconds=time.perf_counter() - started_seconds,
        best_epoch=best_epoch,
        best_val_task_loss=best_val_task_loss,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        proxy_refit_result=proxy_refit_result,
        ablation_diagnostics=ablation_diagnostics,
    )
    save_json(setup.summary_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run_experiment(args)
    test_metrics = summary["metrics"]["test"]
    print("Informal experiment complete.")
    print(
        f"experiment={summary['experiment']} "
        f"best_epoch={summary['best_epoch']} "
        f"test_mse={test_metrics['mse']:.4f} "
        f"test_mae={test_metrics['mae']:.4f} "
        f"test_r2={test_metrics['r2']:.4f} "
        f"kstar_adj_rho={test_metrics.get('kstar_proxy_spearman_adjusted_rho', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
