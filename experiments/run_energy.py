"""Energy-domain experiment entrypoint for real-data CMDL validation.

该脚本复用 Step 5 economics 实验的训练壳子思路，但数据加载与 proxy
诊断口径按 energy 域单独实现，不依赖任何跨域 preprocessing 抽象。
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
from typing import Any, Callable

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

from baselines.panel_ols import add_forecast_calibration
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
from experiments._checkpoint_io import save_torch_checkpoint
from experiments._proxy_shuffle_control import negative_control_metadata
from experiments._runtime_meta import attach_runtime_metadata, start_runtime_timer
from model.cmdl_model import CMDLModel
from model.loss import DomainAgnosticLoss


MLFLOW_EXPERIMENT_NAME = "CMDL-Step5-Energy"


@dataclass(slots=True)
class EnergyExperimentSetup:
	"""Bundle runtime objects for one energy experiment.

	energy 单次实验所需运行时对象集合。
	"""

	cfg: CMDLConfig
	full_panel: EnergyPanel
	train_panel: EnergyPanel
	val_panel: EnergyPanel
	test_panel: EnergyPanel
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
class ProxyRefitResult:
	"""Describe whether the closed-form proxy refit is interpretable.

	记录 proxy head 闭式重拟合是否成功，以及对应诊断是否可解释。
	"""

	status: str
	applied: bool
	metrics_interpretable: bool
	reason: str | None
	design_rank: int | None
	design_columns: int | None
	latent_std: float | None
	proxy_std: float | None

	def to_dict(self) -> dict[str, Any]:
		"""Convert the refit result into a JSON-serializable dictionary."""

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


_METRIC_EPS = 1e-10
GRAD_CLIP_MODE_CHOICES = ("global", "none", "main_only", "split")


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for the energy real-data experiment.

	解析 energy 现实数据实验所需参数。
	"""

	energy_defaults = CMDLConfig.from_domain("energy")

	parser = argparse.ArgumentParser(description="Run the energy-domain CMDL experiment.")
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
	parser.add_argument(
		"--treatment-column",
		type=str,
		default=DEFAULT_TREATMENT_COLUMN,
		help="Energy treatment column, e.g. renewables_share_energy.",
	)
	parser.add_argument(
		"--target-column",
		type=str,
		default=DEFAULT_TARGET_COLUMN,
		help="Energy target column, e.g. co2_per_unit_energy.",
	)
	parser.add_argument(
		"--feature-bundle",
		type=str,
		choices=list(SUPPORTED_ENERGY_FEATURE_BUNDLES),
		default=DEFAULT_ENERGY_FEATURE_BUNDLE,
		help="Energy feature/proxy bundle to use.",
	)
	parser.add_argument("--max-missing-share", type=float, default=0.15)
	parser.add_argument("--seed", type=int, default=energy_defaults.seed)
	parser.add_argument("--seeds", nargs="+", type=int, default=None)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--epochs", type=int, default=120)
	parser.add_argument("--patience", type=int, default=20)
	parser.add_argument("--lambda-r", dest="lambda_r", type=float, default=energy_defaults.lambda_r)
	parser.add_argument("--temperature", type=float, default=energy_defaults.temperature)
	parser.add_argument("--omega-transform", choices=["softmax", "sparsemax"], default=energy_defaults.omega_transform)
	parser.add_argument("--lambda-omega-entropy", type=float, default=energy_defaults.lambda_omega_entropy)
	parser.add_argument("--omega-entropy-min", type=float, default=energy_defaults.omega_entropy_min)
	parser.add_argument("--omega-entropy-max", type=float, default=energy_defaults.omega_entropy_max)
	parser.add_argument("--lambda-z-anchor", type=float, default=energy_defaults.lambda_z_anchor)
	parser.add_argument("--z-anchor-target-sign", type=float, default=energy_defaults.z_anchor_target_sign)
	parser.add_argument("--lag-bias-strength", type=float, default=energy_defaults.lag_bias_strength)
	parser.add_argument("--grad-clip", type=float, default=1.0)
	parser.add_argument("--grad-clip-mode", choices=GRAD_CLIP_MODE_CHOICES, default="global")
	parser.add_argument("--output-dir", type=str, default="outputs/step5/energy")
	parser.add_argument("--experiment-name", type=str, default="E3_energy")
	parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
	parser.add_argument("--disable-mlflow", action="store_true")
	parser.add_argument("--log-every", type=int, default=10)
	parser.add_argument("--smoke", action="store_true", help="Run a one-epoch smoke check instead of a full training run.")
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
	"""Resolve the target execution device.

	解析实验执行设备。
	"""

	if device_name == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA was requested but is not available")
		return torch.device("cuda")
	if device_name == "cpu":
		return torch.device("cpu")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: Path, payload: Any) -> None:
	"""Write JSON payloads with stable UTF-8 formatting.

	使用稳定的 UTF-8 格式写出 JSON 文件。
	"""

	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
	"""Namespace a metric dictionary with a fixed prefix.

	为一组指标统一添加前缀，便于区分 train/val/test 阶段。
	"""

	return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def build_sqlite_tracking_uri(database_path: Path) -> str:
	"""Build a Windows-safe sqlite tracking URI for local MLflow runs.

	为本地 MLflow 构造兼容 Windows 的 sqlite tracking URI。
	"""

	return f"sqlite:///{database_path.resolve().as_posix()}"


def ensure_mlflow_experiment(tracking_uri: str, artifact_root: Path) -> None:
	"""Create the MLflow experiment if it does not already exist.

	若实验不存在，则创建 energy 域的 MLflow 实验。
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


def move_panel_to_device(panel: EnergyPanel, device: torch.device) -> EnergyPanel:
	"""Move an energy panel onto the specified device.

	将 energy 面板中的全部张量字段移动到指定设备。
	"""

	return EnergyPanel(
		X_it=panel.X_it.to(device),
		p_i=panel.p_i.to(device),
		s_i=panel.s_i.to(device),
		Y_it=panel.Y_it.to(device),
		entity_ids=panel.entity_ids.to(device),
		time_index=panel.time_index.to(device),
		entity_codes=list(panel.entity_codes),
		entity_names=list(panel.entity_names),
		metadata=dict(panel.metadata),
	)


def _tensor_population_std(tensor: torch.Tensor) -> float:
	"""Return the population standard deviation of a tensor as a Python float."""

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


def _mean_proxy_signal(panel: EnergyPanel) -> torch.Tensor:
	"""Collapse multi-proxy energy signals into one scalar summary per entity.

	energy 域没有单一主 proxy，这里用标准化后的实体级 proxy 均值作为
	k* 相关性的单标量参考信号。
	"""

	if panel.p_i.dim() != 2 or panel.p_i.size(1) < 1:
		raise ValueError(f"Expected p_i with shape [N, M], got {tuple(panel.p_i.shape)}")
	return panel.p_i.mean(dim=1)


def _build_proxy_refit_matrices(
	model: CMDLModel,
	panel: EnergyPanel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Build the linear system used by the proxy-head refit."""

	output = model(
		entity_ids=panel.entity_ids,
		X_it=panel.X_it,
		p_i=panel.p_i,
		s_i=panel.s_i,
	)
	design_matrix = torch.cat([output.z_i.detach(), torch.ones_like(output.z_i.detach())], dim=1).to(
		dtype=torch.float64
	)
	target_matrix = panel.p_i.detach().to(dtype=torch.float64)
	return output.z_i.detach(), design_matrix, target_matrix


def refit_proxy_reconstructor(model: CMDLModel, panel: EnergyPanel) -> ProxyRefitResult:
	"""Refit the linear proxy head on frozen energy latent scores.

	在固定主体模型参数的前提下，对 energy 域的线性 proxy 重构头做闭式重拟合。
	"""

	reconstructor = model.ac_encoder.proxy_reconstructor
	if not isinstance(reconstructor, torch.nn.Linear):
		raise TypeError("proxy_reconstructor must be an nn.Linear layer")

	was_training = model.training
	model.eval()
	result: ProxyRefitResult
	with torch.no_grad():
		latent_scores, design_matrix, target_matrix = _build_proxy_refit_matrices(model, panel)
		design_rank = int(torch.linalg.matrix_rank(design_matrix).item())
		design_columns = int(design_matrix.shape[1])
		latent_std = _tensor_population_std(latent_scores)
		proxy_std = _tensor_population_std(target_matrix)

		if proxy_std <= _METRIC_EPS:
			result = ProxyRefitResult(
				status="skipped_constant_proxy",
				applied=False,
				metrics_interpretable=False,
				reason="constant_proxy",
				design_rank=design_rank,
				design_columns=design_columns,
				latent_std=latent_std,
				proxy_std=proxy_std,
			)
		elif design_rank < design_columns:
			result = ProxyRefitResult(
				status="skipped_rank_deficient",
				applied=False,
				metrics_interpretable=False,
				reason="rank_deficient_design",
				design_rank=design_rank,
				design_columns=design_columns,
				latent_std=latent_std,
				proxy_std=proxy_std,
			)
		else:
			solution = torch.linalg.lstsq(design_matrix, target_matrix).solution.to(dtype=reconstructor.weight.dtype)
			reconstructor.weight.copy_(solution[0:1].transpose(0, 1))
			reconstructor.bias.copy_(solution[1])
			result = ProxyRefitResult(
				status="applied_ols",
				applied=True,
				metrics_interpretable=True,
				reason=None,
				design_rank=design_rank,
				design_columns=design_columns,
				latent_std=latent_std,
				proxy_std=proxy_std,
			)

	if was_training:
		model.train()
	return result


def train_one_epoch(
	model: CMDLModel,
	criterion: DomainAgnosticLoss,
	optimizer: torch.optim.Optimizer,
	panel: EnergyPanel,
	grad_clip: float,
	grad_clip_mode: str = "global",
) -> dict[str, float]:
	"""Run one optimization step over the energy training split.

	在 energy 训练切分上执行一轮参数更新。
	"""

	model.train()
	optimizer.zero_grad(set_to_none=True)

	output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, p_i=panel.p_i, s_i=panel.s_i)
	losses = criterion(output.y_pred, panel.Y_it, output.p_hat_i, panel.p_i, output.omega, output.z_i)
	if not torch.isfinite(losses.total_loss):
		raise FloatingPointError("Encountered a non-finite total loss during energy training")

	losses.total_loss.backward()
	grad_metrics = _clip_gradients(model, grad_clip=grad_clip, grad_clip_mode=grad_clip_mode)
	optimizer.step()

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
	model: CMDLModel,
	criterion: DomainAgnosticLoss,
	panel: EnergyPanel,
	include_outputs: bool = False,
	proxy_refit_result: ProxyRefitResult | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
	"""Evaluate one energy split and optionally return detailed outputs.

	评估 energy 的一个时间切分，并在需要时返回详细输出。
	"""

	model.eval()
	with torch.no_grad():
		output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, p_i=panel.p_i, s_i=panel.s_i)
		losses = criterion(output.y_pred, panel.Y_it, output.p_hat_i, panel.p_i, output.omega, output.z_i)

	aligned_y_true = panel.Y_it[:, model.cfg.max_lag :]
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
	"""Persist energy test predictions and entity-level lag diagnostics.

	保存 energy 测试集预测结果和实体级 lag 诊断信息。
	"""

	rows: list[dict[str, Any]] = []
	years = outputs["years"].astype(int)
	omega = outputs["omega"]

	for entity_index, entity_id in enumerate(outputs["entity_ids"].astype(int)):
		omega_row = omega[entity_index]
		omega_peak = int(np.argmax(omega_row) + 1)
		proxy_true_row = outputs["p_true"][entity_index]
		proxy_pred_row = outputs["p_pred"][entity_index]
		proxy_signal_true = float(np.mean(proxy_true_row))
		proxy_signal_pred = float(np.mean(proxy_pred_row))

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
				"proxy_signal_true": proxy_signal_true,
				"proxy_signal_pred": proxy_signal_pred,
			}
			for proxy_index in range(proxy_true_row.shape[0]):
				row[f"proxy_{proxy_index + 1}_true"] = float(proxy_true_row[proxy_index])
				row[f"proxy_{proxy_index + 1}_pred"] = float(proxy_pred_row[proxy_index])
			for lag_index, lag_weight in enumerate(omega_row, start=1):
				row[f"omega_{lag_index}"] = float(lag_weight)
			rows.append(row)

	dataframe = pd.DataFrame(rows)
	path.parent.mkdir(parents=True, exist_ok=True)
	dataframe.to_csv(path, index=False)


def summarize_run(
	setup: EnergyExperimentSetup,
	args: argparse.Namespace,
	tracking_backend: str,
	best_epoch: int,
	best_val_task_loss: float,
	train_metrics: dict[str, float],
	val_metrics: dict[str, float],
	test_metrics: dict[str, float],
	proxy_refit_result: ProxyRefitResult | None = None,
) -> dict[str, Any]:
	"""Build the JSON summary for one completed energy experiment.

	生成 energy 实验完成后的 JSON 汇总结果。
	"""

	train_metrics = add_forecast_calibration(train_metrics, setup.train_panel, setup.train_panel, setup.cfg.max_lag)
	val_metrics = add_forecast_calibration(val_metrics, setup.train_panel, setup.val_panel, setup.cfg.max_lag)
	test_metrics = add_forecast_calibration(test_metrics, setup.train_panel, setup.test_panel, setup.cfg.max_lag)
	proxy_payload = proxy_metadata_payload(setup.full_panel.metadata, setup.cfg.n_proxies)

	return {
		"experiment": args.experiment_name,
		"tracking_backend": tracking_backend,
		"device": str(setup.device),
		"best_epoch": int(best_epoch),
		"best_val_task_loss": float(best_val_task_loss),
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
		"diagnostics": {
			"proxy_refit": None if proxy_refit_result is None else proxy_refit_result.to_dict(),
			"training_controls": {
				"grad_clip_mode": getattr(args, "grad_clip_mode", "global"),
				"omega_transform": getattr(args, "omega_transform", "softmax"),
				"lambda_omega_entropy": float(getattr(args, "lambda_omega_entropy", 0.0)),
				"omega_entropy_min": getattr(args, "omega_entropy_min", None),
				"omega_entropy_max": getattr(args, "omega_entropy_max", None),
				"lambda_z_anchor": float(getattr(args, "lambda_z_anchor", 0.0)),
				"z_anchor_target_sign": float(getattr(args, "z_anchor_target_sign", 1.0)),
			},
		},
	}


def setup_experiment(args: argparse.Namespace) -> EnergyExperimentSetup:
	"""Prepare data, model, optimizer, and output layout for energy.

	为 energy 实验准备数据、模型、优化器与输出目录。
	"""

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
		lambda_r=args.lambda_r,
		temperature=args.temperature,
		lag_bias_strength=args.lag_bias_strength,
		omega_transform=getattr(args, "omega_transform", "softmax"),
		lambda_omega_entropy=getattr(args, "lambda_omega_entropy", 0.0),
		omega_entropy_min=getattr(args, "omega_entropy_min", None),
		omega_entropy_max=getattr(args, "omega_entropy_max", None),
		lambda_z_anchor=getattr(args, "lambda_z_anchor", 0.0),
		z_anchor_target_sign=getattr(args, "z_anchor_target_sign", 1.0),
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

	model = CMDLModel(cfg).to(device)
	criterion = DomainAgnosticLoss(
		lambda_r=cfg.lambda_r,
		warmup_steps=cfg.max_lag,
		anchor_proxy_index=int(full_panel_cpu.metadata.get("anchor_proxy_index", 0)),
		lambda_omega_entropy=cfg.lambda_omega_entropy,
		omega_entropy_min=cfg.omega_entropy_min,
		omega_entropy_max=cfg.omega_entropy_max,
		lambda_z_anchor=cfg.lambda_z_anchor,
		z_anchor_target_sign=cfg.z_anchor_target_sign,
	)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	run_dir = Path(args.output_dir).resolve() / args.experiment_name
	run_dir.mkdir(parents=True, exist_ok=True)

	return EnergyExperimentSetup(
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
	)


def run_experiment(
	args: argparse.Namespace,
	setup_transform: Callable[[EnergyExperimentSetup], EnergyExperimentSetup | None] | None = None,
) -> dict[str, Any]:
	"""Train and evaluate the energy real-data experiment.

	完成 energy 现实数据实验的训练、评估、日志与落盘。
	"""

	run_started_at = start_runtime_timer()
	setup = setup_experiment(args)
	if setup_transform is not None:
		transformed_setup = setup_transform(setup)
		if transformed_setup is not None:
			setup = transformed_setup
	save_json(setup.run_dir / "args.json", vars(args))

	tracking_backend = try_start_mlflow(
		args=args,
		output_root=Path(args.output_dir).resolve(),
		run_name=args.experiment_name,
		params={
			"experiment": args.experiment_name,
			"treatment_column": args.treatment_column,
			"target_column": args.target_column,
			"feature_bundle": getattr(args, "feature_bundle", DEFAULT_ENERGY_FEATURE_BUNDLE),
			"seed": args.seed,
			"lr": args.lr,
			"epochs": args.epochs,
			"patience": args.patience,
			"lambda_r": args.lambda_r,
			"temperature": args.temperature,
			"lag_bias_strength": args.lag_bias_strength,
			"grad_clip": args.grad_clip,
			"grad_clip_mode": getattr(args, "grad_clip_mode", "global"),
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
				criterion=setup.criterion,
				optimizer=setup.optimizer,
				panel=setup.train_panel,
				grad_clip=args.grad_clip,
				grad_clip_mode=getattr(args, "grad_clip_mode", "global"),
			)
			val_metrics, _ = evaluate(setup.model, setup.criterion, setup.val_panel)

			epoch_record: dict[str, float] = {"epoch": float(epoch)}
			epoch_record.update(prefix_metrics("train", train_metrics))
			epoch_record.update(prefix_metrics("val", val_metrics))
			history.append(epoch_record)
			log_mlflow_metrics({key: value for key, value in epoch_record.items() if key != "epoch"}, step=epoch)

			if epoch == 1 or (args.log_every > 0 and epoch % args.log_every == 0):
				print(
					f"[{args.experiment_name}] epoch={epoch:03d} "
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
				save_torch_checkpoint(
					{
						"experiment": args.experiment_name,
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
		proxy_refit_result = refit_proxy_reconstructor(setup.model, setup.train_panel)
		if not proxy_refit_result.applied:
			print(
				f"[{args.experiment_name}] proxy refit skipped: "
				f"{proxy_refit_result.reason} (rank={proxy_refit_result.design_rank}/"
				f"{proxy_refit_result.design_columns})"
			)
		save_torch_checkpoint(
			{
				"experiment": args.experiment_name,
				"best_epoch": best_epoch,
				"best_val_task_loss": best_val_task_loss,
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
			raise RuntimeError("Expected test outputs from energy evaluation")

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
			proxy_refit_result=proxy_refit_result,
		)
		control_metadata = negative_control_metadata(setup.full_panel.metadata)
		if control_metadata is not None:
			summary["negative_control"] = control_metadata
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
	"""Aggregate repeated-seed energy CMDL results into mean/std tables."""

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
		"proxy_refit_status",
		"proxy_refit_reason",
		"grad_clip_mode",
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
	"""Execute energy CMDL for one or more seeds and write suite summaries."""

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
				"model": "cmdl",
				"treatment_column": summary["data"].get("treatment_column"),
				"target_column": summary["data"]["target_column"],
				"feature_bundle": summary["data"].get("feature_bundle"),
				"source_path": summary["data"]["source_path"],
				"seed": summary["config"]["seed"],
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
	summary_frame.to_csv(output_root / "cmdl_results.csv", index=False)
	save_json(output_root / "cmdl_results.json", summary_rows)

	aggregated = aggregate_results(summary_rows)
	if not aggregated.empty:
		aggregated.to_csv(output_root / "cmdl_results_aggregated.csv", index=False)
		aggregated.to_csv(output_root / "cmdl_mechanism_summary.csv", index=False)

	return summary_frame, aggregated


def main() -> None:
	"""Execute the energy real-data experiment.

	执行 energy 现实数据实验，并打印关键指标摘要。
	"""

	args = parse_args()
	if getattr(args, "seeds", None) is not None:
		summary_frame, aggregated = run_suite(args)
		print("Energy CMDL multi-seed experiments complete.")
		if not summary_frame.empty:
			print(summary_frame.to_string(index=False))
		if not aggregated.empty:
			print("\nAggregated across seeds:")
			print(aggregated.to_string(index=False))
		return

	summary = run_experiment(args)

	test_metrics = summary["metrics"]["test"]
	print("Energy experiment complete.")
	print(
		f"best_epoch={summary['best_epoch']} "
		f"test_mse={test_metrics['mse']:.4f} "
		f"test_mae={test_metrics['mae']:.4f} "
		f"test_r2={test_metrics['r2']:.4f} "
		f"proxy_recon_r2={test_metrics['proxy_recon_r2']:.4f}"
	)


if __name__ == "__main__":
	main()
