"""Economics-domain experiment entrypoint for real-data CMDL validation.

该脚本复用 Step 4 的训练壳子思路，但数据清洗与时间切分完全按 economics 域
单独实现，不依赖任何跨域 preprocessing 抽象。
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
from data.economics.economics_loader import (
	DEFAULT_ECONOMICS_FEATURE_BUNDLE,
	DEFAULT_YEAR_END,
	DEFAULT_YEAR_START,
	EconomicsPanel,
	SUPPORTED_ECONOMICS_FEATURE_BUNDLES,
	build_temporal_splits,
	get_prediction_years,
	load_economics_panel,
)
from evaluation.metrics import compute_mae, compute_mse, compute_r2, compute_spearman
from model.cmdl_model import CMDLModel
from model.loss import DomainAgnosticLoss


MLFLOW_EXPERIMENT_NAME = "CMDL-Step5-Economics"


@dataclass(slots=True)
class EconomicsExperimentSetup:
	"""Bundle runtime objects for one economics experiment.

	economics 单次实验所需运行时对象集合。
	"""

	cfg: CMDLConfig
	full_panel: EconomicsPanel
	train_panel: EconomicsPanel
	val_panel: EconomicsPanel
	test_panel: EconomicsPanel
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


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for the economics real-data experiment.

	解析 economics 现实数据实验所需参数。
	"""

	economics_defaults = CMDLConfig.from_domain("economics")

	parser = argparse.ArgumentParser(description="Run the economics-domain CMDL experiment.")
	parser.add_argument(
		"--csv-path",
		type=str,
		default=None,
		help="Local PWT table path. Supports the cached CSV default and explicit CSV/XLSX files.",
	)
	parser.add_argument("--year-start", type=int, default=DEFAULT_YEAR_START)
	parser.add_argument("--year-end", type=int, default=DEFAULT_YEAR_END)
	parser.add_argument("--train-end-year", type=int, default=2007)
	parser.add_argument("--val-end-year", type=int, default=2013)
	parser.add_argument("--target-column", type=str, default="ctfp", help="PWT target column, e.g. ctfp or rtfpna.")
	parser.add_argument(
		"--feature-bundle",
		type=str,
		choices=list(SUPPORTED_ECONOMICS_FEATURE_BUNDLES),
		default=DEFAULT_ECONOMICS_FEATURE_BUNDLE,
		help="Economics feature/proxy bundle to use, e.g. minimal, growth_aware, or effective_labor_aware.",
	)
	parser.add_argument("--max-missing-share", type=float, default=0.15)
	parser.add_argument("--seed", type=int, default=economics_defaults.seed)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--epochs", type=int, default=120)
	parser.add_argument("--patience", type=int, default=20)
	parser.add_argument("--lambda-r", dest="lambda_r", type=float, default=economics_defaults.lambda_r)
	parser.add_argument("--temperature", type=float, default=economics_defaults.temperature)
	parser.add_argument("--lag-bias-strength", type=float, default=economics_defaults.lag_bias_strength)
	parser.add_argument("--grad-clip", type=float, default=1.0)
	parser.add_argument("--output-dir", type=str, default="outputs/step5/economics")
	parser.add_argument("--experiment-name", type=str, default="E4_economics")
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

	若实验不存在，则创建 economics 域的 MLflow 实验。
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


def move_panel_to_device(panel: EconomicsPanel, device: torch.device) -> EconomicsPanel:
	"""Move an economics panel onto the specified device.

	将 economics 面板中的全部张量字段移动到指定设备。
	"""

	return EconomicsPanel(
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


def _build_proxy_refit_matrices(
	model: CMDLModel,
	panel: EconomicsPanel,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def refit_proxy_reconstructor(model: CMDLModel, panel: EconomicsPanel) -> ProxyRefitResult:
	"""Refit the linear proxy head on frozen economics latent scores.

	在固定主体模型参数的前提下，对 economics 域的线性 proxy 重构头做闭式重拟合。
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
	panel: EconomicsPanel,
	grad_clip: float,
) -> dict[str, float]:
	"""Run one optimization step over the economics training split.

	在 economics 训练切分上执行一轮参数更新。
	"""

	model.train()
	optimizer.zero_grad(set_to_none=True)

	output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, p_i=panel.p_i, s_i=panel.s_i)
	losses = criterion(output.y_pred, panel.Y_it, output.p_hat_i, panel.p_i)
	if not torch.isfinite(losses.total_loss):
		raise FloatingPointError("Encountered a non-finite total loss during economics training")

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
	panel: EconomicsPanel,
	include_outputs: bool = False,
	proxy_refit_result: ProxyRefitResult | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
	"""Evaluate one economics split and optionally return detailed outputs.

	评估 economics 的一个时间切分，并在需要时返回详细输出。
	"""

	model.eval()
	with torch.no_grad():
		output = model(entity_ids=panel.entity_ids, X_it=panel.X_it, p_i=panel.p_i, s_i=panel.s_i)
		losses = criterion(output.y_pred, panel.Y_it, output.p_hat_i, panel.p_i)

	aligned_y_true = panel.Y_it[:, model.cfg.max_lag :]
	omega = output.omega.detach().cpu().numpy()
	omega_entropy = -np.sum(omega * np.log(np.clip(omega, 1e-8, None)), axis=1)
	kstar_std = float(output.k_star.std(unbiased=False).item()) if output.k_star.numel() > 1 else 0.0
	proxy_std = _tensor_population_std(panel.p_i[:, 0])
	kstar_proxy_metric_valid = kstar_std > _METRIC_EPS and proxy_std > _METRIC_EPS
	if kstar_proxy_metric_valid:
		kstar_rho, kstar_p_value = compute_spearman(output.k_star, panel.p_i[:, 0])
	else:
		kstar_rho, kstar_p_value = float("nan"), float("nan")

	proxy_metric_valid = True if proxy_refit_result is None else proxy_refit_result.metrics_interpretable
	proxy_recon_r2 = float(compute_r2(output.p_hat_i, panel.p_i)) if proxy_metric_valid else float("nan")

	metrics = {
		"total_loss": float(losses.total_loss.item()),
		"task_loss": float(losses.task_loss.item()),
		"recon_loss": float(losses.recon_loss.item()),
		"mse": float(compute_mse(output.y_pred, aligned_y_true)),
		"mae": float(compute_mae(output.y_pred, aligned_y_true)),
		"r2": float(compute_r2(output.y_pred, aligned_y_true)),
		"proxy_recon_r2": proxy_recon_r2,
		"proxy_metric_valid": float(proxy_metric_valid),
		"kstar_proxy_spearman_rho": float(kstar_rho),
		"kstar_proxy_spearman_p": float(kstar_p_value),
		"kstar_mean": float(output.k_star.mean().item()),
		"kstar_std": kstar_std,
		"kstar_proxy_metric_valid": float(kstar_proxy_metric_valid),
		"omega_entropy_mean": float(np.mean(omega_entropy)),
	}

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
	"""Persist economics test predictions and entity-level lag diagnostics.

	保存 economics 测试集预测结果和实体级 lag 诊断信息。
	"""

	rows: list[dict[str, Any]] = []
	years = outputs["years"].astype(int)
	omega = outputs["omega"]

	for entity_index, entity_id in enumerate(outputs["entity_ids"].astype(int)):
		omega_row = omega[entity_index]
		omega_peak = int(np.argmax(omega_row) + 1)
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
				"proxy_1_true": float(outputs["p_true"][entity_index, 0]),
				"proxy_1_pred": float(outputs["p_pred"][entity_index, 0]),
			}
			for lag_index, lag_weight in enumerate(omega_row, start=1):
				row[f"omega_{lag_index}"] = float(lag_weight)
			rows.append(row)

	dataframe = pd.DataFrame(rows)
	path.parent.mkdir(parents=True, exist_ok=True)
	dataframe.to_csv(path, index=False)


def summarize_run(
	setup: EconomicsExperimentSetup,
	args: argparse.Namespace,
	tracking_backend: str,
	best_epoch: int,
	best_val_task_loss: float,
	train_metrics: dict[str, float],
	val_metrics: dict[str, float],
	test_metrics: dict[str, float],
	proxy_refit_result: ProxyRefitResult | None = None,
) -> dict[str, Any]:
	"""Build the JSON summary for one completed economics experiment.

	生成 economics 实验完成后的 JSON 汇总结果。
	"""

	return {
		"experiment": args.experiment_name,
		"tracking_backend": tracking_backend,
		"device": str(setup.device),
		"best_epoch": int(best_epoch),
		"best_val_task_loss": float(best_val_task_loss),
		"config": setup.cfg.to_dict(),
		"data": {
			"source_path": setup.full_panel.metadata["source_path"],
			"target_column": setup.full_panel.metadata["target_column"],
			"feature_bundle": setup.full_panel.metadata["feature_bundle"],
			"seq_feature_columns": list(setup.full_panel.metadata["seq_feature_columns"]),
			"proxy_columns": list(setup.full_panel.metadata["proxy_columns"]),
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
		},
	}


def setup_experiment(args: argparse.Namespace) -> EconomicsExperimentSetup:
	"""Prepare data, model, optimizer, and output layout for economics.

	为 economics 实验准备数据、模型、优化器与输出目录。
	"""

	if args.smoke:
		args.epochs = 1
		args.patience = 1
	feature_bundle = getattr(args, "feature_bundle", DEFAULT_ECONOMICS_FEATURE_BUNDLE)

	set_seed(args.seed)
	full_panel_cpu = load_economics_panel(
		csv_path=args.csv_path,
		target_column=args.target_column,
		feature_bundle=feature_bundle,
		year_start=args.year_start,
		year_end=args.year_end,
		stats_end_year=args.train_end_year,
		max_missing_share=args.max_missing_share,
	)

	cfg = CMDLConfig.from_domain(
		"economics",
		seed=args.seed,
		lambda_r=args.lambda_r,
		temperature=args.temperature,
		lag_bias_strength=args.lag_bias_strength,
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
	criterion = DomainAgnosticLoss(lambda_r=cfg.lambda_r, warmup_steps=cfg.max_lag)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	run_dir = Path(args.output_dir).resolve() / args.experiment_name
	run_dir.mkdir(parents=True, exist_ok=True)

	return EconomicsExperimentSetup(
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


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
	"""Train and evaluate the economics real-data experiment.

	完成 economics 现实数据实验的训练、评估、日志与落盘。
	"""

	setup = setup_experiment(args)
	save_json(setup.run_dir / "args.json", vars(args))

	tracking_backend = try_start_mlflow(
		args=args,
		output_root=Path(args.output_dir).resolve(),
		run_name=args.experiment_name,
		params={
			"experiment": args.experiment_name,
			"target_column": args.target_column,
			"feature_bundle": getattr(args, "feature_bundle", DEFAULT_ECONOMICS_FEATURE_BUNDLE),
			"seed": args.seed,
			"lr": args.lr,
			"epochs": args.epochs,
			"patience": args.patience,
			"lambda_r": args.lambda_r,
			"temperature": args.temperature,
			"lag_bias_strength": args.lag_bias_strength,
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
				torch.save(
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
		torch.save(
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
			raise RuntimeError("Expected test outputs from economics evaluation")

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


def main() -> None:
	"""Execute the economics real-data experiment.

	执行 economics 现实数据实验，并打印关键指标摘要。
	"""

	args = parse_args()
	summary = run_experiment(args)

	test_metrics = summary["metrics"]["test"]
	print("Economics experiment complete.")
	print(
		f"best_epoch={summary['best_epoch']} "
		f"test_mse={test_metrics['mse']:.4f} "
		f"test_mae={test_metrics['mae']:.4f} "
		f"test_r2={test_metrics['r2']:.4f} "
		f"proxy_recon_r2={test_metrics['proxy_recon_r2']:.4f}"
	)


if __name__ == "__main__":
	main()
