"""Synthetic ablation runner for the three core CMDL variants.

该脚本复用当前 Step 4 的训练与评估链，只替换模型或损失的关键部件，
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
from torch import nn

from config.cmdl_config import CMDLConfig
from experiments._checkpoint_io import save_torch_checkpoint
from experiments._runtime_meta import attach_runtime_metadata, start_runtime_timer
from experiments.run_synthetic import (
    ExperimentResult,
    evaluate,
    finish_mlflow,
    log_mlflow_metrics,
    refit_proxy_reconstructor,
    save_json,
    save_predictions,
    select_device,
    set_seed,
    split_entity_indices,
    subset_panel,
    summarize_run,
    train_one_epoch,
    try_start_mlflow,
)
from data.synthetic.generate import SyntheticPanel, generate_cmdl_synthetic
from model.cmdl_model import CMDLModel, CMDLModelOutput
from model.loss import DomainAgnosticLoss
from visualization.kstar_distribution import plot_kstar_scatter
from visualization.omega_heatmap import plot_omega_heatmap


VARIANT_CHOICES = ["all", "no_ac_encoder", "uniform_lag", "no_recon_regularization"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the synthetic ablation suite.

    解析 synthetic ablation 运行所需的命令行参数。
    """

    synthetic_defaults = CMDLConfig.from_domain("synthetic")

    parser = argparse.ArgumentParser(description="Run Step 4.5 synthetic ablations.")
    parser.add_argument("--variant", choices=VARIANT_CHOICES, default="all")
    parser.add_argument("--scenario", choices=["all", "linear", "nonlinear"], default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[synthetic_defaults.seed])
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
    parser.add_argument("--output-dir", type=str, default="outputs/step4_ablation")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def _validate_cmdl_inputs(
    cfg: CMDLConfig,
    entity_ids: torch.Tensor,
    X_it: torch.Tensor,
    p_i: torch.Tensor,
    s_i: torch.Tensor,
    macro_controls: torch.Tensor | None,
) -> None:
    if entity_ids.dim() != 1:
        raise ValueError(f"Expected entity_ids with shape [B], got {tuple(entity_ids.shape)}")
    if X_it.dim() != 3 or X_it.size(-1) != cfg.seq_features:
        raise ValueError(f"Expected X_it with shape [B, T, {cfg.seq_features}], got {tuple(X_it.shape)}")
    if p_i.dim() != 2 or p_i.size(-1) != cfg.n_proxies:
        raise ValueError(f"Expected p_i with shape [B, {cfg.n_proxies}], got {tuple(p_i.shape)}")
    if s_i.dim() != 2 or s_i.size(-1) != cfg.static_dim:
        raise ValueError(f"Expected s_i with shape [B, {cfg.static_dim}], got {tuple(s_i.shape)}")

    batch_size = entity_ids.size(0)
    if X_it.size(0) != batch_size or p_i.size(0) != batch_size or s_i.size(0) != batch_size:
        raise ValueError("entity_ids, X_it, p_i, and s_i must agree on batch size")
    if macro_controls is not None and macro_controls.shape[:2] != X_it.shape[:2]:
        raise ValueError("macro_controls must align with X_it on the batch and time dimensions")


class NoACEncoderCMDLModel(CMDLModel):
    """Ablation A: replace entity-specific z with one global scalar.

    该变体移除实体级 AC 条件化，使所有实体共享一套 lag 分布和 z 初始化信号。
    """

    def __init__(self, cfg: CMDLConfig) -> None:
        super().__init__(cfg)
        self.shared_z = nn.Parameter(torch.zeros(1, 1))

    def forward(
        self,
        entity_ids: torch.Tensor,
        X_it: torch.Tensor,
        p_i: torch.Tensor,
        s_i: torch.Tensor,
        macro_controls: torch.Tensor | None = None,
    ) -> CMDLModelOutput:
        _validate_cmdl_inputs(self.cfg, entity_ids, X_it, p_i, s_i, macro_controls)

        batch_size = entity_ids.size(0)
        shared_z = self.shared_z.expand(batch_size, -1)
        adapted_sequence = self.input_adapter(X_it)
        lag_gate_output = self.lag_gate(shared_z)

        lagged_windows = self._build_lagged_windows(adapted_sequence)
        lag_context_sequence = torch.einsum("bk,btkd->btd", lag_gate_output.omega, lagged_windows)
        current_sequence = adapted_sequence[:, self.cfg.max_lag :, :]
        valid_macro_controls = None if macro_controls is None else macro_controls[:, self.cfg.max_lag :, :]
        entity_context = self.entity_embedding(entity_ids)

        backbone_output = self.backbone(
            current_sequence=current_sequence,
            lag_context_sequence=lag_context_sequence,
            entity_embedding=entity_context,
            static_features=s_i,
            z_i=shared_z,
            macro_controls=valid_macro_controls,
        )
        y_pred = self.regression_head(backbone_output.sequence).squeeze(-1)
        p_hat_i = self.ac_encoder.proxy_reconstructor(shared_z.detach())

        return CMDLModelOutput(
            y_pred=y_pred,
            omega=lag_gate_output.omega,
            z_i=shared_z,
            p_hat_i=p_hat_i,
            k_star=lag_gate_output.k_star,
            lag_context_sequence=lag_context_sequence,
            backbone_sequence=backbone_output.sequence,
        )


class UniformLagCMDLModel(CMDLModel):
    """Ablation B: force lag weights to be uniform over the whole window.

    该变体保留 AC encoder 和 backbone，但固定 omega_k = 1 / K。
    """

    def forward(
        self,
        entity_ids: torch.Tensor,
        X_it: torch.Tensor,
        p_i: torch.Tensor,
        s_i: torch.Tensor,
        macro_controls: torch.Tensor | None = None,
    ) -> CMDLModelOutput:
        _validate_cmdl_inputs(self.cfg, entity_ids, X_it, p_i, s_i, macro_controls)

        encoder_output = self.ac_encoder(p_i)
        adapted_sequence = self.input_adapter(X_it)
        batch_size = entity_ids.size(0)

        omega = adapted_sequence.new_full((batch_size, self.cfg.max_lag), 1.0 / self.cfg.max_lag)
        k_star = adapted_sequence.new_full((batch_size,), (self.cfg.max_lag + 1) / 2.0)

        lagged_windows = self._build_lagged_windows(adapted_sequence)
        lag_context_sequence = torch.einsum("bk,btkd->btd", omega, lagged_windows)
        current_sequence = adapted_sequence[:, self.cfg.max_lag :, :]
        valid_macro_controls = None if macro_controls is None else macro_controls[:, self.cfg.max_lag :, :]
        entity_context = self.entity_embedding(entity_ids)

        backbone_output = self.backbone(
            current_sequence=current_sequence,
            lag_context_sequence=lag_context_sequence,
            entity_embedding=entity_context,
            static_features=s_i,
            z_i=encoder_output.z_i,
            macro_controls=valid_macro_controls,
        )
        y_pred = self.regression_head(backbone_output.sequence).squeeze(-1)

        return CMDLModelOutput(
            y_pred=y_pred,
            omega=omega,
            z_i=encoder_output.z_i,
            p_hat_i=encoder_output.p_hat_i,
            k_star=k_star,
            lag_context_sequence=lag_context_sequence,
            backbone_sequence=backbone_output.sequence,
        )


def move_panel_to_device(panel: SyntheticPanel, device: torch.device) -> SyntheticPanel:
    """Move every tensor field of a synthetic panel onto one device.

    将 synthetic panel 中的全部张量字段移动到同一设备。
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


def prepare_panels(args: argparse.Namespace, scenario: str, seed: int) -> tuple[CMDLConfig, SyntheticPanel, SyntheticPanel, SyntheticPanel, torch.device]:
    """Prepare full/train/val synthetic panels for one ablation run.

    为单次 ablation 运行准备完整、训练和验证面板。
    """

    cfg = CMDLConfig.from_domain(
        "synthetic",
        seed=seed,
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
    return cfg, full_panel, train_panel, val_panel, device


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


def run_variant(
    args: argparse.Namespace,
    variant: str,
    scenario: str,
    seed: int,
) -> ExperimentResult:
    """Run one ablation variant on one synthetic scenario and seed.

    在给定场景和随机种子下运行一个 ablation 变体。
    """

    run_started_at = start_runtime_timer()
    cfg, full_panel, train_panel, val_panel, device = prepare_panels(args, scenario, seed)
    model, effective_lambda_r = build_variant_model(variant, cfg)
    model = model.to(device)
    criterion = DomainAgnosticLoss(lambda_r=effective_lambda_r, warmup_steps=cfg.max_lag)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    experiment_name = f"{variant}_{scenario}_seed{seed}"
    run_dir = Path(args.output_dir).resolve() / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history_json_path = run_dir / "history.json"
    history_csv_path = run_dir / "history.csv"
    summary_path = run_dir / "summary.json"
    predictions_path = run_dir / "predictions.csv"
    checkpoint_path = run_dir / "best_model.pt"

    save_json(run_dir / "args.json", vars(args))

    tracking_backend = try_start_mlflow(
        args=args,
        output_root=Path(args.output_dir).resolve(),
        run_name=experiment_name,
        params={
            "experiment": experiment_name,
            "variant": variant,
            "scenario": scenario,
            "seed": seed,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "lambda_r": args.lambda_r,
            "temperature": args.temperature,
            "lag_bias_strength": args.lag_bias_strength,
            "grad_clip": args.grad_clip,
            "val_fraction": args.val_fraction,
            "effective_lambda_r": effective_lambda_r,
        },
    )

    history: list[dict[str, float]] = []
    best_epoch = 0
    best_val_task_loss = float("inf")
    best_val_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                panel=train_panel,
                grad_clip=args.grad_clip,
            )
            val_metrics, _ = evaluate(model, criterion, val_panel)

            epoch_record: dict[str, float] = {"epoch": float(epoch)}
            epoch_record.update({f"train_{key}": value for key, value in train_metrics.items()})
            epoch_record.update({f"val_{key}": value for key, value in val_metrics.items()})
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

            val_score = val_metrics["task_loss"] + val_metrics["kstar_mae"]
            if val_score < best_val_score - 1e-8:
                best_val_score = float(val_score)
                best_val_task_loss = float(val_metrics["task_loss"])
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
                save_torch_checkpoint(
                    {
                        "experiment": experiment_name,
                        "variant": variant,
                        "best_epoch": best_epoch,
                        "best_val_task_loss": best_val_task_loss,
                        "config": cfg.to_dict(),
                        "model_state_dict": best_state,
                    },
                    checkpoint_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"[{experiment_name}] early stopping at epoch {epoch}")
                    break

        model.load_state_dict(best_state)
        refit_proxy_reconstructor(model, train_panel)
        save_torch_checkpoint(
            {
                "experiment": experiment_name,
                "variant": variant,
                "best_epoch": best_epoch,
                "best_val_task_loss": best_val_task_loss,
                "config": cfg.to_dict(),
                "proxy_head_refit": True,
                "model_state_dict": copy.deepcopy(model.state_dict()),
            },
            checkpoint_path,
        )

        final_metrics, outputs = evaluate(model, criterion, full_panel, include_outputs=True)
        if outputs is None:
            raise RuntimeError("Expected outputs from final ablation evaluation")

        pd.DataFrame(history).to_csv(history_csv_path, index=False)
        save_json(history_json_path, history)
        save_predictions(outputs, predictions_path)

        omega_ax = plot_omega_heatmap(
            outputs["omega"],
            outputs["z_true"],
            outputs["kstar_true"],
            run_dir / "omega_heatmap.png",
        )
        kstar_ax = plot_kstar_scatter(
            outputs["k_star"],
            outputs["kstar_true"],
            run_dir / "kstar_scatter.png",
            z_values=outputs["z_true"],
        )
        omega_ax.figure.clf()
        kstar_ax.figure.clf()

        summary = summarize_run(
            setup=type(
                "AblationSummaryProxy",
                (),
                {"cfg": cfg, "device": device},
            )(),
            experiment_name=experiment_name,
            tracking_backend=tracking_backend,
            best_epoch=best_epoch,
            best_val_task_loss=best_val_task_loss,
            metrics=final_metrics,
        )
        summary["variant"] = variant
        summary["effective_lambda_r"] = float(effective_lambda_r)
        attach_runtime_metadata(summary, device, started_at=run_started_at)
        save_json(summary_path, summary)

        return ExperimentResult(
            experiment_name=experiment_name,
            scenario=scenario,
            run_dir=run_dir,
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
                run_dir / "args.json",
                history_json_path,
                history_csv_path,
                summary_path,
                predictions_path,
                checkpoint_path,
                run_dir / "omega_heatmap.png",
                run_dir / "kstar_scatter.png",
            ]
        )


def selected_variants(variant: str) -> list[str]:
    """Expand the CLI variant selector into concrete ablation variants.

    将命令行的 variant 选择器展开为具体变体列表。
    """

    if variant == "all":
        return [choice for choice in VARIANT_CHOICES if choice != "all"]
    return [variant]


def selected_scenarios(scenario: str) -> list[str]:
    """Expand the CLI scenario selector into concrete scenario values.

    将命令行的 scenario 选择器展开为具体场景列表。
    """

    if scenario == "all":
        return ["linear", "nonlinear"]
    return [scenario]


def aggregate_results(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate repeated-seed ablation results into mean/std tables.

    将多随机种子的 ablation 结果聚合为 mean/std 表。
    """

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    numeric_columns = [
        column
        for column in frame.columns
        if column not in {"experiment", "variant", "scenario", "seed"}
    ]
    aggregated = frame.groupby(["variant", "scenario"], as_index=False)[numeric_columns].agg(["mean", "std"])
    aggregated.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in aggregated.columns.to_flat_index()
    ]
    return aggregated


def main() -> None:
    """Execute the requested synthetic ablation runs and collect summary tables.

    按参数执行 synthetic ablation，并输出逐 run 与聚合汇总结果。
    """

    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for variant in selected_variants(args.variant):
        for scenario in selected_scenarios(args.scenario):
            for seed in args.seeds:
                result = run_variant(args=args, variant=variant, scenario=scenario, seed=seed)
                summary_rows.append(
                    {
                        "experiment": result.experiment_name,
                        "variant": variant,
                        "scenario": scenario,
                        "seed": seed,
                        "best_epoch": result.best_epoch,
                        "best_val_task_loss": result.best_val_task_loss,
                        **result.metrics,
                    }
                )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_root / "ablation_results.csv", index=False)
    save_json(output_root / "ablation_results.json", summary_rows)

    aggregated = aggregate_results(summary_rows)
    if not aggregated.empty:
        aggregated.to_csv(output_root / "ablation_results_aggregated.csv", index=False)

    print("Synthetic ablation experiments complete.")
    print(summary_frame.to_string(index=False))
    if not aggregated.empty:
        print("\nAggregated across seeds:")
        print(aggregated.to_string(index=False))


if __name__ == "__main__":
    main()