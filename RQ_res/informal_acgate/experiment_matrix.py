"""Experiment matrix definitions for the Informal RQ improvement suite."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parent
RQ_RES_ROOT = PACKAGE_ROOT.parent
DEFAULT_MATRIX_ROOT = RQ_RES_ROOT / "outputs" / "informal_acgate" / "improvement_matrix"
DEFAULT_REPORT_ROOT = RQ_RES_ROOT / "outputs" / "informal_acgate" / "improvement_report"
DEFAULT_SEEDS = tuple(range(20))


@dataclass(frozen=True)
class RunSpec:
    name: str
    model: str
    ablation: str = "none"


@dataclass(frozen=True)
class ExperimentVariant:
    variant_id: str
    track: str
    scenario: str
    feature_bundle: str
    max_lag: int
    run_specs: tuple[RunSpec, ...]
    year_start: int | None = None
    year_end: int | None = None
    stats_end_year: int | None = None
    train_end_year: int = 2021
    val_end_year: int = 2022
    proxy_perturbation: str = "none"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    interpretation: str = "sensitivity"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_specs"] = [asdict(spec) for spec in self.run_specs]
        return payload


@dataclass(frozen=True)
class MatrixTask:
    variant: ExperimentVariant
    run_spec: RunSpec
    seed: int
    output_dir: Path
    experiment_name: str
    args: Namespace

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.experiment_name

    def metadata(self, matrix_name: str) -> dict[str, Any]:
        return {
            "matrix_name": matrix_name,
            "variant": self.variant.to_dict(),
            "run_spec": asdict(self.run_spec),
            "seed": int(self.seed),
            "output_dir": str(self.output_dir),
            "experiment_name": self.experiment_name,
            "run_dir": str(self.run_dir),
        }


CMDL = RunSpec("cmdl", "cmdl", "none")
PLAIN_LSTM = RunSpec("plain_lstm", "plain_lstm", "none")
NO_AC = RunSpec("no_ac_encoder", "cmdl", "no_ac_encoder")
UNIFORM_LAG = RunSpec("uniform_lag", "cmdl", "uniform_lag")
NO_RECON = RunSpec("no_recon_regularization", "cmdl", "no_recon_regularization")

CORE_RUN_SPECS = (CMDL, PLAIN_LSTM, NO_AC, UNIFORM_LAG, NO_RECON)
PREDICTION_RUN_SPECS = (CMDL, PLAIN_LSTM)
CMDL_ONLY = (CMDL,)


def _fullspan_variant(
    variant_id: str,
    track: str,
    feature_bundle: str,
    max_lag: int,
    run_specs: tuple[RunSpec, ...],
    interpretation: str,
    hyperparameters: dict[str, Any] | None = None,
    proxy_perturbation: str = "none",
) -> ExperimentVariant:
    return ExperimentVariant(
        variant_id=variant_id,
        track=track,
        scenario="fullspan_region_proxy",
        feature_bundle=feature_bundle,
        max_lag=max_lag,
        run_specs=run_specs,
        year_start=2006,
        year_end=2023,
        stats_end_year=2018,
        train_end_year=2018,
        val_end_year=2020,
        proxy_perturbation=proxy_perturbation,
        hyperparameters={} if hyperparameters is None else dict(hyperparameters),
        interpretation=interpretation,
    )


def _overlap_variant(
    variant_id: str,
    track: str,
    feature_bundle: str,
    run_specs: tuple[RunSpec, ...],
    interpretation: str,
    hyperparameters: dict[str, Any] | None = None,
    proxy_perturbation: str = "none",
) -> ExperimentVariant:
    return ExperimentVariant(
        variant_id=variant_id,
        track=track,
        scenario="overlap_region_proxy",
        feature_bundle=feature_bundle,
        max_lag=2,
        run_specs=run_specs,
        year_start=2019,
        year_end=2023,
        stats_end_year=2021,
        train_end_year=2021,
        val_end_year=2022,
        proxy_perturbation=proxy_perturbation,
        hyperparameters={} if hyperparameters is None else dict(hyperparameters),
        interpretation=interpretation,
    )


def _compact_grid_variants() -> list[ExperimentVariant]:
    grid = [
        ("grid_fullspan_d8_temp10_dropout005", {"d_model": 8, "temperature": 1.0, "dropout": 0.05}),
        ("grid_fullspan_d16_temp15_dropout015", {"d_model": 16, "temperature": 1.5, "dropout": 0.15}),
        ("grid_fullspan_d32_temp10_dropout015", {"d_model": 32, "temperature": 1.0, "dropout": 0.15}),
        ("grid_fullspan_d16_temp20_dropout015", {"d_model": 16, "temperature": 2.0, "dropout": 0.15}),
        ("grid_fullspan_d16_temp15_dropout030", {"d_model": 16, "temperature": 1.5, "dropout": 0.30}),
        (
            "grid_fullspan_d16_temp15_dropout015_entropy001",
            {"d_model": 16, "temperature": 1.5, "dropout": 0.15, "lambda_omega_entropy": 0.01},
        ),
    ]
    return [
        _fullspan_variant(
            variant_id=variant_id,
            track="capacity_gate_grid",
            feature_bundle="single_fullspan_region_proxy",
            max_lag=3,
            run_specs=CMDL_ONLY,
            interpretation="capacity_gate_screen",
            hyperparameters=hyperparameters,
        )
        for variant_id, hyperparameters in grid
    ]


def _expanded_grid_variants() -> list[ExperimentVariant]:
    variants: list[ExperimentVariant] = []
    for d_model in (8, 16, 32):
        for temperature in (1.0, 1.5, 2.0):
            for dropout in (0.05, 0.15, 0.30):
                for entropy in (0.0, 0.01):
                    suffix = f"d{d_model}_temp{int(temperature * 10):02d}_dropout{int(dropout * 1000):03d}"
                    if entropy > 0.0:
                        suffix = f"{suffix}_entropy{int(entropy * 1000):03d}"
                    variants.append(
                        _fullspan_variant(
                            variant_id=f"grid_fullspan_{suffix}",
                            track="capacity_gate_grid",
                            feature_bundle="single_fullspan_region_proxy",
                            max_lag=3,
                            run_specs=CMDL_ONLY,
                            interpretation="capacity_gate_factorial",
                            hyperparameters={
                                "d_model": d_model,
                                "temperature": temperature,
                                "dropout": dropout,
                                "lambda_omega_entropy": entropy,
                            },
                        )
                    )
    return variants


def build_default_matrix(expanded_grid: bool = False) -> list[ExperimentVariant]:
    variants = [
        _overlap_variant(
            "reference_overlap_multiseq_k2",
            "reference",
            "multiseq_overlap_region_proxy",
            CORE_RUN_SPECS,
            "current_overlap_reference",
        ),
        _overlap_variant(
            "reference_overlap_single_k2",
            "reference",
            "single_overlap_region_proxy",
            CMDL_ONLY,
            "current_single_feature_reference",
        ),
        _fullspan_variant(
            "fullspan_income_k3",
            "fullspan_income",
            "single_fullspan_region_proxy",
            3,
            CORE_RUN_SPECS,
            "main_expanded_sample",
        ),
        _fullspan_variant(
            "fullspan_income_k2",
            "fullspan_income",
            "single_fullspan_region_proxy",
            2,
            CORE_RUN_SPECS,
            "lag_window_sensitivity",
        ),
        _fullspan_variant(
            "feature_subset_rpcyd_fullspan_k3",
            "feature_subset",
            "rpcyd_fullspan_region_proxy",
            3,
            PREDICTION_RUN_SPECS,
            "rpcyd_only_fullspan_sensitivity",
        ),
        _overlap_variant(
            "feature_subset_rpcyd_overlap_k2",
            "feature_subset",
            "rpcyd_overlap_region_proxy",
            PREDICTION_RUN_SPECS,
            "rpcyd_only_overlap_sensitivity",
        ),
        _overlap_variant(
            "feature_subset_rydgdp_ratio_overlap_k2",
            "feature_subset",
            "rydgdp_ratio_overlap_region_proxy",
            PREDICTION_RUN_SPECS,
            "rydgdp_ratio_overlap_sensitivity",
        ),
        _overlap_variant(
            "feature_subset_rpcyd_ratio_overlap_k2",
            "feature_subset",
            "rpcyd_ratio_overlap_region_proxy",
            PREDICTION_RUN_SPECS,
            "rpcyd_ratio_overlap_sensitivity",
        ),
        _overlap_variant(
            "feature_subset_no_per_capita_overlap_k2",
            "feature_subset",
            "multiseq_overlap_no_per_capita_region_proxy",
            PREDICTION_RUN_SPECS,
            "no_per_capita_multiseq_sensitivity",
        ),
        _overlap_variant(
            "feature_subset_pca1_overlap_k2",
            "feature_subset",
            "multiseq_overlap_pca1_region_proxy",
            PREDICTION_RUN_SPECS,
            "train_window_pca1_sensitivity",
        ),
        _fullspan_variant(
            "falsification_proxy_shuffle_fullspan_k3",
            "falsification",
            "single_fullspan_region_proxy",
            3,
            CMDL_ONLY,
            "proxy_shuffle_null",
            proxy_perturbation="shuffle",
        ),
        _fullspan_variant(
            "falsification_noise_proxy_fullspan_k3",
            "falsification",
            "single_fullspan_region_proxy",
            3,
            CMDL_ONLY,
            "noise_proxy_null",
            proxy_perturbation="noise",
        ),
    ]
    variants.extend(_expanded_grid_variants() if expanded_grid else _compact_grid_variants())
    return variants


def select_variants(
    variants: Iterable[ExperimentVariant],
    include_tracks: Iterable[str] | None = None,
    exclude_tracks: Iterable[str] | None = None,
    include_variants: Iterable[str] | None = None,
) -> list[ExperimentVariant]:
    include_track_set = {value for value in include_tracks or []}
    exclude_track_set = {value for value in exclude_tracks or []}
    include_variant_set = {value for value in include_variants or []}
    selected: list[ExperimentVariant] = []
    for variant in variants:
        if include_track_set and variant.track not in include_track_set:
            continue
        if exclude_track_set and variant.track in exclude_track_set:
            continue
        if include_variant_set and variant.variant_id not in include_variant_set:
            continue
        selected.append(variant)
    return selected


def build_runner_args(
    variant: ExperimentVariant,
    run_spec: RunSpec,
    seed: int,
    output_dir: Path,
    experiment_name: str,
    device: str,
    epochs: int,
    patience: int,
    smoke: bool,
) -> Namespace:
    hyperparameters = {
        "d_model": 32,
        "lstm_layers": 1,
        "dropout": 0.05,
        "lr": 1e-3,
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
    }
    hyperparameters.update(variant.hyperparameters)
    return Namespace(
        csv_path=None,
        feature_bundle=variant.feature_bundle,
        year_start=variant.year_start,
        year_end=variant.year_end,
        stats_end_year=variant.stats_end_year,
        train_end_year=variant.train_end_year,
        val_end_year=variant.val_end_year,
        missing_policy="error",
        proxy_perturbation=variant.proxy_perturbation,
        proxy_perturbation_seed=int(seed),
        model=run_spec.model,
        ablation=run_spec.ablation,
        seed=int(seed),
        max_lag=variant.max_lag,
        epochs=1 if smoke else int(epochs),
        patience=1 if smoke else int(patience),
        output_dir=str(output_dir),
        experiment_name=experiment_name,
        device=device,
        log_every=10,
        smoke=bool(smoke),
        **hyperparameters,
    )


def build_tasks(
    variants: Iterable[ExperimentVariant],
    seeds: Iterable[int],
    matrix_dir: Path,
    device: str,
    epochs: int,
    patience: int,
    smoke: bool,
) -> list[MatrixTask]:
    tasks: list[MatrixTask] = []
    for variant in variants:
        output_dir = matrix_dir / variant.variant_id
        for seed in seeds:
            for run_spec in variant.run_specs:
                experiment_name = f"{variant.variant_id}_{run_spec.name}_seed{int(seed)}"
                args = build_runner_args(
                    variant=variant,
                    run_spec=run_spec,
                    seed=int(seed),
                    output_dir=output_dir,
                    experiment_name=experiment_name,
                    device=device,
                    epochs=epochs,
                    patience=patience,
                    smoke=smoke,
                )
                tasks.append(
                    MatrixTask(
                        variant=variant,
                        run_spec=run_spec,
                        seed=int(seed),
                        output_dir=output_dir,
                        experiment_name=experiment_name,
                        args=args,
                    )
                )
    return tasks


__all__ = [
    "CMDL_ONLY",
    "CORE_RUN_SPECS",
    "DEFAULT_MATRIX_ROOT",
    "DEFAULT_REPORT_ROOT",
    "DEFAULT_SEEDS",
    "ExperimentVariant",
    "MatrixTask",
    "PREDICTION_RUN_SPECS",
    "RunSpec",
    "build_default_matrix",
    "build_runner_args",
    "build_tasks",
    "select_variants",
]