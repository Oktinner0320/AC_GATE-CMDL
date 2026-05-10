from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent
DATA_DIR = DEST / "data"
RUNS_DIR = DEST / "runs"
SRC_VARIANT_DIR = (
    ROOT
    / "RQ_res"
    / "outputs"
    / "informal_acgate"
    / "improvement_matrix"
    / "full_all_improvements"
    / "fullspan_income_k3"
)
REPORT_DIR = (
    ROOT
    / "RQ_res"
    / "outputs"
    / "informal_acgate"
    / "improvement_report"
    / "full_all_improvements"
)

COPIED_FILES = [
    "args.json",
    "history.csv",
    "history.json",
    "matrix_task.json",
    "matrix_task_status.json",
    "panel_audit.json",
    "predictions.csv",
    "summary.json",
]


def seed_from_name(name: str) -> int:
    return int(name.rsplit("seed", 1)[1])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def numeric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is not None and value != "":
            values.append(float(value))
    return values


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        SRC_VARIANT_DIR.glob("fullspan_income_k3_cmdl_seed*"),
        key=lambda path: seed_from_name(path.name),
    )
    if len(run_dirs) != 20:
        raise RuntimeError(f"Expected 20 CMDL seed directories, found {len(run_dirs)}")

    for run_dir in run_dirs:
        seed = seed_from_name(run_dir.name)
        seed_dir = RUNS_DIR / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for name in COPIED_FILES:
            shutil.copy2(run_dir / name, seed_dir / name)

    all_runs = [
        row
        for row in read_csv(REPORT_DIR / "all_runs.csv")
        if row.get("variant_id") == "fullspan_income_k3" and row.get("run_group") == "CMDL"
    ]
    all_runs.sort(key=lambda row: int(row["seed"]))
    write_csv(DATA_DIR / "runs_summary.csv", all_runs)

    variant_rows = [
        row
        for row in read_csv(REPORT_DIR / "variant_summary.csv")
        if row.get("variant_id") == "fullspan_income_k3" and row.get("run_group") == "CMDL"
    ]
    write_csv(DATA_DIR / "variant_summary_cmdl.csv", variant_rows)

    predictions: list[dict[str, object]] = []
    entity_by_seed: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    baseline_by_seed: list[dict[str, object]] = []

    for run_dir in run_dirs:
        seed = seed_from_name(run_dir.name)
        experiment = run_dir.name
        summary = read_json(run_dir / "summary.json")
        test_metrics = summary["metrics"]["test"]
        baseline_by_seed.append(
            {
                "seed": seed,
                "experiment": experiment,
                "cmdl_mse": test_metrics.get("mse"),
                "cmdl_mae": test_metrics.get("mae"),
                "train_mean_mse": test_metrics.get("baseline_train_mean_mse"),
                "train_mean_mae": test_metrics.get("baseline_train_mean_mae"),
                "persistence_mse": test_metrics.get("baseline_persistence_mse"),
                "persistence_mae": test_metrics.get("baseline_persistence_mae"),
                "panel_ols_mse": test_metrics.get("baseline_panel_ols_mse"),
                "panel_ols_mae": test_metrics.get("baseline_panel_ols_mae"),
                "grouped_ardl_mse": test_metrics.get("baseline_grouped_ardl_mse"),
                "grouped_ardl_mae": test_metrics.get("baseline_grouped_ardl_mae"),
                "grouped_ardl_effective_lag_mean": test_metrics.get(
                    "baseline_grouped_ardl_effective_lag_mean"
                ),
            }
        )
        pred_rows = read_csv(run_dir / "predictions.csv")
        for row in pred_rows:
            y_true = float(row["y_true"])
            y_pred = float(row["y_pred"])
            error = y_pred - y_true
            predictions.append(
                {
                    "seed": seed,
                    "experiment": experiment,
                    "entity_id": int(row["entity_id"]),
                    "entity_code": row["entity_code"],
                    "entity_name": row["entity_name"],
                    "year": int(row["year"]),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "error": error,
                    "abs_error": abs(error),
                    "squared_error": error * error,
                    "k_star": float(row["k_star"]),
                    "omega_peak": int(row["omega_peak"]),
                    "omega_1": float(row["omega_1"]),
                    "omega_2": float(row["omega_2"]),
                    "omega_3": float(row["omega_3"]),
                    "proxy_income_level_true": float(row["proxy_1_true"]),
                    "proxy_income_level_pred": float(row["proxy_1_pred"]),
                    "proxy_income_recent_level_true": float(row["proxy_2_true"]),
                    "proxy_income_recent_level_pred": float(row["proxy_2_pred"]),
                    "proxy_income_growth_signal_true": float(row["proxy_3_true"]),
                    "proxy_income_growth_signal_pred": float(row["proxy_3_pred"]),
                }
            )

        by_entity: dict[int, list[dict[str, str]]] = {}
        for row in pred_rows:
            by_entity.setdefault(int(row["entity_id"]), []).append(row)
        for rows in by_entity.values():
            first = rows[0]
            errors = [float(row["y_pred"]) - float(row["y_true"]) for row in rows]
            entity_by_seed.append(
                {
                    "seed": seed,
                    "experiment": experiment,
                    "entity_id": int(first["entity_id"]),
                    "entity_code": first["entity_code"],
                    "entity_name": first["entity_name"],
                    "test_year_count": len(rows),
                    "entity_test_mse": mean([error * error for error in errors]),
                    "entity_test_mae": mean([abs(error) for error in errors]),
                    "k_star": float(first["k_star"]),
                    "omega_peak": int(first["omega_peak"]),
                    "omega_1": float(first["omega_1"]),
                    "omega_2": float(first["omega_2"]),
                    "omega_3": float(first["omega_3"]),
                    "proxy_income_level_true": float(first["proxy_1_true"]),
                    "proxy_income_level_pred": float(first["proxy_1_pred"]),
                    "proxy_income_recent_level_true": float(first["proxy_2_true"]),
                    "proxy_income_recent_level_pred": float(first["proxy_2_pred"]),
                    "proxy_income_growth_signal_true": float(first["proxy_3_true"]),
                    "proxy_income_growth_signal_pred": float(first["proxy_3_pred"]),
                }
            )

        for row in read_csv(run_dir / "history.csv"):
            histories.append(
                {
                    "seed": seed,
                    "experiment": experiment,
                    "epoch": int(float(row["epoch"])),
                    "train_total_loss": to_float(row.get("train_total_loss")),
                    "train_task_loss": to_float(row.get("train_task_loss")),
                    "train_recon_loss": to_float(row.get("train_recon_loss")),
                    "val_total_loss": to_float(row.get("val_total_loss")),
                    "val_task_loss": to_float(row.get("val_task_loss")),
                    "val_mse": to_float(row.get("val_mse")),
                    "val_mae": to_float(row.get("val_mae")),
                    "val_kstar_mean": to_float(row.get("val_kstar_mean")),
                    "val_kstar_std": to_float(row.get("val_kstar_std")),
                    "val_kstar_proxy_spearman_adjusted_rho": to_float(
                        row.get("val_kstar_proxy_spearman_adjusted_rho")
                    ),
                    "val_lag_gate_sensitivity_range": to_float(
                        row.get("val_lag_gate_sensitivity_range")
                    ),
                    "val_omega_entropy_mean": to_float(row.get("val_omega_entropy_mean")),
                    "val_omega_peak_share_1": to_float(row.get("val_omega_peak_share_1")),
                    "val_omega_peak_share_2": to_float(row.get("val_omega_peak_share_2")),
                    "val_omega_peak_share_3": to_float(row.get("val_omega_peak_share_3")),
                }
            )

    write_csv(DATA_DIR / "predictions_all_seeds.csv", predictions)
    write_csv(DATA_DIR / "history_all_seeds.csv", histories)
    write_csv(DATA_DIR / "entity_summary_by_seed.csv", entity_by_seed)
    write_csv(DATA_DIR / "baseline_metrics_by_seed.csv", baseline_by_seed)

    baseline_summary_rows: list[dict[str, object]] = []
    for label, mse_key, mae_key in [
        ("CMDL", "cmdl_mse", "cmdl_mae"),
        ("Train mean", "train_mean_mse", "train_mean_mae"),
        ("Persistence", "persistence_mse", "persistence_mae"),
        ("Panel OLS", "panel_ols_mse", "panel_ols_mae"),
        ("Grouped ARDL", "grouped_ardl_mse", "grouped_ardl_mae"),
    ]:
        mse_values = numeric_values(baseline_by_seed, mse_key)
        mae_values = numeric_values(baseline_by_seed, mae_key)
        baseline_summary_rows.append(
            {
                "model": label,
                "seed_count": len(mse_values),
                "test_mse_mean": mean(mse_values),
                "test_mse_std": sample_std(mse_values),
                "test_mae_mean": mean(mae_values),
                "test_mae_std": sample_std(mae_values),
            }
        )
    write_csv(DATA_DIR / "baseline_summary.csv", baseline_summary_rows)

    entity_mean_rows: list[dict[str, object]] = []
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in entity_by_seed:
        grouped.setdefault(int(row["entity_id"]), []).append(row)
    for entity_id, rows in sorted(grouped.items()):
        first = rows[0]
        peak_counts: dict[int, int] = {}
        for row in rows:
            peak = int(row["omega_peak"])
            peak_counts[peak] = peak_counts.get(peak, 0) + 1
        peak_mode = sorted(peak_counts.items(), key=lambda item: (-item[1], item[0]))[0]
        entity_mean_rows.append(
            {
                "entity_id": entity_id,
                "entity_code": first["entity_code"],
                "entity_name": first["entity_name"],
                "seed_count": len(rows),
                "proxy_income_level_true": mean(numeric_values(rows, "proxy_income_level_true")),
                "proxy_income_recent_level_true": mean(
                    numeric_values(rows, "proxy_income_recent_level_true")
                ),
                "proxy_income_growth_signal_true": mean(
                    numeric_values(rows, "proxy_income_growth_signal_true")
                ),
                "k_star_mean": mean(numeric_values(rows, "k_star")),
                "k_star_std": sample_std(numeric_values(rows, "k_star")),
                "omega_1_mean": mean(numeric_values(rows, "omega_1")),
                "omega_2_mean": mean(numeric_values(rows, "omega_2")),
                "omega_3_mean": mean(numeric_values(rows, "omega_3")),
                "omega_peak_mode": peak_mode[0],
                "omega_peak_mode_count": peak_mode[1],
                "entity_test_mse_mean": mean(numeric_values(rows, "entity_test_mse")),
                "entity_test_mae_mean": mean(numeric_values(rows, "entity_test_mae")),
            }
        )
    write_csv(DATA_DIR / "entity_summary_seed_mean.csv", entity_mean_rows)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_variant_dir": str(SRC_VARIANT_DIR),
        "source_report_dir": str(REPORT_DIR),
        "destination": str(DEST),
        "included_variant": "fullspan_income_k3",
        "included_run_group": "CMDL",
        "seed_count": len(run_dirs),
        "copied_per_seed_files": COPIED_FILES,
        "excluded_files": ["best_model.pt"],
        "generated_files": [
            "data/runs_summary.csv",
            "data/variant_summary_cmdl.csv",
            "data/predictions_all_seeds.csv",
            "data/history_all_seeds.csv",
            "data/entity_summary_by_seed.csv",
            "data/entity_summary_seed_mean.csv",
            "data/baseline_metrics_by_seed.csv",
            "data/baseline_summary.csv",
        ],
    }
    (DATA_DIR / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"archived_seeds={len(run_dirs)}")
    print(f"prediction_rows={len(predictions)}")
    print(f"history_rows={len(histories)}")
    print(f"entity_seed_rows={len(entity_by_seed)}")
    print(f"entity_mean_rows={len(entity_mean_rows)}")
    print(f"baseline_rows={len(baseline_by_seed)}")


if __name__ == "__main__":
    main()