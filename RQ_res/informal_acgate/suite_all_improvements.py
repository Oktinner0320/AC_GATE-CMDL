"""One-shot runner for all Informal RQ improvement variants."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
RQ_RES_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = RQ_RES_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(RQ_RES_ROOT) not in sys.path:
    sys.path.insert(0, str(RQ_RES_ROOT))

from informal_acgate.aggregate_multi import aggregate_matrix
from informal_acgate.experiment_matrix import (
    DEFAULT_MATRIX_ROOT,
    DEFAULT_REPORT_ROOT,
    DEFAULT_SEEDS,
    MatrixTask,
    build_default_matrix,
    build_tasks,
    select_variants,
)
from informal_acgate.runner import run_experiment


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=str)


def cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def ensure_inside_rq_res(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(RQ_RES_ROOT.resolve()):
        raise ValueError(f"RQ improvement outputs must stay under {RQ_RES_ROOT}; got {resolved}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Informal RQ improvement matrix under RQ_res.")
    parser.add_argument("--matrix-name", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_MATRIX_ROOT))
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--screening-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--use-all-seeds-for-screening", action="store_true")
    parser.add_argument("--include-track", nargs="+", default=None)
    parser.add_argument("--exclude-track", nargs="+", default=None)
    parser.add_argument("--include-variant", nargs="+", default=None)
    parser.add_argument("--expanded-grid", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args()


def _matrix_name(args: argparse.Namespace) -> str:
    if args.matrix_name:
        return str(args.matrix_name)
    return "default_smoke" if args.smoke or args.dry_run else "default"


def _variant_seeds(args: argparse.Namespace, track: str) -> list[int]:
    if args.use_all_seeds_for_screening:
        return [int(seed) for seed in args.seeds]
    if track in {"capacity_gate_grid", "falsification"}:
        return [int(seed) for seed in args.screening_seeds]
    return [int(seed) for seed in args.seeds]


def _build_matrix_tasks(args: argparse.Namespace, matrix_dir: Path) -> list[MatrixTask]:
    variants = select_variants(
        build_default_matrix(expanded_grid=bool(args.expanded_grid)),
        include_tracks=args.include_track,
        exclude_tracks=args.exclude_track,
        include_variants=args.include_variant,
    )
    tasks: list[MatrixTask] = []
    for variant in variants:
        tasks.extend(
            build_tasks(
                variants=[variant],
                seeds=_variant_seeds(args, variant.track),
                matrix_dir=matrix_dir,
                device=args.device,
                epochs=args.epochs,
                patience=args.patience,
                smoke=bool(args.smoke),
            )
        )
    return tasks


def _write_manifest(matrix_dir: Path, matrix_name: str, tasks: list[MatrixTask], args: argparse.Namespace) -> Path:
    variants: dict[str, dict[str, Any]] = {}
    for task in tasks:
        variants[task.variant.variant_id] = task.variant.to_dict()
    manifest = {
        "matrix_name": matrix_name,
        "created_at": now_iso(),
        "dry_run": bool(args.dry_run),
        "smoke": bool(args.smoke),
        "expanded_grid": bool(args.expanded_grid),
        "task_count": len(tasks),
        "variant_count": len(variants),
        "variants": list(variants.values()),
        "tasks": [task.metadata(matrix_name) for task in tasks],
    }
    manifest_path = matrix_dir / "matrix_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def _summary_exists(task: MatrixTask) -> bool:
    return (task.run_dir / "summary.json").exists()


def _write_task_status(task: MatrixTask, status: str, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "status": status,
        "updated_at": now_iso(),
        "variant_id": task.variant.variant_id,
        "track": task.variant.track,
        "run_spec": task.run_spec.name,
        "seed": int(task.seed),
        "experiment_name": task.experiment_name,
        "run_dir": str(task.run_dir),
    }
    if extra:
        payload.update(extra)
    write_json(task.run_dir / "matrix_task_status.json", payload)


def run_task(task: MatrixTask, matrix_name: str, force: bool) -> str:
    task.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(task.run_dir / "matrix_task.json", task.metadata(matrix_name))
    if _summary_exists(task) and not force:
        _write_task_status(task, "success", {"skipped_existing_summary": True})
        print(f"[skip] {task.experiment_name}", flush=True)
        return "skipped"
    print(f"[run] {task.experiment_name}", flush=True)
    run_experiment(task.args)
    if not _summary_exists(task):
        raise RuntimeError(f"summary.json was not created for {task.experiment_name}")
    _write_task_status(task, "success", {"skipped_existing_summary": False})
    cleanup()
    return "success"


def main() -> None:
    args = parse_args()
    if args.max_workers != 1:
        print("[info] max-workers is accepted for the matrix contract; execution is sequential in this CPU-safe runner.")
    matrix_name = _matrix_name(args)
    matrix_dir = Path(args.output_root).resolve() / matrix_name
    report_dir = Path(args.report_root).resolve() / matrix_name
    ensure_inside_rq_res(matrix_dir)
    ensure_inside_rq_res(report_dir)
    matrix_dir.mkdir(parents=True, exist_ok=True)
    tasks = _build_matrix_tasks(args, matrix_dir)
    manifest_path = _write_manifest(matrix_dir, matrix_name, tasks, args)
    print(f"Matrix manifest: {manifest_path}")
    print(f"Variants: {len({task.variant.variant_id for task in tasks})}; tasks: {len(tasks)}")

    if args.dry_run:
        for task in tasks:
            print(
                f"[dry-run] {task.variant.track} | {task.variant.variant_id} | "
                f"{task.run_spec.name} | seed={task.seed} | {task.run_dir}"
            )
        return

    successes = 0
    skips = 0
    failures = 0
    for task in tasks:
        try:
            result = run_task(task, matrix_name, force=bool(args.force))
            if result == "skipped":
                skips += 1
            else:
                successes += 1
        except Exception as exc:
            failures += 1
            _write_task_status(
                task,
                "failed",
                {"error": str(exc), "traceback": traceback.format_exc()},
            )
            print(f"[failed] {task.experiment_name}: {exc}", flush=True)
            cleanup()
            if args.stop_on_error:
                raise

    print(f"Matrix run complete: success={successes}, skipped={skips}, failed={failures}")
    if not args.no_aggregate:
        paths = aggregate_matrix(matrix_dir=matrix_dir, report_dir=report_dir)
        print(f"Aggregated matrix outputs into {report_dir}")
        for name, path in paths.items():
            print(f"{name}: {path}")
    if not args.no_report:
        from informal_acgate.report_improvements import build_report

        report_path = build_report(matrix_dir=matrix_dir, report_dir=report_dir, build_figures=True)
        print(f"Report: {report_path}")


if __name__ == "__main__":
    main()