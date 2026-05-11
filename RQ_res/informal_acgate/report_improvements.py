"""Build a compact Markdown report for the Informal RQ improvement matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


PACKAGE_ROOT = Path(__file__).resolve().parent
RQ_RES_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = RQ_RES_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(RQ_RES_ROOT) not in sys.path:
    sys.path.insert(0, str(RQ_RES_ROOT))

from informal_acgate.aggregate_multi import aggregate_matrix, infer_report_dir
from informal_acgate.visualize import build_matrix_visualizations


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _fmt(value: Any, digits: int = 4) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "NA"
    return f"{float(numeric):.{digits}f}"


def _markdown_table(dataframe: pd.DataFrame, columns: list[str], limit: int = 10) -> str:
    if dataframe.empty:
        return "No rows available."
    rows = dataframe.loc[:, [column for column in columns if column in dataframe.columns]].head(limit)
    if rows.empty:
        return "No requested columns available."
    header = "| " + " | ".join(rows.columns) + " |"
    divider = "| " + " | ".join("---" for _ in rows.columns) + " |"
    body = []
    for _, row in rows.iterrows():
        body.append("| " + " | ".join(str(row[column]) for column in rows.columns) + " |")
    return "\n".join([header, divider, *body])


def _load_manifest(matrix_dir: Path) -> dict[str, Any]:
    manifest_path = matrix_dir / "matrix_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _best_prediction_rows(all_runs: pd.DataFrame) -> pd.DataFrame:
    if all_runs.empty or "test_mse" not in all_runs.columns:
        return pd.DataFrame()
    frame = all_runs.copy()
    frame["test_mse"] = pd.to_numeric(frame["test_mse"], errors="coerce")
    return frame.loc[frame["test_mse"].notna()].sort_values("test_mse", ascending=True)


def _mechanism_rows(all_runs: pd.DataFrame) -> pd.DataFrame:
    if all_runs.empty or "run_group" not in all_runs.columns:
        return pd.DataFrame()
    frame = all_runs.loc[all_runs["run_group"].eq("CMDL")].copy()
    if frame.empty or "test_kstar_proxy_spearman_adjusted_rho" not in frame.columns:
        return pd.DataFrame()
    frame["test_kstar_proxy_spearman_adjusted_rho"] = pd.to_numeric(
        frame["test_kstar_proxy_spearman_adjusted_rho"], errors="coerce"
    )
    frame["test_kstar_std"] = pd.to_numeric(frame.get("test_kstar_std"), errors="coerce")
    return frame.loc[frame["test_kstar_proxy_spearman_adjusted_rho"].notna()].sort_values(
        ["test_kstar_proxy_spearman_adjusted_rho", "test_kstar_std"], ascending=[False, False]
    )


def _figure_list(figure_paths: dict[str, str]) -> str:
    if not figure_paths:
        return "No figures were built."
    return "\n".join(f"- `{name}`: `{path}`" for name, path in sorted(figure_paths.items()))


def build_report(matrix_dir: str | Path, report_dir: str | Path | None = None, build_figures: bool = True) -> Path:
    matrix_path = Path(matrix_dir).resolve()
    report_path = infer_report_dir(matrix_path) if report_dir is None else Path(report_dir).resolve()
    report_path.mkdir(parents=True, exist_ok=True)
    aggregate_paths = aggregate_matrix(matrix_path, report_path)
    all_runs = _read_csv(aggregate_paths["all_runs_csv"])
    variant_summary = _read_csv(aggregate_paths["variant_summary_csv"])
    failed_runs = _read_csv(aggregate_paths["failed_runs_csv"])
    manifest = _load_manifest(matrix_path)

    figure_paths: dict[str, str] = {}
    if build_figures and not all_runs.empty:
        figure_paths = build_matrix_visualizations(aggregate_paths["all_runs_csv"], report_path / "figures")

    best_predictions = _best_prediction_rows(all_runs)
    mechanism_candidates = _mechanism_rows(all_runs)
    completed_runs = int(len(all_runs))
    planned_tasks = int(manifest.get("task_count", completed_runs))
    failed_count = int(len(failed_runs))
    seed_count = int(all_runs["seed"].nunique()) if "seed" in all_runs.columns and not all_runs.empty else 0
    variant_count = int(all_runs["variant_id"].nunique()) if "variant_id" in all_runs.columns and not all_runs.empty else 0

    lines = [
        "# Informal RQ Improvement Matrix Report",
        "",
        "## Run Status",
        "",
        f"- Matrix directory: `{matrix_path}`",
        f"- Planned tasks: `{planned_tasks}`",
        f"- Completed summaries: `{completed_runs}`",
        f"- Failed tasks: `{failed_count}`",
        f"- Variants with completed runs: `{variant_count}`",
        f"- Distinct seeds observed: `{seed_count}`",
        "",
        "## Best Prediction Rows",
        "",
        _markdown_table(
            best_predictions.assign(test_mse=best_predictions.get("test_mse", pd.Series(dtype=float)).map(_fmt)),
            ["track", "variant_id", "run_group", "seed", "test_mse", "test_mae", "feature_bundle", "max_lag"],
            limit=12,
        ),
        "",
        "## Mechanism-Oriented CMDL Rows",
        "",
        _markdown_table(
            mechanism_candidates.assign(
                test_kstar_proxy_spearman_adjusted_rho=mechanism_candidates.get(
                    "test_kstar_proxy_spearman_adjusted_rho", pd.Series(dtype=float)
                ).map(_fmt),
                test_kstar_std=mechanism_candidates.get("test_kstar_std", pd.Series(dtype=float)).map(_fmt),
                test_omega_entropy_mean=mechanism_candidates.get("test_omega_entropy_mean", pd.Series(dtype=float)).map(_fmt),
            ),
            [
                "track",
                "variant_id",
                "seed",
                "proxy_perturbation",
                "test_kstar_proxy_spearman_adjusted_rho",
                "test_kstar_std",
                "test_omega_entropy_mean",
            ],
            limit=12,
        ),
        "",
        "## Variant Summary Files",
        "",
        f"- `all_runs.csv`: `{aggregate_paths['all_runs_csv']}`",
        f"- `variant_summary.csv`: `{aggregate_paths['variant_summary_csv']}`",
        f"- `track_summary.csv`: `{aggregate_paths['track_summary_csv']}`",
        f"- `mechanism_summary.csv`: `{aggregate_paths['mechanism_summary_csv']}`",
        f"- `failed_runs.csv`: `{aggregate_paths['failed_runs_csv']}`",
        "",
        "## Built Figures",
        "",
        _figure_list(figure_paths),
        "",
        "## Interpretation Guardrails",
        "",
        "- Prediction wins and mechanism evidence are ranked separately.",
        "- Seed-mean Omega and seed-std Omega are preferred over best-seed snapshots.",
        "- Full-span income/RPCYD evidence is not full-span RYDGDP multiseq evidence.",
        "- Structural 2006-2018 RYDGDP gaps are not imputed in the main analysis.",
        "- Falsification variants should weaken proxy-kstar alignment before mechanism claims are treated as credible.",
    ]
    if not variant_summary.empty:
        lines.extend(
            [
                "",
                "## Variant Summary Preview",
                "",
                _markdown_table(
                    variant_summary,
                    [
                        "track",
                        "variant_id",
                        "run_group",
                        "test_mse_mean",
                        "test_mse_std",
                        "test_kstar_std_mean",
                        "run_count",
                        "seed_count",
                    ],
                    limit=16,
                ),
            ]
        )
    report_file = report_path / "README.md"
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Informal RQ improvement matrix report.")
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = build_report(args.matrix_dir, args.report_dir, build_figures=not args.no_figures)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()