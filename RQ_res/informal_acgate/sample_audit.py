"""Audit Informal panel coverage and effective supervised sample counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PACKAGE_ROOT = Path(__file__).resolve().parent
RQ_RES_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = RQ_RES_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(RQ_RES_ROOT) not in sys.path:
    sys.path.insert(0, str(RQ_RES_ROOT))

from informal_acgate.loader import DEFAULT_INPUT_DIR, MULTISEQ_COLUMNS, TARGET_COLUMN, LEGACY_SEQUENCE_COLUMN


DEFAULT_AUDIT_DIR = RQ_RES_ROOT / "outputs" / "informal_acgate" / "sample_expansion_audit"
PANEL_FILES = {
    "single_fullspan": "informal_acgate_single_feature_ready.csv",
    "multiseq_fullspan": "informal_acgate_multiseq_ready.csv",
    "multiseq_overlap": "informal_acgate_multiseq_overlap_ready.csv",
}
SCENARIO_SPLITS = [
    {
        "scenario": "overlap_region_proxy_maxlag2",
        "panel": "multiseq_overlap",
        "year_start": 2019,
        "year_end": 2023,
        "train_end_year": 2021,
        "val_end_year": 2022,
        "max_lag": 2,
    },
    {
        "scenario": "fullspan_income_region_proxy_maxlag3",
        "panel": "single_fullspan",
        "year_start": 2006,
        "year_end": 2023,
        "train_end_year": 2018,
        "val_end_year": 2020,
        "max_lag": 3,
    },
    {
        "scenario": "fullspan_income_region_proxy_maxlag2",
        "panel": "single_fullspan",
        "year_start": 2006,
        "year_end": 2023,
        "train_end_year": 2018,
        "val_end_year": 2020,
        "max_lag": 2,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Informal sample expansion options.")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_AUDIT_DIR))
    return parser.parse_args()


def _read_panel(input_dir: Path, panel: str) -> pd.DataFrame:
    path = input_dir / PANEL_FILES[panel]
    if not path.exists():
        raise FileNotFoundError(f"Missing Informal panel CSV: {path}")
    frame = pd.read_csv(path)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    return frame


def _split_years(year_start: int, year_end: int, train_end_year: int, val_end_year: int, max_lag: int) -> dict[str, list[int]]:
    return {
        "train": list(range(year_start, train_end_year + 1)),
        "val": list(range(train_end_year + 1 - max_lag, val_end_year + 1)),
        "test": list(range(val_end_year + 1 - max_lag, year_end + 1)),
    }


def _panel_summary(panel: str, frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "panel": panel,
        "row_count": int(len(frame)),
        "entity_count": int(frame["entity_code"].nunique()),
        "year_start": int(frame["year"].min()),
        "year_end": int(frame["year"].max()),
        "year_count": int(frame["year"].nunique()),
        "column_count": int(len(frame.columns)),
    }


def _coverage_rows(panel: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    candidate_columns = [TARGET_COLUMN, LEGACY_SEQUENCE_COLUMN, *MULTISEQ_COLUMNS]
    candidate_columns = [column for column in dict.fromkeys(candidate_columns) if column in frame.columns]
    rows: list[dict[str, Any]] = []
    for year, year_frame in frame.groupby("year", sort=True):
        entity_count = int(year_frame["entity_code"].nunique())
        for column_name in candidate_columns:
            observed = int(year_frame[column_name].notna().sum())
            rows.append(
                {
                    "panel": panel,
                    "year": int(year),
                    "feature": column_name,
                    "entity_count": entity_count,
                    "observed_count": observed,
                    "missing_count": int(entity_count - observed),
                    "support_rate": float(observed / entity_count) if entity_count else 0.0,
                }
            )
    return rows


def _effective_sample_rows(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SCENARIO_SPLITS:
        frame = frames[str(spec["panel"])]
        n_entities = int(frame["entity_code"].nunique())
        split_years = _split_years(
            year_start=int(spec["year_start"]),
            year_end=int(spec["year_end"]),
            train_end_year=int(spec["train_end_year"]),
            val_end_year=int(spec["val_end_year"]),
            max_lag=int(spec["max_lag"]),
        )
        for split_name, years in split_years.items():
            prediction_years = years[int(spec["max_lag"]) :]
            rows.append(
                {
                    **spec,
                    "split": split_name,
                    "split_years": ",".join(str(year) for year in years),
                    "prediction_years": ",".join(str(year) for year in prediction_years),
                    "entity_count": n_entities,
                    "split_year_count": len(years),
                    "prediction_year_count": len(prediction_years),
                    "effective_samples": int(n_entities * len(prediction_years)),
                }
            )
    return rows


def _plot_support_heatmap(coverage: pd.DataFrame, output_dir: Path) -> Path:
    subset = coverage.loc[coverage["panel"].isin(["single_fullspan", "multiseq_fullspan"])]
    subset = subset.copy()
    subset["feature_label"] = subset["panel"] + ": " + subset["feature"]
    heatmap_frame = subset.pivot_table(index="feature_label", columns="year", values="support_rate", aggfunc="mean")
    figure, axis = plt.subplots(figsize=(13, max(4.5, 0.34 * len(heatmap_frame))), constrained_layout=True)
    sns.heatmap(heatmap_frame, cmap="YlGnBu", vmin=0.0, vmax=1.0, cbar_kws={"label": "support rate"}, ax=axis)
    axis.set_title("Informal Feature-Year Support")
    axis.set_xlabel("Year")
    axis.set_ylabel("Panel and feature")
    output_path = output_dir / "feature_year_support_heatmap.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _plot_effective_samples(effective_samples: pd.DataFrame, output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    sns.barplot(data=effective_samples, x="scenario", y="effective_samples", hue="split", ax=axis)
    axis.set_title("Effective Supervised Samples After Lag Warm-Up")
    axis.set_xlabel("")
    axis.set_ylabel("Entity-year prediction count")
    axis.tick_params(axis="x", rotation=25)
    output_path = output_dir / "effective_sample_counts.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def build_sample_expansion_audit(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_AUDIT_DIR,
) -> dict[str, str]:
    sns.set_theme(style="whitegrid")
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    frames = {panel: _read_panel(input_root, panel) for panel in PANEL_FILES}
    panel_summary = pd.DataFrame([_panel_summary(panel, frame) for panel, frame in frames.items()])
    coverage = pd.DataFrame([row for panel, frame in frames.items() for row in _coverage_rows(panel, frame)])
    effective_samples = pd.DataFrame(_effective_sample_rows(frames))

    panel_summary_path = output_root / "panel_summary.csv"
    coverage_path = output_root / "feature_year_support.csv"
    effective_samples_path = output_root / "effective_sample_counts.csv"
    panel_summary.to_csv(panel_summary_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    effective_samples.to_csv(effective_samples_path, index=False)

    paths = {
        "panel_summary_csv": str(panel_summary_path),
        "feature_year_support_csv": str(coverage_path),
        "effective_sample_counts_csv": str(effective_samples_path),
        "feature_year_support_heatmap": str(_plot_support_heatmap(coverage, output_root)),
        "effective_sample_counts": str(_plot_effective_samples(effective_samples, output_root)),
    }
    with (output_root / "sample_expansion_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(paths, handle, indent=2, ensure_ascii=True)
    return paths


def main() -> None:
    args = parse_args()
    paths = build_sample_expansion_audit(input_dir=args.input_dir, output_dir=args.output_dir)
    print(f"Built {len(paths)} sample-expansion audit artifacts.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()