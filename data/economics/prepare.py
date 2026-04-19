"""Economics-domain cleaned-table export utility.

该脚本把 PWT 原始表清洗为可直接检查的 long-form CSV，
但不在这里做训练窗口相关的标准化与 proxy/static 聚合。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
	sys.path.insert(0, str(WORKSPACE_ROOT))

from data.economics.economics_loader import (
	DEFAULT_YEAR_END,
	DEFAULT_YEAR_START,
	build_cleaned_economics_dataframe,
)


DEFAULT_ECONOMICS_CLEANED_OUTPUT_PATH = Path(__file__).resolve().parent / "processed" / "economics_cleaned_long.csv"


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for exporting the cleaned economics table.

	解析 economics 清洗表导出的命令行参数。
	"""

	parser = argparse.ArgumentParser(description="Export the cleaned economics long-form table.")
	parser.add_argument(
		"--csv-path",
		type=str,
		default=None,
		help="Optional local PWT cache path. Defaults to data/economics/raw/pwt110.csv.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=str(DEFAULT_ECONOMICS_CLEANED_OUTPUT_PATH),
		help="Output CSV path for the cleaned economics table.",
	)
	parser.add_argument("--target-column", type=str, default="ctfp")
	parser.add_argument("--year-start", type=int, default=DEFAULT_YEAR_START)
	parser.add_argument("--year-end", type=int, default=DEFAULT_YEAR_END)
	parser.add_argument("--max-missing-share", type=float, default=0.15)
	return parser.parse_args()


def save_cleaned_economics_table(
	csv_path: str | Path | None = None,
	output_path: str | Path = DEFAULT_ECONOMICS_CLEANED_OUTPUT_PATH,
	target_column: str = "ctfp",
	year_start: int = DEFAULT_YEAR_START,
	year_end: int = DEFAULT_YEAR_END,
	max_missing_share: float = 0.15,
) -> Path:
	"""Build and save the cleaned economics long-form table.

	生成并落盘 economics 清洗长表。
	"""

	dataframe = build_cleaned_economics_dataframe(
		csv_path=csv_path,
		target_column=target_column,
		year_start=year_start,
		year_end=year_end,
		max_missing_share=max_missing_share,
	)

	destination = Path(output_path).resolve()
	destination.parent.mkdir(parents=True, exist_ok=True)
	dataframe.to_csv(destination, index=False)
	return destination


def main() -> None:
	"""Execute the cleaned-table export and print a compact summary.

	执行 economics 清洗表导出，并打印简要摘要。
	"""

	args = parse_args()
	destination = save_cleaned_economics_table(
		csv_path=args.csv_path,
		output_path=args.output,
		target_column=args.target_column,
		year_start=args.year_start,
		year_end=args.year_end,
		max_missing_share=args.max_missing_share,
	)
	dataframe = build_cleaned_economics_dataframe(
		csv_path=args.csv_path,
		target_column=args.target_column,
		year_start=args.year_start,
		year_end=args.year_end,
		max_missing_share=args.max_missing_share,
	)

	entity_count = int(dataframe["entity_code"].nunique())
	year_min = int(dataframe["year"].min())
	year_max = int(dataframe["year"].max())
	interpolated_share = float(dataframe["row_was_missing"].mean())

	print(f"Saved cleaned economics table to: {destination}")
	print(f"Rows: {len(dataframe)}, Entities: {entity_count}, Years: {year_min}..{year_max}")
	print(f"Interpolated row share: {interpolated_share:.4f}")


if __name__ == "__main__":
	main()


__all__ = [
	"DEFAULT_ECONOMICS_CLEANED_OUTPUT_PATH",
	"save_cleaned_economics_table",
]