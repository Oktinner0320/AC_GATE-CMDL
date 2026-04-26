"""Penn World Table download utility for the economics domain.

Step 5 的 economics 域下载脚本。默认使用官方最新版 PWT 11.0 的 Dataverse
Excel 直链，并在本地缓存为 CSV，便于下游 loader 继续使用稳定的文本格式。
同时保留对本地 CSV/Excel 或其他 URL 的覆盖能力，以避免上游地址变更时阻断
后续 loader 与实验入口的实现。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_PWT_VERSION = "11.0"
DEFAULT_PWT_SOURCE = "https://dataverse.nl/api/access/datafile/554105"
DEFAULT_PWT_SHEET_NAME = "Data"
DEFAULT_PWT_OUTPUT_PATH = Path(__file__).resolve().parent / "raw" / "pwt110.csv"


def _looks_like_excel_source(source: str) -> bool:
	"""Infer whether a URL or path should be read as Excel.

	根据 URL 或路径推断源文件是否应按 Excel 读取。
	"""

	normalized = source.lower()
	return normalized.endswith(".xlsx") or normalized.endswith(".xls") or "datafile/554105" in normalized


def _metadata_path(cache_path: Path) -> Path:
	return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _write_meta(cache_path: Path, source_url: str) -> Path:
	digest = hashlib.sha256(cache_path.read_bytes()).hexdigest()
	meta = {
		"dataset": "Penn World Table",
		"version": DEFAULT_PWT_VERSION,
		"source_url": str(source_url),
		"downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
		"sha256": digest,
		"bytes": cache_path.stat().st_size,
	}
	meta_path = _metadata_path(cache_path)
	meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
	return meta_path


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments for the economics data download step.

	解析 economics 数据下载所需的命令行参数。
	"""

	parser = argparse.ArgumentParser(description="Download or copy the Penn World Table table file.")
	parser.add_argument(
		"--source",
		type=str,
		default=DEFAULT_PWT_SOURCE,
		help="Table URL or local table path. Defaults to the official PWT 11.0 Excel file.",
	)
	parser.add_argument(
		"--sheet-name",
		type=str,
		default=DEFAULT_PWT_SHEET_NAME,
		help="Excel sheet name to read when the source is an .xls/.xlsx file.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=str(DEFAULT_PWT_OUTPUT_PATH),
		help="Destination path for the local PWT cache. The cache is always written as CSV.",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Overwrite the existing local CSV cache if it already exists.",
	)
	return parser.parse_args()


def load_pwt_source(source: str, sheet_name: str = DEFAULT_PWT_SHEET_NAME) -> pd.DataFrame:
	"""Load the PWT table from a URL or a local CSV/Excel file.

	从 URL 或本地 CSV/Excel 读取 PWT 表。
	"""

	try:
		if _looks_like_excel_source(source):
			return pd.read_excel(source, sheet_name=sheet_name)
		return pd.read_csv(source, encoding="utf-8")
	except Exception as error:
		raise RuntimeError(
			"Failed to read the Penn World Table source. "
			"The official source may have changed or the file format may not match the expected sheet. "
			"Pass --source with a working CSV/Excel URL or local path, and use --sheet-name when needed."
		) from error


def download_pwt_table(
	source: str = DEFAULT_PWT_SOURCE,
	output_path: str | Path = DEFAULT_PWT_OUTPUT_PATH,
	sheet_name: str = DEFAULT_PWT_SHEET_NAME,
	force: bool = False,
) -> Path:
	"""Download or copy the PWT table into the local raw-data cache.

	下载或复制 PWT 表到 economics 域的本地原始数据目录，并统一缓存为 CSV。
	"""

	destination = Path(output_path).resolve()
	if destination.exists() and not force:
		if not _metadata_path(destination).exists():
			_write_meta(destination, source)
		return destination

	dataframe = load_pwt_source(source, sheet_name=sheet_name)
	destination.parent.mkdir(parents=True, exist_ok=True)
	dataframe.to_csv(destination, index=False)
	_write_meta(destination, source)
	return destination


def main() -> None:
	"""Execute the economics raw-data download step.

	执行 economics 域的原始数据下载步骤。
	"""

	args = parse_args()
	local_path = download_pwt_table(
		source=args.source,
		output_path=args.output,
		sheet_name=args.sheet_name,
		force=args.force,
	)
	dataframe = pd.read_csv(local_path)
	print(f"Saved economics raw data to: {local_path}")
	print(f"PWT version source: {DEFAULT_PWT_VERSION}")
	print(f"Rows: {len(dataframe)}, Columns: {len(dataframe.columns)}")


if __name__ == "__main__":
	main()


__all__ = [
	"DEFAULT_PWT_OUTPUT_PATH",
	"DEFAULT_PWT_SOURCE",
	"DEFAULT_PWT_SHEET_NAME",
	"DEFAULT_PWT_VERSION",
	"download_pwt_table",
	"load_pwt_source",
]
