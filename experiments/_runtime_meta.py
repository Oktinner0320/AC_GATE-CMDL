"""Runtime metadata helpers for experiment summaries."""

from __future__ import annotations

import platform
import time
from typing import Any

import torch


def start_runtime_timer() -> float:
    """Return a monotonic timestamp for wall-time accounting."""

    return time.perf_counter()


def build_runtime_metadata(device: torch.device | str | None = None, started_at: float | None = None) -> dict[str, Any]:
    """Collect hardware and software metadata for one experiment run."""

    resolved_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(resolved_device)
        except Exception:
            device_name = "cuda"
    else:
        device_name = platform.processor() or platform.machine() or "cpu"

    metadata: dict[str, Any] = {
        "device": str(resolved_device),
        "device_name": device_name,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    if started_at is not None:
        metadata["wall_time_seconds"] = float(time.perf_counter() - started_at)
    return metadata


def attach_runtime_metadata(
    summary: dict[str, Any],
    device: torch.device | str | None = None,
    started_at: float | None = None,
) -> dict[str, Any]:
    """Attach runtime metadata to a summary dictionary and return it."""

    summary["runtime"] = build_runtime_metadata(device=device, started_at=started_at)
    return summary


__all__ = ["attach_runtime_metadata", "build_runtime_metadata", "start_runtime_timer"]