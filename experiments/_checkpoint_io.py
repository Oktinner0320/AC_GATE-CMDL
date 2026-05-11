"""Robust checkpoint writes for Windows-heavy experiment runs."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch


def save_torch_checkpoint(payload: dict[str, Any], checkpoint_path: str | Path, retries: int = 5) -> None:
    """Write a torch checkpoint via a unique temporary file, then replace the target."""

    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        for attempt in range(retries):
            try:
                temporary.replace(target)
                return
            except PermissionError:
                if attempt >= retries - 1:
                    raise
                time.sleep(0.2 * (attempt + 1))
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


__all__ = ["save_torch_checkpoint"]
