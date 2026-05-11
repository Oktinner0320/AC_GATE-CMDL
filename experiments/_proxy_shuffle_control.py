"""Utilities for proxy-shuffle negative-control experiments.

The negative control keeps each panel's time series, targets, static features,
entity order, and split windows fixed while permuting only the entity-level
proxy matrix used by the AC encoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


NEGATIVE_CONTROL_NAME = "proxy_shuffle"
LOCKED_OUTPUT_MARKERS = ("complete_20seed_20260426",)


def ensure_safe_negative_control_output_root(output_root: str | Path) -> Path:
    """Return a resolved output root, rejecting existing locked result trees."""

    resolved = Path(output_root).resolve()
    normalized = str(resolved).replace("\\", "/").lower()
    if any(marker in normalized for marker in LOCKED_OUTPUT_MARKERS):
        raise ValueError(f"Refusing to write a negative control into locked outputs: {resolved}")
    if "/outputs/notebook_" in normalized:
        raise ValueError(f"Refusing to write a negative control into notebook result outputs: {resolved}")
    return resolved


def build_proxy_permutation(n_entities: int, proxy_perm_seed: int) -> np.ndarray:
    """Build a deterministic non-identity row permutation for entity proxies."""

    if n_entities < 2:
        raise ValueError("Proxy permutation requires at least two entities")
    rng = np.random.default_rng(int(proxy_perm_seed))
    order = rng.permutation(n_entities).astype(np.int64)
    identity = np.arange(n_entities, dtype=np.int64)
    if np.array_equal(order, identity):
        order = np.roll(identity, 1)
    return order


def validate_proxy_permutation(
    original_proxy: torch.Tensor,
    shuffled_proxy: torch.Tensor,
    order: np.ndarray,
) -> None:
    """Validate that proxy rows were permuted without changing distributions."""

    if original_proxy.shape != shuffled_proxy.shape:
        raise ValueError(f"Proxy shape changed: {tuple(original_proxy.shape)} vs {tuple(shuffled_proxy.shape)}")
    if original_proxy.ndim != 2:
        raise ValueError(f"Expected a 2D proxy matrix, got shape {tuple(original_proxy.shape)}")
    if len(order) != original_proxy.shape[0]:
        raise ValueError("Permutation length does not match the proxy row count")
    if np.array_equal(order, np.arange(len(order), dtype=np.int64)):
        raise ValueError("Proxy permutation is identity")

    order_tensor = torch.as_tensor(order, dtype=torch.long, device=original_proxy.device)
    expected = original_proxy.index_select(0, order_tensor)
    torch.testing.assert_close(shuffled_proxy, expected)

    original_sorted = torch.sort(original_proxy.detach().cpu(), dim=0).values
    shuffled_sorted = torch.sort(shuffled_proxy.detach().cpu(), dim=0).values
    torch.testing.assert_close(shuffled_sorted, original_sorted)


def apply_proxy_permutation_to_panel(
    panel: Any,
    order: np.ndarray,
    proxy_perm_seed: int,
    training_seed: int | None = None,
) -> Any:
    """Apply one entity-level proxy row permutation to a panel in place."""

    entity_codes = list(panel.entity_codes)
    if len(entity_codes) != len(order):
        raise ValueError("Panel entity_codes do not align with the permutation order")

    original_proxy = panel.p_i.detach().clone()
    order_tensor = torch.as_tensor(order, dtype=torch.long, device=panel.p_i.device)
    shuffled_proxy = panel.p_i.index_select(0, order_tensor).clone()
    validate_proxy_permutation(original_proxy, shuffled_proxy, order)

    panel.p_i = shuffled_proxy
    metadata = dict(panel.metadata)
    metadata.update(
        {
            "negative_control": NEGATIVE_CONTROL_NAME,
            "proxy_shuffle_applied": True,
            "training_seed": None if training_seed is None else int(training_seed),
            "proxy_perm_seed": int(proxy_perm_seed),
            "proxy_permutation_order": [int(value) for value in order.tolist()],
            "proxy_permutation_target_entity_codes": entity_codes,
            "proxy_permutation_source_entity_codes": [entity_codes[int(index)] for index in order.tolist()],
        }
    )
    panel.metadata = metadata
    return panel


def apply_proxy_shuffle_to_setup(
    setup: Any,
    training_seed: int,
    proxy_perm_seed: int,
) -> Any:
    """Apply a paired proxy shuffle to all panels in one experiment setup."""

    entity_codes = list(setup.full_panel.entity_codes)
    order = build_proxy_permutation(len(entity_codes), proxy_perm_seed)
    for panel_name in ("full_panel", "train_panel", "val_panel", "test_panel"):
        panel = getattr(setup, panel_name)
        if list(panel.entity_codes) != entity_codes:
            raise ValueError(f"{panel_name} entity order differs from full_panel")
        apply_proxy_permutation_to_panel(
            panel=panel,
            order=order,
            proxy_perm_seed=proxy_perm_seed,
            training_seed=training_seed,
        )
    return setup


def negative_control_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    """Extract JSON-safe negative-control metadata from panel metadata."""

    if metadata.get("negative_control") != NEGATIVE_CONTROL_NAME:
        return None
    keys = [
        "negative_control",
        "proxy_shuffle_applied",
        "training_seed",
        "proxy_perm_seed",
        "proxy_permutation_order",
        "proxy_permutation_target_entity_codes",
        "proxy_permutation_source_entity_codes",
    ]
    return {key: metadata.get(key) for key in keys if key in metadata}