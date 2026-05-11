"""Proxy perturbation helpers for Informal RQ falsification checks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from informal_acgate.loader import InformalPanel


PROXY_PERTURBATION_CHOICES = ("none", "shuffle", "noise")


def _zscore_columns(values: torch.Tensor) -> torch.Tensor:
    means = values.mean(dim=0, keepdim=True)
    stds = values.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    return (values - means) / stds


def _metadata_with_perturbation(metadata: dict[str, Any], mode: str, seed: int | None, details: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(metadata)
    payload = {
        "mode": mode,
        "seed": None if seed is None else int(seed),
        "uses_target_column": False,
        **details,
    }
    output["proxy_perturbation"] = payload
    audit = dict(output.get("audit", {}))
    audit["proxy_perturbation"] = payload
    output["audit"] = audit
    return output


def apply_proxy_perturbation(panel: InformalPanel, mode: str = "none", seed: int | None = None) -> InformalPanel:
    """Return a copy of one panel with falsified proxy rows when requested."""

    normalized = str(mode or "none").strip().lower()
    if normalized not in PROXY_PERTURBATION_CHOICES:
        raise ValueError(f"Unsupported proxy perturbation: {mode}. Expected one of {PROXY_PERTURBATION_CHOICES}")
    if normalized == "none":
        return panel

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(0 if seed is None else seed))
    proxy_cpu = panel.p_i.detach().cpu().clone()
    details: dict[str, Any]
    if normalized == "shuffle":
        permutation = torch.randperm(proxy_cpu.shape[0], generator=generator)
        perturbed_proxy = proxy_cpu[permutation]
        details = {"description": "Entity proxy rows were randomly permuted.", "permutation": permutation.tolist()}
    else:
        noise = torch.randn(proxy_cpu.shape, generator=generator, dtype=proxy_cpu.dtype)
        perturbed_proxy = _zscore_columns(noise)
        details = {"description": "Entity proxies were replaced with standardized Gaussian noise."}

    return InformalPanel(
        X_it=panel.X_it.clone(),
        p_i=perturbed_proxy.to(device=panel.p_i.device, dtype=panel.p_i.dtype),
        s_i=panel.s_i.clone(),
        Y_it=panel.Y_it.clone(),
        entity_ids=panel.entity_ids.clone(),
        time_index=panel.time_index.clone(),
        entity_codes=list(panel.entity_codes),
        entity_names=list(panel.entity_names),
        metadata=_metadata_with_perturbation(panel.metadata, normalized, seed, details),
    )


__all__ = ["PROXY_PERTURBATION_CHOICES", "apply_proxy_perturbation"]