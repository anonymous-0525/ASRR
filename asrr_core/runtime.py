from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ASRROutput:
    """Standard output object for policy wrappers."""

    action: torch.Tensor
    base_action: torch.Tensor
    delta_action: torch.Tensor
    info: Dict[str, torch.Tensor]


def apply_residual(
    base_action: torch.Tensor,
    delta_action: torch.Tensor,
    *,
    alpha: float = 1.0,
    action_mask: Optional[torch.Tensor] = None,
    clamp: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Apply the ASRR residual contract in a policy-agnostic way."""

    if base_action.ndim != 3 or delta_action.ndim != 3:
        raise ValueError("base_action and delta_action must be [B,H,D]")
    if base_action.shape != delta_action.shape:
        raise ValueError(f"base_action and delta_action shape mismatch: {base_action.shape} vs {delta_action.shape}")
    delta = delta_action
    if action_mask is not None:
        if action_mask.ndim != 1 or action_mask.shape[0] != base_action.shape[-1]:
            raise ValueError("action_mask must be [D]")
        delta = delta * action_mask.to(device=delta.device, dtype=delta.dtype).view(1, 1, -1)
    refined = base_action + float(alpha) * delta
    if clamp is not None:
        refined = refined.clamp(float(clamp[0]), float(clamp[1]))
    return refined


def refine_with_adapter(
    adapter: nn.Module,
    base_action: torch.Tensor,
    *,
    alpha: float = 1.0,
    action_mask: Optional[torch.Tensor] = None,
    clamp: Optional[Tuple[float, float]] = None,
    **adapter_kwargs,
) -> ASRROutput:
    """Run an adapter and return the standardized ASRR output bundle."""

    delta, info = adapter(base_action, return_info=True, **adapter_kwargs)
    refined = apply_residual(
        base_action,
        delta,
        alpha=alpha,
        action_mask=action_mask,
        clamp=clamp,
    )
    return ASRROutput(action=refined, base_action=base_action, delta_action=delta, info=info)


def count_trainable_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))
