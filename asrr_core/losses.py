from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def supervised_asrr_loss(
    *,
    base_action: torch.Tensor,
    delta_action: torch.Tensor,
    target_action: torch.Tensor,
    is_pad: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    loss_type: str = "l1",
    delta_l2_weight: float = 0.01,
    smooth_l2_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Supervised residual loss shared by ACT/DP/VQ-BeT/VLA wrappers."""

    if base_action.ndim != 3 or delta_action.ndim != 3 or target_action.ndim != 3:
        raise ValueError("base_action, delta_action, and target_action must be [B,H,D]")
    horizon = min(base_action.shape[1], delta_action.shape[1], target_action.shape[1])
    base_short = base_action[:, :horizon]
    delta_short = delta_action[:, :horizon]
    target_short = target_action[:, :horizon]
    refined_short = base_short + float(alpha) * delta_short

    if is_pad is None:
        valid = torch.ones_like(target_short, dtype=torch.bool)
    else:
        valid = ~is_pad[:, :horizon].unsqueeze(-1)

    if loss_type == "l1":
        refined_err = F.l1_loss(refined_short, target_short, reduction="none")
        base_err = F.l1_loss(base_short, target_short, reduction="none")
    elif loss_type == "mse":
        refined_err = F.mse_loss(refined_short, target_short, reduction="none")
        base_err = F.mse_loss(base_short, target_short, reduction="none")
    elif loss_type == "huber":
        refined_err = F.smooth_l1_loss(refined_short, target_short, reduction="none")
        base_err = F.smooth_l1_loss(base_short, target_short, reduction="none")
    else:
        raise ValueError("loss_type must be one of: l1, mse, huber")

    refined_loss = (refined_err * valid).mean()
    base_loss = (base_err * valid).mean()
    delta_l2 = (delta_short.pow(2) * valid).mean()
    if horizon > 1:
        delta_smooth = (delta_short[:, 1:] - delta_short[:, :-1]).pow(2).mean()
    else:
        delta_smooth = torch.zeros((), dtype=delta_short.dtype, device=delta_short.device)

    loss = refined_loss + float(delta_l2_weight) * delta_l2 + float(smooth_l2_weight) * delta_smooth
    return {
        "loss": loss,
        "refined_loss": refined_loss,
        "base_loss": base_loss,
        "delta_l2": delta_l2,
        "delta_smooth": delta_smooth,
    }
