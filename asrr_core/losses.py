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
    """Supervised residual loss for cached action-sequence refinement."""

    if base_action.ndim != 3 or delta_action.ndim != 3 or target_action.ndim != 3:
        raise ValueError("base_action, delta_action, and target_action must be [B,H,D]")
    if base_action.shape[0] != delta_action.shape[0] or base_action.shape[0] != target_action.shape[0]:
        raise ValueError("base_action, delta_action, and target_action must share batch size")
    if base_action.shape[2] != delta_action.shape[2] or base_action.shape[2] != target_action.shape[2]:
        raise ValueError("base_action, delta_action, and target_action must share action_dim")
    horizon = min(base_action.shape[1], delta_action.shape[1], target_action.shape[1])
    base_short = base_action[:, :horizon]
    delta_short = delta_action[:, :horizon]
    target_short = target_action[:, :horizon]
    refined_short = base_short + float(alpha) * delta_short

    if is_pad is None:
        valid = torch.ones_like(target_short, dtype=torch.bool)
    else:
        if is_pad.ndim != 2 or is_pad.shape[0] != base_action.shape[0] or is_pad.shape[1] < horizon:
            raise ValueError("is_pad must have shape [B,H] and cover the supervised horizon")
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

    valid = valid.expand_as(target_short)
    normalizer = valid.sum().clamp_min(1)
    refined_loss = (refined_err * valid).sum() / normalizer
    base_loss = (base_err * valid).sum() / normalizer
    delta_l2 = (delta_short.pow(2) * valid).sum() / normalizer
    if horizon > 1:
        delta_diff = (delta_short[:, 1:] - delta_short[:, :-1]).pow(2)
        if is_pad is None:
            delta_smooth = delta_diff.mean()
        else:
            transition_valid = (~is_pad[:, 1:horizon] & ~is_pad[:, : horizon - 1]).unsqueeze(-1)
            transition_valid = transition_valid.expand_as(delta_diff)
            transition_count = transition_valid.sum()
            if int(transition_count) == 0:
                delta_smooth = torch.zeros((), dtype=delta_short.dtype, device=delta_short.device)
            else:
                delta_smooth = (delta_diff * transition_valid).sum() / transition_count
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
