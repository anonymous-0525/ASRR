from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from asrr_core import ActionSequenceResidualAdapter, apply_residual, supervised_asrr_loss


def build_act_asrr_refiner(
    *,
    action_dim: int,
    horizon: int,
    variant: str = "qpos_add",
    qpos_dim: Optional[int] = None,
    hidden_dim: int = 256,
    encoder_type: str = "mlp",
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.0,
    head_type: str = "dense",
    max_delta: Optional[float] = None,
    freeze_last_action_dim: bool = False,
) -> ActionSequenceResidualAdapter:
    """Build the ASRR core used by an ACT-style policy wrapper."""

    fusion_map = {
        "action_only": "action_only",
        "qpos_add": "state_add",
        "qpos_concat": "state_concat",
        "film": "film",
    }
    if variant not in fusion_map:
        raise ValueError(f"Unsupported ACT ASRR variant={variant}; valid={sorted(fusion_map)}")
    uses_qpos = variant != "action_only"
    if uses_qpos and (qpos_dim is None or qpos_dim <= 0):
        raise ValueError("qpos_dim must be provided for qpos-conditioned variants")

    return ActionSequenceResidualAdapter(
        action_dim=action_dim,
        horizon=horizon,
        state_context_dim=int(qpos_dim or 0),
        hidden_dim=hidden_dim,
        fusion_mode=fusion_map[variant],
        encoder_type=encoder_type,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        head_type=head_type,
        max_delta=max_delta,
        freeze_last_action_dim=freeze_last_action_dim,
    )


class ASRRACTWrapper(nn.Module):
    """Wrap a frozen ACT-like policy with an ASRR action sequence refiner.

    The base policy is expected to return action chunks with shape `[B,H,D]`.
    This wrapper is intentionally independent of any specific ACT repository.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        refiner: ActionSequenceResidualAdapter,
        *,
        alpha: float = 1.0,
        train_loss_type: str = "l1",
        delta_l2_weight: float = 0.01,
        smooth_l2_weight: float = 0.0,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.refiner = refiner
        self.alpha = float(alpha)
        self.train_loss_type = str(train_loss_type)
        self.delta_l2_weight = float(delta_l2_weight)
        self.smooth_l2_weight = float(smooth_l2_weight)

        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.base_policy.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_policy.eval()
        return self

    @torch.no_grad()
    def predict_base_action(self, *base_args, **base_kwargs) -> torch.Tensor:
        return self.base_policy(*base_args, **base_kwargs)

    def refine(
        self,
        base_action: torch.Tensor,
        *,
        qpos: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        state_context = qpos if self.refiner.uses_state else None
        delta, info = self.refiner(base_action, state_context=state_context, return_info=True)
        refined = apply_residual(base_action, delta, alpha=self.alpha if alpha is None else alpha)
        if return_info:
            info = dict(info)
            info["delta_action"] = delta
            info["base_action"] = base_action
            return refined, info
        return refined

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        *,
        target_action: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        base_kwargs: Optional[dict] = None,
    ):
        base_kwargs = base_kwargs or {}
        with torch.no_grad():
            base_action = self.base_policy(qpos, image, **base_kwargs)
        state_context = qpos if self.refiner.uses_state else None
        delta, info = self.refiner(base_action, state_context=state_context, return_info=True)
        refined = apply_residual(base_action, delta, alpha=self.alpha)

        if target_action is None:
            return refined

        loss_info = supervised_asrr_loss(
            base_action=base_action,
            delta_action=delta,
            target_action=target_action,
            is_pad=is_pad,
            alpha=self.alpha,
            loss_type=self.train_loss_type,
            delta_l2_weight=self.delta_l2_weight,
            smooth_l2_weight=self.smooth_l2_weight,
        )
        loss_info["gate_mean"] = info["gate_mean"]
        loss_info["delta_abs_mean"] = info["delta_abs_mean"]
        return loss_info
