from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from asrr_core import ExecutableStepResidualAdapter, apply_residual


class Pi05ExecutableASRRWrapper(nn.Module):
    """Generic pi0.5-style executable-step ASRR wrapper.

    The base policy can either implement `predict_action_chunk(batch, **kwargs)`
    or be directly callable.  pi0.5-specific preprocessing and action
    unnormalization should stay in the method runner.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        refiner: ExecutableStepResidualAdapter,
        *,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.base_policy = base_policy
        self.refiner = refiner
        self.alpha = float(alpha)
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.base_policy.eval()
        self.refiner.eval()

    @torch.no_grad()
    def predict_base_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        if hasattr(self.base_policy, "predict_action_chunk"):
            return self.base_policy.predict_action_chunk(batch, **kwargs)
        return self.base_policy(batch, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, torch.Tensor],
        *,
        obs_context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        base_action = self.predict_base_action(batch, **kwargs)
        if self.refiner.uses_obs and obs_context is None:
            raise ValueError("obs_context must be provided when the pi0.5 ASRR refiner uses obs context")
        if obs_context is not None:
            obs_context = obs_context.to(device=base_action.device, dtype=base_action.dtype)
        delta, info = self.refiner(base_action, obs_context=obs_context, return_info=True)
        refined = apply_residual(base_action, delta, alpha=self.alpha)
        return {
            "action": refined,
            "base_action": base_action,
            "delta_action": delta,
            "asrr_info": info,
        }
