from __future__ import annotations

from collections import deque
from typing import Optional

import torch
import torch.nn as nn

from asrr_core import ExecutableStepResidualAdapter, apply_residual


def flatten_obs_context(
    batch: dict[str, torch.Tensor],
    *,
    state_key: str = "observation.state",
    state_prefix: str = "observation.state.",
) -> torch.Tensor:
    """Extract a low-dimensional state vector from a generic VLA batch."""

    if state_key in batch and isinstance(batch[state_key], torch.Tensor):
        state = batch[state_key]
        return state.reshape(state.shape[0], -1)

    parts = []
    for key in sorted(batch):
        value = batch[key]
        if key.startswith(state_prefix) and isinstance(value, torch.Tensor):
            parts.append(value.reshape(value.shape[0], -1))
    if parts:
        return torch.cat(parts, dim=-1)

    tensor = next(value for value in batch.values() if isinstance(value, torch.Tensor))
    return torch.empty(tensor.shape[0], 0, device=tensor.device)


class SmolVLAExecutableASRRWrapper(nn.Module):
    """Generic SmolVLA-style executable-step ASRR wrapper.

    The base policy must implement `predict_action_chunk(batch, **kwargs)`.
    """

    def __init__(
        self,
        base_policy: nn.Module,
        refiner: ExecutableStepResidualAdapter,
        *,
        alpha: float = 1.0,
        n_action_steps: int = 1,
        action_key: str = "action",
    ):
        super().__init__()
        self.base_policy = base_policy
        self.refiner = refiner
        self.alpha = float(alpha)
        self.n_action_steps = int(n_action_steps)
        self.action_key = str(action_key)
        self._queue = deque(maxlen=max(1, self.n_action_steps))
        self.chunk_metrics: list[dict[str, float]] = []

        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.base_policy.eval()
        self.refiner.eval()

    def reset(self) -> None:
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()
        self._queue.clear()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        base_action = self.base_policy.predict_action_chunk(batch, **kwargs)
        obs_context: Optional[torch.Tensor] = None
        if self.refiner.uses_obs:
            obs_context = flatten_obs_context(batch).to(device=base_action.device, dtype=base_action.dtype)
        delta, info = self.refiner(base_action, obs_context=obs_context, return_info=True)
        refined = apply_residual(base_action, delta, alpha=self.alpha)
        self.chunk_metrics.append(
            {
                "gate_mean": float(info["gate_mean"].detach().cpu()),
                "delta_abs_mean": float(info["delta_abs_mean"].detach().cpu()),
                "delta_first_abs_mean": float(info["delta_first_abs_mean"].detach().cpu()),
                "base_abs_mean": float(base_action.abs().mean().detach().cpu()),
                "refined_abs_mean": float(refined.abs().mean().detach().cpu()),
            }
        )
        return refined

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        if not self._queue:
            refined = self.predict_action_chunk(batch, **kwargs)
            self._queue.extend(refined.transpose(0, 1)[: self.n_action_steps])
        return self._queue.popleft()

    def chunk_metric_summary(self) -> dict[str, float]:
        if not self.chunk_metrics:
            return {}
        keys = self.chunk_metrics[0]
        return {
            f"asrr_{key}": float(sum(item[key] for item in self.chunk_metrics) / len(self.chunk_metrics))
            for key in keys
        } | {"asrr_num_chunks": float(len(self.chunk_metrics))}
