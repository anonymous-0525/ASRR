from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from asrr_core import ActionSequenceResidualAdapter, apply_residual
from examples.openvla_oft.openvla_oft_refiner import OpenVLAOFTResidualAdapter


def clip_openvla_oft_normalized_actions(actions: torch.Tensor) -> torch.Tensor:
    """Clip OpenVLA-OFT normalized actions.

    Pose dimensions use [-1, 1].  The final gripper dimension follows the
    common OpenVLA-OFT normalized-action convention and is clipped to [0, 1].
    """

    out = actions.clone()
    if out.shape[-1] > 1:
        out[..., :-1] = out[..., :-1].clamp(-1.0, 1.0)
        out[..., -1] = out[..., -1].clamp(0.0, 1.0)
    else:
        out = out.clamp(-1.0, 1.0)
    return out


def build_openvla_oft_refiner(
    *,
    action_dim: int = 7,
    horizon: int = 8,
    action_context_dim: int = 4096,
    obs_context_dim: int = 8,
    num_task_embeddings: int = 10,
    hidden_dim: int = 384,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.0,
    head_type: str = "sigmoid_gate",
    context_mode: str = "mean_context",
    lowrank_num_basis: int = 4,
    gate_bias_init: float = -5.0,
    max_delta: Optional[list[float]] = None,
    freeze_last_action_dim: bool = True,
) -> OpenVLAOFTResidualAdapter:
    if max_delta is None:
        max_delta = [0.04, 0.04, 0.04, 0.015, 0.015, 0.015, 0.0]
    return OpenVLAOFTResidualAdapter(
        action_dim=action_dim,
        horizon=horizon,
        action_context_dim=action_context_dim,
        obs_context_dim=obs_context_dim,
        num_task_embeddings=num_task_embeddings,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_delta=max_delta,
        gate_bias_init=gate_bias_init,
        freeze_gripper=freeze_last_action_dim,
        head_type=head_type,
        lowrank_num_basis=lowrank_num_basis,
        context_mode=context_mode,
    )


def build_openvla_oft_core_refiner(
    *,
    action_dim: int = 7,
    horizon: int = 8,
    action_context_dim: int = 4096,
    obs_context_dim: int = 8,
    num_task_embeddings: int = 10,
    hidden_dim: int = 384,
    num_layers: int = 2,
    num_heads: int = 4,
    head_type: str = "sigmoid_gate",
    max_delta: Optional[list[float]] = None,
    freeze_last_action_dim: bool = True,
) -> ActionSequenceResidualAdapter:
    """Build the generic ASRR sequence refiner for non-checkpoint-compatible use."""

    if max_delta is None:
        max_delta = [0.04, 0.04, 0.04, 0.015, 0.015, 0.015, 0.0]
    if head_type == "bounded_delta_only":
        head_type = "bounded_dense"
    fusion_mode = "state_action_context_add" if obs_context_dim > 0 else "action_context_add"
    return ActionSequenceResidualAdapter(
        action_dim=action_dim,
        horizon=horizon,
        state_context_dim=obs_context_dim,
        action_context_dim=action_context_dim,
        num_task_embeddings=num_task_embeddings,
        hidden_dim=hidden_dim,
        fusion_mode=fusion_mode,
        encoder_type="transformer",
        num_layers=num_layers,
        num_heads=num_heads,
        head_type=head_type,
        max_delta=max_delta,
        freeze_last_action_dim=freeze_last_action_dim,
    )


def _to_batch_tensor(value: Any, *, device: torch.device, dtype: torch.dtype, ndim: int) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=dtype)
    if tensor.ndim == ndim - 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != ndim:
        raise ValueError(f"Expected tensor with ndim={ndim}, got shape={tuple(tensor.shape)}")
    return tensor


class OpenVLAOFTASRRWrapper(nn.Module):
    """Wrap an OpenVLA-OFT predictor that returns normalized action context."""

    def __init__(
        self,
        predict_base_and_context: Callable[..., dict[str, Any]],
        refiner: nn.Module,
        *,
        alpha: float = 1.0,
        device: str | torch.device = "cuda",
        unnormalize_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__()
        self.predict_base_and_context = predict_base_and_context
        self.refiner = refiner
        self.alpha = float(alpha)
        self.device = torch.device(device)
        self.unnormalize_fn = unnormalize_fn
        self.refiner.to(self.device)
        self.refiner.eval()

    @torch.no_grad()
    def predict_action_chunk(self, *args, **kwargs) -> dict[str, Any]:
        pred = self.predict_base_and_context(*args, **kwargs)
        base = _to_batch_tensor(
            pred["base_action_norm"],
            device=self.device,
            dtype=torch.float32,
            ndim=3,
        )
        raw_context = pred["action_context"]
        context_ndim = 3 if getattr(self.refiner, "context_mode", "mean_context") == "mean_context" else 4
        action_context = _to_batch_tensor(raw_context, device=self.device, dtype=torch.float32, ndim=context_ndim)
        obs_context = None
        uses_local_openvla_refiner = hasattr(self.refiner, "context_mode")
        uses_state = bool(getattr(self.refiner, "uses_state", False)) or bool(getattr(self.refiner, "obs_context_dim", 0) > 0)
        if uses_state:
            obs_context = _to_batch_tensor(
                pred["obs_context"],
                device=self.device,
                dtype=torch.float32,
                ndim=2,
            )
        task_index = None
        if getattr(self.refiner, "num_task_embeddings", 0) > 0:
            raw_task = pred.get("task_index", 0)
            task_index = torch.as_tensor(raw_task, device=self.device, dtype=torch.long).reshape(-1)
            if task_index.shape[0] == 1 and base.shape[0] > 1:
                task_index = task_index.expand(base.shape[0])

        if uses_local_openvla_refiner:
            delta, info = self.refiner(
                base,
                action_context=action_context,
                obs_context=obs_context,
                task_index=task_index,
                return_info=True,
            )
        else:
            delta, info = self.refiner(
                base,
                state_context=obs_context,
                action_context=action_context,
                task_index=task_index,
                return_info=True,
            )
        refined_norm = clip_openvla_oft_normalized_actions(apply_residual(base, delta, alpha=self.alpha))
        refined_np = refined_norm.squeeze(0).detach().cpu().numpy()
        action = self.unnormalize_fn(refined_np) if self.unnormalize_fn is not None else refined_np
        return {
            "action": action,
            "refined_action_norm": refined_np,
            "base_action_norm": base.squeeze(0).detach().cpu().numpy(),
            "delta_action_norm": delta.squeeze(0).detach().cpu().numpy(),
            "asrr_info": {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in info.items()},
            "raw_prediction": pred,
        }
