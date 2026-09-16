from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from asrr_core import ActionSequenceResidualAdapter, apply_residual


def build_pi05_refiner(
    *,
    action_dim: int,
    horizon: int,
    global_context_dim: int = 0,
    step_context_dim: int = 0,
    num_task_embeddings: int = 0,
    editable_action_dims: Optional[int] = None,
    hidden_dim: int = 384,
    num_layers: int = 2,
    num_heads: int = 4,
    max_delta: Union[float, Sequence[float]] = 0.05,
) -> ActionSequenceResidualAdapter:
    """Build a sequence refiner for a frozen pi0.5-style action proposal."""

    if global_context_dim > 0 and step_context_dim > 0:
        fusion_mode = "state_action_context_add"
    elif global_context_dim > 0:
        fusion_mode = "state_add"
    elif step_context_dim > 0:
        fusion_mode = "action_context_add"
    else:
        fusion_mode = "action_only"

    if editable_action_dims is None:
        action_mask = [1.0] * action_dim
    else:
        if not 0 < editable_action_dims <= action_dim:
            raise ValueError("editable_action_dims must lie in [1, action_dim]")
        action_mask = [1.0] * editable_action_dims + [0.0] * (action_dim - editable_action_dims)

    return ActionSequenceResidualAdapter(
        action_dim=action_dim,
        horizon=horizon,
        state_context_dim=global_context_dim,
        action_context_dim=step_context_dim,
        num_task_embeddings=num_task_embeddings,
        hidden_dim=hidden_dim,
        fusion_mode=fusion_mode,
        encoder_type="transformer",
        num_layers=num_layers,
        num_heads=num_heads,
        head_type="bounded_dense",
        max_delta=max_delta,
        action_mask=action_mask,
    )


def _as_batch(value: Any, *, device: torch.device, ndim: int) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device)
    if tensor.ndim == ndim - 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != ndim:
        raise ValueError(f"expected ndim={ndim}, got shape={tuple(tensor.shape)}")
    return tensor


class Pi05ASRRWrapper(nn.Module):
    """Apply ASRR to tensors exposed by an external pi0.5 runner.

    ``predict_base_and_context`` must return ``base_action`` and may additionally
    return ``global_context``, ``step_context``, and ``task_index``. Image,
    language, normalization, and environment logic remain in the base runner.
    """

    def __init__(
        self,
        predict_base_and_context: Callable[..., Dict[str, Any]],
        refiner: ActionSequenceResidualAdapter,
        *,
        alpha: float = 1.0,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        self.predict_base_and_context = predict_base_and_context
        self.refiner = refiner.to(device)
        self.alpha = float(alpha)
        self.device = torch.device(device)
        self.refiner.eval()

    @torch.no_grad()
    def predict_action_chunk(self, *args, **kwargs) -> Dict[str, Any]:
        prediction = self.predict_base_and_context(*args, **kwargs)
        base = _as_batch(prediction["base_action"], device=self.device, ndim=3).float()

        global_context = None
        if self.refiner.uses_state:
            global_context = _as_batch(
                prediction["global_context"], device=self.device, ndim=2
            ).to(dtype=base.dtype)

        step_context = None
        if self.refiner.uses_action_context:
            step_context = _as_batch(
                prediction["step_context"], device=self.device, ndim=3
            ).to(dtype=base.dtype)

        task_index = None
        if self.refiner.num_task_embeddings > 0:
            task_index = torch.as_tensor(
                prediction["task_index"], device=self.device, dtype=torch.long
            ).reshape(-1)
            if task_index.shape[0] == 1 and base.shape[0] > 1:
                task_index = task_index.expand(base.shape[0])

        delta, info = self.refiner(
            base,
            state_context=global_context,
            action_context=step_context,
            task_index=task_index,
            return_info=True,
        )
        refined = apply_residual(base, delta, alpha=self.alpha)
        return {
            "action": refined,
            "base_action": base,
            "delta_action": delta,
            "asrr_info": info,
            "raw_prediction": prediction,
        }
