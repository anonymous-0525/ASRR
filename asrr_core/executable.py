from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn


class ExecutableStepResidualAdapter(nn.Module):
    """Bounded residual refiner for policies that execute only the first action.

    This is the policy-agnostic version of the Simple-ASRR structure used by
    SmolVLA-style and pi0.5-style receding-horizon policies:

        concat(base_action[:, :input_horizon], optional low-dimensional state)
          -> LayerNorm -> MLP -> tanh(delta) * sigmoid(gate) * max_delta

    The module returns a full action chunk with the same shape as ``base_action``,
    but only ``delta[:, 0]`` is non-zero.  Parameter names intentionally match the
    ``SimpleExecutableResidualAdapter`` naming convention so saved state_dict
    checkpoints can be reused across wrappers.
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        input_horizon: int = 5,
        obs_context_dim: int = 0,
        hidden_dim: int = 512,
        max_delta: Union[float, Sequence[float]] = 0.05,
        freeze_last_action_dim: bool = True,
        gate_bias_init: float = -2.0,
    ):
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if input_horizon <= 0:
            raise ValueError("input_horizon must be positive")
        if input_horizon > horizon:
            raise ValueError(f"input_horizon={input_horizon} cannot exceed horizon={horizon}")
        if obs_context_dim < 0:
            raise ValueError("obs_context_dim must be non-negative")

        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.input_horizon = int(input_horizon)
        self.obs_context_dim = int(obs_context_dim)
        self.hidden_dim = int(hidden_dim)
        self.gate_bias_init = float(gate_bias_init)

        input_dim = self.input_horizon * self.action_dim + self.obs_context_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.gate_head = nn.Linear(self.hidden_dim, self.action_dim)

        if isinstance(max_delta, (list, tuple)):
            if len(max_delta) != self.action_dim:
                raise ValueError(f"max_delta sequence must match action_dim={self.action_dim}")
            max_delta_tensor = torch.tensor([float(x) for x in max_delta], dtype=torch.float32)
        else:
            max_delta_tensor = torch.full((self.action_dim,), float(max_delta), dtype=torch.float32)
        self.register_buffer("max_delta", max_delta_tensor)

        action_mask = torch.ones(self.action_dim, dtype=torch.float32)
        if freeze_last_action_dim and self.action_dim > 0:
            action_mask[-1] = 0.0
        self.register_buffer("action_mask", action_mask)
        self.reset_parameters()

    @property
    def uses_obs(self) -> bool:
        return self.obs_context_dim > 0

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, self.gate_bias_init)

    def forward(
        self,
        base_action: torch.Tensor,
        obs_context: Union[torch.Tensor, None] = None,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if base_action.ndim != 3:
            raise ValueError(f"base_action must be [B,H,D], got {tuple(base_action.shape)}")
        batch_size, horizon, action_dim = base_action.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_dim}")
        if horizon < self.input_horizon:
            raise ValueError(f"Expected horizon >= {self.input_horizon}, got {horizon}")

        action_flat = base_action[:, : self.input_horizon].reshape(batch_size, -1)
        if self.uses_obs:
            if obs_context is None:
                raise ValueError("obs_context is required when obs_context_dim > 0")
            if obs_context.ndim != 2 or obs_context.shape[0] != batch_size:
                raise ValueError(
                    "obs_context must be [B,C], "
                    f"got {tuple(obs_context.shape)} for batch_size={batch_size}"
                )
            if obs_context.shape[-1] != self.obs_context_dim:
                raise ValueError(f"Expected obs_context_dim={self.obs_context_dim}, got {obs_context.shape[-1]}")
            inputs = torch.cat([action_flat, obs_context], dim=-1)
        else:
            inputs = action_flat

        hidden = self.net(inputs)
        raw_unit = torch.tanh(self.delta_head(hidden))
        gate = torch.sigmoid(self.gate_head(hidden))
        delta_first = raw_unit * self.max_delta.view(1, -1) * gate * self.action_mask.view(1, -1)

        delta = torch.zeros_like(base_action)
        delta[:, 0, :] = delta_first
        info = {
            "raw_unit": raw_unit,
            "gate": gate,
            "gate_mean": gate.mean(),
            "delta_abs_mean": delta.abs().mean(),
            "delta_first_abs_mean": delta_first.abs().mean(),
            "head_type": "executable_step",
        }
        if return_info:
            return delta, info
        return delta
