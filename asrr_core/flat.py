from __future__ import annotations

from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn


class FlatFeatureResidualAdapter(nn.Module):
    """MLP residual refiner from a flat feature vector to an action chunk.

    This is the PyTorch reference form for the Octo refiner family, where
    policy-specific code builds a compact flat feature vector such as:

        action
        action + task one-hot
        action + readout_action
        action + readout_action + task one-hot

    The adapter itself has no dependency on Octo, LIBERO, or VLA internals.
    """

    def __init__(
        self,
        feature_dim: int,
        action_horizon: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        max_correction: Union[float, Sequence[float]] = 0.25,
        adapter_type: str = "residual_mlp",
        gate_bias_init: float = -2.0,
        action_mask: Union[None, Sequence[float]] = None,
        layer_norm: bool = False,
    ):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if adapter_type not in {"residual_mlp", "gated_mlp"}:
            raise ValueError("adapter_type must be either 'residual_mlp' or 'gated_mlp'")
        hidden_dims = tuple(int(x) for x in hidden_dims)
        if not hidden_dims or any(x <= 0 for x in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")

        self.feature_dim = int(feature_dim)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.hidden_dims = hidden_dims
        self.adapter_type = str(adapter_type)
        self.gate_bias_init = float(gate_bias_init)

        layers = []
        in_dim = self.feature_dim
        for hidden_dim in hidden_dims:
            if layer_norm:
                layers.append(nn.LayerNorm(in_dim))
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

        output_dim = self.action_horizon * self.action_dim
        self.delta_head = nn.Linear(in_dim, output_dim)
        self.gate_head = nn.Linear(in_dim, output_dim)

        if isinstance(max_correction, (list, tuple)):
            if len(max_correction) != self.action_dim:
                raise ValueError("max_correction sequence must match action_dim")
            max_tensor = torch.tensor([float(x) for x in max_correction], dtype=torch.float32)
        else:
            max_tensor = torch.full((self.action_dim,), float(max_correction), dtype=torch.float32)
        self.register_buffer("max_correction", max_tensor)

        if action_mask is None:
            mask = torch.ones(self.action_dim, dtype=torch.float32)
        else:
            if len(action_mask) != self.action_dim:
                raise ValueError("action_mask sequence must match action_dim")
            mask = torch.tensor([float(x) for x in action_mask], dtype=torch.float32)
        self.register_buffer("action_mask", mask)
        self.reset_parameters()

    @property
    def uses_gate(self) -> bool:
        return self.adapter_type == "gated_mlp"

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, self.gate_bias_init)
        if not self.uses_gate:
            for parameter in self.gate_head.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        features: torch.Tensor,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if features.ndim != 2:
            raise ValueError(f"features must be [B,C], got {tuple(features.shape)}")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {features.shape[-1]}")

        hidden = self.net(features)
        raw = self.delta_head(hidden).view(-1, self.action_horizon, self.action_dim)
        raw_delta = torch.tanh(raw) * self.max_correction.view(1, 1, -1)
        if self.uses_gate:
            gate = torch.sigmoid(self.gate_head(hidden)).view(-1, self.action_horizon, self.action_dim)
        else:
            gate = torch.ones_like(raw_delta)
        delta = raw_delta * gate * self.action_mask.view(1, 1, -1)
        info = {
            "raw_delta": raw_delta,
            "gate": gate,
            "gate_mean": gate.mean(),
            "delta_abs_mean": delta.abs().mean(),
            "adapter_type": self.adapter_type,
            "head_type": self.adapter_type,
        }
        if return_info:
            return delta, info
        return delta
