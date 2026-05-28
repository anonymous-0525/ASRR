from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


class JaxFlatFeatureResidualAdapter(nn.Module):
    """Flax/JAX flat-feature residual refiner used by Octo-style wrappers."""

    hidden_dims: tuple[int, ...]
    action_horizon: int
    action_dim: int
    max_correction: float = 0.25

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        x = nn.Dense(
            self.action_horizon * self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)
        x = jnp.tanh(x) * self.max_correction
        return x.reshape((features.shape[0], self.action_horizon, self.action_dim))


class JaxGatedFlatFeatureResidualAdapter(nn.Module):
    """Flax/JAX gated flat-feature refiner for Octo readout/action/task inputs."""

    hidden_dims: tuple[int, ...]
    action_horizon: int
    action_dim: int
    max_correction: float = 0.25
    gate_bias_init: float = -2.0

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        x = features
        for hidden_dim in self.hidden_dims:
            x = nn.LayerNorm()(x)
            x = nn.Dense(hidden_dim)(x)
            x = nn.gelu(x)
        output_dim = self.action_horizon * self.action_dim
        delta = nn.Dense(
            output_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="delta_head",
        )(x)
        gate = nn.Dense(
            output_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(self.gate_bias_init),
            name="gate_head",
        )(x)
        delta = jnp.tanh(delta) * self.max_correction
        gated_delta = delta * jax.nn.sigmoid(gate)
        return gated_delta.reshape((features.shape[0], self.action_horizon, self.action_dim))


def parse_hidden_dims(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        dims = tuple(int(item) for item in value)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError(f"Invalid hidden dims: {value}")
    return dims


def create_jax_flat_adapter(config: dict[str, Any]) -> nn.Module:
    adapter_type = str(config.get("adapter_type", "residual_mlp"))
    kwargs = {
        "hidden_dims": parse_hidden_dims(config["hidden_dims"]),
        "action_horizon": int(config["action_horizon"]),
        "action_dim": int(config["action_dim"]),
        "max_correction": float(config["max_correction"]),
    }
    if adapter_type == "gated_mlp":
        return JaxGatedFlatFeatureResidualAdapter(
            **kwargs,
            gate_bias_init=float(config.get("gate_bias_init", -2.0)),
        )
    if adapter_type == "residual_mlp":
        return JaxFlatFeatureResidualAdapter(**kwargs)
    raise ValueError(f"Unknown adapter_type={adapter_type!r}")
