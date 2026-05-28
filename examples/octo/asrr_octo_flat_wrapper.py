from __future__ import annotations

from typing import Any, Optional, Sequence

import jax
import jax.numpy as jnp

from asrr_core.jax_refiners import create_jax_flat_adapter


INTERNAL_FEATURE_MODES = {
    "readout_action",
    "readout_action_task",
    "readout_action_lang",
    "full_internal",
}


def task_one_hot(task_ids: jax.Array, batch_size: int, num_tasks: int) -> jax.Array:
    task_ids = jnp.asarray(task_ids, dtype=jnp.int32).reshape((-1,))
    if task_ids.shape[0] != batch_size:
        raise ValueError(f"task_ids batch size {task_ids.shape[0]} does not match batch_size={batch_size}")
    task_ids = jnp.clip(task_ids, 0, int(num_tasks) - 1)
    return jax.nn.one_hot(task_ids, int(num_tasks), dtype=jnp.float32)


def require_internal_feature(
    internal_features: dict[str, jax.Array],
    name: str,
    batch_size: int,
) -> jax.Array:
    if name not in internal_features:
        raise KeyError(f"Missing internal feature {name!r}; available={sorted(internal_features)}")
    feature = jnp.asarray(internal_features[name], dtype=jnp.float32)
    if feature.ndim != 2 or feature.shape[0] != batch_size:
        raise ValueError(f"Internal feature {name!r} must be [B,C], got {feature.shape}")
    return feature


def build_octo_flat_features(
    base_action: jax.Array,
    *,
    feature_mode: str,
    task_ids: Optional[jax.Array] = None,
    num_tasks: int = 10,
    internal_features: Optional[dict[str, jax.Array]] = None,
) -> jax.Array:
    """Build flat ASRR features from Octo outputs.

    Octo-specific code should produce `base_action` and `internal_features`.
    This helper only concatenates already-extracted tensors.
    """

    if base_action.ndim != 3:
        raise ValueError(f"base_action must be [B,H,D], got {base_action.shape}")
    batch_size = int(base_action.shape[0])
    action_flat = jnp.asarray(base_action, dtype=jnp.float32).reshape((batch_size, -1))

    if feature_mode == "action":
        return action_flat
    if feature_mode == "action_task":
        return jnp.concatenate([action_flat, task_one_hot(task_ids, batch_size, num_tasks)], axis=-1)

    if feature_mode not in INTERNAL_FEATURE_MODES:
        raise ValueError(f"Unsupported Octo ASRR feature_mode={feature_mode}")
    if internal_features is None:
        raise ValueError(f"internal_features must be provided for feature_mode={feature_mode}")

    parts = [action_flat, require_internal_feature(internal_features, "readout_action", batch_size)]
    if feature_mode in {"readout_action_task", "full_internal"}:
        parts.append(task_one_hot(task_ids, batch_size, num_tasks))
    if feature_mode in {"readout_action_lang", "full_internal"}:
        parts.append(require_internal_feature(internal_features, "task", batch_size))
    if feature_mode == "full_internal":
        parts.append(require_internal_feature(internal_features, "obs_primary", batch_size))
        parts.append(require_internal_feature(internal_features, "obs_wrist", batch_size))
    return jnp.concatenate(parts, axis=-1)


def make_octo_flat_refiner(config: dict[str, Any]):
    """Create the JAX ASRR refiner from an Octo-style adapter config."""

    return create_jax_flat_adapter(config)


def make_octo_flat_config(
    *,
    feature_mode: str,
    feature_dim: int,
    action_horizon: int = 4,
    action_dim: int = 7,
    hidden_dims: Sequence[int] = (256, 256),
    max_correction: float = 0.25,
    adapter_type: str = "residual_mlp",
    gate_bias_init: float = -2.0,
    num_tasks: int = 10,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "feature_mode": str(feature_mode),
        "feature_dim": int(feature_dim),
        "action_horizon": int(action_horizon),
        "action_dim": int(action_dim),
        "hidden_dims": [int(x) for x in hidden_dims],
        "max_correction": float(max_correction),
        "adapter_type": str(adapter_type),
        "gate_bias_init": float(gate_bias_init),
        "num_tasks": int(num_tasks),
        "metadata": metadata or {},
    }


def apply_octo_flat_refiner(
    refiner,
    params: Any,
    base_action: jax.Array,
    features: jax.Array,
    *,
    alpha: float = 1.0,
    action_mask: Optional[jax.Array] = None,
) -> tuple[jax.Array, jax.Array]:
    """Return `(delta_action, refined_action)` for an Octo action chunk."""

    delta = refiner.apply(params, features)
    if action_mask is not None:
        mask = jnp.asarray(action_mask, dtype=delta.dtype).reshape((1, 1, -1))
        delta = delta * mask
    refined = jnp.asarray(base_action, dtype=jnp.float32) + float(alpha) * delta
    return delta, refined
