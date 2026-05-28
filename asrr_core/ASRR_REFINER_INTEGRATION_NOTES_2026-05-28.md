# ASRR Refiner Integration Notes

This document summarizes how the ASRR refiner family maps to common robot
policy families.  It is written as implementation guidance for the public
repository.

## Unified Contract

```text
base_action = frozen_policy(observation, task)      # [B, H, D]
delta_action = asrr_refiner(base_action, context)   # [B, H, D]
refined_action = base_action + alpha * delta_action # [B, H, D]
```

The base policy stays frozen.  ASRR trains a small residual model on top of that
base action sequence.

## Core Refiner Families

| Refiner | Use case |
|---|---|
| `ActionSequenceResidualAdapter` | Full action-sequence residual refinement with optional state/action/task context. |
| `ExecutableStepResidualAdapter` | Receding-horizon policies that execute only the first action from a predicted chunk. |
| `FlatFeatureResidualAdapter` | PyTorch flat-feature residual refinement. |
| `JaxFlatFeatureResidualAdapter` | JAX/Flax flat-feature residual refinement for Octo-style wrappers. |
| `JaxGatedFlatFeatureResidualAdapter` | Gated JAX/Flax flat-feature residual refinement. |

## ACT

ACT produces action chunks.  The wrapper usually provides robot state (`qpos`)
as optional ASRR context.

Recommended variants:

| Variant | Core setting |
|---|---|
| action-only | `fusion_mode="action_only"` |
| qpos/state-add | `fusion_mode="state_add"` |
| qpos/state-concat | `fusion_mode="state_concat"` |
| FiLM state | `fusion_mode="film"` |

Public example:

```text
examples/act/asrr_act_wrapper.py
```

## Diffusion Policy

Diffusion Policy produces an executable action chunk.  The wrapper typically
flattens low-dimensional observation history into `obs_context`.

Recommended variants:

| Variant | Description |
|---|---|
| `asrr_action` | Action-only residual baseline. |
| `asrr_obs_add` | Observation/state-conditioned residual. |
| `asrr_mode_embed` | GRU mode embedding from the base action chunk and observation context. |
| `asrr_obs_mode` | Observation plus trajectory mode context. |
| `asrr_mode_moe` | Mixture of residual experts routed by mode context. |

Public example:

```text
examples/diffusion_policy/asrr_dp_wrapper.py
```

## Octo

Octo wrappers can construct a flat feature vector from the base action chunk,
task identifiers, and Octo internal readout features.

Common feature modes:

| Mode | Feature vector |
|---|---|
| `action` | `flatten(base_action)` |
| `action_task` | `flatten(base_action) + task_one_hot` |
| `readout_action` | `flatten(base_action) + readout_action` |
| `readout_action_task` | `flatten(base_action) + readout_action + task_one_hot` |

The Octo wrapper owns transformer execution, token pooling, image preprocessing,
task text handling, and action statistics.  The ASRR JAX refiner only maps flat
features to residual action chunks.

Public example:

```text
examples/octo/asrr_octo_flat_wrapper.py
```

## SmolVLA

SmolVLA-style runners often predict a long action chunk and execute one action
per query.  The recommended ASRR structure is executable-step refinement:

```text
input:  base_action[:, :input_horizon] + optional low-dimensional state
output: delta[:, 0]
delta[:, 1:] = 0
```

Public example:

```text
examples/smolvla/asrr_smolvla_wrapper.py
```

## pi0.5

pi0.5-style runners can use the same executable-step ASRR structure when the
policy predicts an action chunk and executes actions in a receding-horizon loop.

Public example:

```text
examples/pi05/asrr_pi05_wrapper.py
```

## OpenVLA-OFT

OpenVLA-OFT wrappers can expose normalized action chunks and VLA internal action
features:

```text
base_action_norm: [B, H, D]
action_context:   [B, H, C]
obs_context:      [B, C_obs]
task_index:       [B]
```

`learned_pool` and `dimtoken` OpenVLA-OFT variants use per-action-dimension
contexts with shape `[B, H, D, C]`.

Recommended mappings:

| OpenVLA-OFT structure | ASRR mapping |
|---|---|
| low-rank temporal residual | `OpenVLAOFTResidualAdapter(head_type="lowrank_delta")` |
| gated residual | `OpenVLAOFTResidualAdapter(head_type="sigmoid_gate")` |
| direct bounded residual | `OpenVLAOFTResidualAdapter(head_type="bounded_delta_only")` |
| MLP-mixer gated residual | `OpenVLAOFTResidualAdapter(head_type="mlp_mixer_gate")` |
| mean, pooled, or dimension-token context | `OpenVLAOFTResidualAdapter(context_mode=...)` |

The OpenVLA-OFT wrapper owns processor calls, action-head calls, proprio/state
normalization, action unnormalization, task mapping, and action queueing.

Public example:

```text
examples/openvla_oft/asrr_openvla_oft_wrapper.py
```

## Release Boundary

The public package should include:

- `asrr_core`;
- lightweight method examples;
- interface documentation;
- tensor-level tests.

The public package should not include:

- datasets;
- model checkpoints;
- generated videos;
- experiment logs;
- machine-specific paths.
