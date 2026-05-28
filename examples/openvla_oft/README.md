# OpenVLA-OFT + ASRR Example

OpenVLA-OFT produces normalized action chunks plus VLA internal action context.

This example provides two mappings:

- `OpenVLAOFTResidualAdapter`: the OpenVLA-OFT refiner structure used by the
  local experiments.  It is intended for reproducing those adapter checkpoints
  and covers `mean_context`, `learned_pool`, `dimtoken`, `sigmoid_gate`,
  `bounded_delta_only`, `lowrank_delta`, and `mlp_mixer_gate`.
- `ActionSequenceResidualAdapter`: a policy-agnostic ASRR sequence refiner that
  is useful when direct checkpoint compatibility is not required.

## Interface

Expected predictor output:

```text
base_action_norm: [H, D] or [B, H, D]
action_context:   [H, C] or [B, H, C]
obs_context:      [C_obs] or [B, C_obs]
task_index:       int or [B]
```

For `learned_pool` and `dimtoken`, `action_context` is `[B, H, D, C]`.

The ASRR wrapper applies:

```text
delta_norm = refiner(base_action_norm, action_context, obs_context, task_index)
refined_norm = clip(base_action_norm + alpha * delta_norm)
```

A method runner should then unnormalize `refined_norm` back to environment
actions using OpenVLA-OFT action statistics.

## Recommended Mapping

| OpenVLA-OFT structure | ASRR mapping |
|---|---|
| `lowrank_delta` | `OpenVLAOFTResidualAdapter(head_type="lowrank_delta")` |
| `sigmoid_gate` | `OpenVLAOFTResidualAdapter(head_type="sigmoid_gate")` |
| `bounded_delta_only` | `OpenVLAOFTResidualAdapter(head_type="bounded_delta_only")` |
| `mlp_mixer_gate` | `OpenVLAOFTResidualAdapter(head_type="mlp_mixer_gate")` |
| `learned_pool` / `dimtoken` contexts | `OpenVLAOFTResidualAdapter(context_mode=...)` |

Use `build_openvla_oft_refiner` for the local-compatible OpenVLA-OFT structure.
Use `build_openvla_oft_core_refiner` only for a generic ASRR implementation that
does not need to load OpenVLA-OFT adapter checkpoints.

## Files

```text
asrr_openvla_oft_wrapper.py
openvla_oft_refiner.py
```

The wrapper does not import OpenVLA-OFT.  It expects a callable that returns the
base normalized actions and contexts.
