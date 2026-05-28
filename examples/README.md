# ASRR Examples

This directory contains lightweight examples for integrating `asrr_core` with
existing robot-learning policies.

The examples intentionally do not copy full ACT, Diffusion Policy, Octo,
SmolVLA, pi0.5, or OpenVLA-OFT repositories.  They show the boundary:

```text
method-specific policy/environment code -> tensors -> asrr_core -> refined tensors
```

## What Belongs Here

Include:

- small policy wrappers;
- cache schemas;
- minimal training/evaluation commands;
- notes about action normalization and executed horizons.

Do not include:

- checkpoints;
- datasets;
- long experiment logs;
- project-specific absolute paths;
- method source trees that should remain external dependencies.

## Available Examples

| Example | Purpose |
|---|---|
| `act/` | Wrap a frozen ACT policy with `ActionSequenceResidualAdapter`. |
| `diffusion_policy/` | Wrap Diffusion Policy action chunks with ASRR action/state/mode variants. |
| `octo/` | Build Octo flat features and run the optional JAX ASRR refiner. |
| `smolvla/` | Wrap SmolVLA-style chunk policies with executable-step ASRR. |
| `pi05/` | Generic pi0.5-style executable-step ASRR wrapper. |
| `openvla_oft/` | Refine normalized OpenVLA-OFT action chunks with action context. |

These examples are intentionally thin.  Real experiments still require the
corresponding base method repository, model weights, environment, datasets, and
normalization statistics.

## What Clone Alone Can Do

A fresh clone can run tensor-level smoke tests and show how to instantiate ASRR
modules.  It cannot run ACT/DP/VLA rollouts by itself, because the base policy
and environment are external dependencies.

To run a real VLA experiment, a user must install and configure the target
method first.  For example, OpenVLA-OFT use requires:

```text
1. install OpenVLA-OFT and LIBERO dependencies
2. download/load an OpenVLA-OFT checkpoint
3. implement predict_base_and_context(...) that returns:
   - base_action_norm
   - action_context
   - obs_context
   - task_index
4. instantiate examples/openvla_oft/OpenVLAOFTASRRWrapper
5. run the original OpenVLA-OFT LIBERO loop with the refined actions
```

## Shared Contract

Every example follows:

```text
base_action = frozen_policy(...)
delta_action = refiner(base_action, context)
refined_action = base_action + alpha * delta_action
```

Training uses expert actions:

```text
target_action = expert_action
loss = distance(base_action + alpha * delta_action, target_action)
```

Evaluation should report paired base/refined metrics:

```text
base_success
refined_success
rescue
negative/regression
preserve
both_fail
```
