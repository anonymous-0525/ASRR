# ASRR Core

This package contains policy-agnostic components for Action Sequence Residual
Refinement.

Core contract:

```text
base_action: [B, H, D]
optional context: qpos / obs / mode / VLA action hidden states
adapter(base_action, context) -> delta_action: [B, H, D]
refined_action = base_action + alpha * delta_action
```

The package should stay independent of ACT, Diffusion Policy, VQ-BeT, SmolVLA,
OpenVLA-OFT, Octo, and pi0/pi0.5 code.  Policy-specific wrappers should live in
their own method directories and import this package.

## Implemented Refiner Families

`ActionSequenceResidualAdapter`

- General `[B,H,D] -> [B,H,D]` sequence residual module.
- Used by ACT/DP-style wrappers and OpenVLA-like action-context wrappers.
- Supports action-only, state fusion, action-context fusion, task embeddings,
  dense/gated/bounded/low-rank heads.

`ExecutableStepResidualAdapter`

- Simple executable-step ASRR used by the latest SmolVLA and pi0.5-style runs.
- Inputs `base_action[:, :input_horizon]` plus optional low-dimensional state.
- Returns a full chunk, but only `delta[:, 0]` is non-zero.
- State-dict names match the old `SimpleExecutableResidualAdapter`.

`FlatFeatureResidualAdapter`

- PyTorch reference for Octo-style flat-feature refiners.
- Policy-specific code constructs flat features such as `action`,
  `action + task`, or `action + readout_action + task`.
- The core module only maps `[B,C] -> [B,H,D]`.

`jax_refiners.py`

- Optional Flax/JAX flat-feature implementation for Octo wrappers.
- Not imported by default, so PyTorch-only users do not need JAX/Flax.

## Boundary

The core owns:

- bounded residual heads
- conservative zero initialization
- gating / low-rank / executable-step behavior
- residual application helpers
- supervised residual losses

Method wrappers own:

- environment setup
- dataset/cache construction
- image/language/internal-feature extraction
- action normalization/unnormalization
- policy-specific rollout loops

## Method Interface Guide

See `ASRR_METHOD_INTERFACE_GUIDE_2026-05-28.md` for the wrapper contract and
method-specific connection notes for ACT, Diffusion Policy, Octo, SmolVLA,
pi0.5, and OpenVLA-OFT.
