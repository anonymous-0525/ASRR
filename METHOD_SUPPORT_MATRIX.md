# ASRR Method Support Matrix

This repository is split into a method-agnostic ASRR core and thin method
examples.  The examples define the tensor boundary needed to connect ASRR to
external robot-policy repositories.  They are not full copies of those external
methods.

## Status Summary

| Method | ASRR refiner family | Public package status | Production runner status | Clone-alone status |
|---|---|---|---|---|
| ACT | `ActionSequenceResidualAdapter` | Lightweight wrapper included | Mature project integration | Tensor wrapper only |
| Diffusion Policy | `ActionSequenceResidualAdapter` | Lightweight wrapper included | Mature project integration | Tensor wrapper only |
| Octo | `JaxFlatFeatureResidualAdapter` / `JaxGatedFlatFeatureResidualAdapter` | Flat-feature wrapper included | Requires external Octo stack, fresh fine-tuning, and method runner | Cannot run Octo rollouts by itself |
| SmolVLA | `ExecutableStepResidualAdapter` | Executable-step wrapper included | Requires external SmolVLA/LeRobot runner and cache builder | Cannot run SmolVLA rollouts by itself |
| pi0.5 | `ExecutableStepResidualAdapter` | Generic executable-step wrapper included | Fine-tuning path is external; ASRR hidden-state/refiner runner must be supplied by the method integration | Wrapper contract only |
| OpenVLA-OFT | `OpenVLAOFTResidualAdapter` or generic `ActionSequenceResidualAdapter` | OpenVLA-OFT-compatible wrapper/refiner included | Requires external OpenVLA-OFT runner, action statistics, and cache builder | Cannot run OpenVLA-OFT rollouts by itself |

## What A Fresh Clone Can Do

A fresh clone can:

- install and import `asrr_core`;
- instantiate ASRR refiners;
- run tensor-level tests;
- inspect method wrapper contracts in `examples/`;
- use the wrappers inside an existing method runner after the runner provides
  the required tensors.

A fresh clone cannot:

- download or load method checkpoints automatically;
- create LIBERO or real-robot environments;
- run ACT, Diffusion Policy, Octo, SmolVLA, pi0.5, or OpenVLA-OFT rollouts
  without installing the corresponding external method stack;
- reproduce long experiments without the project-specific harness, datasets,
  checkpoints, and environment configuration.

## Required Method-Side Responsibilities

Every production integration must own the following method-specific pieces:

- image, proprioception, and language preprocessing;
- base policy loading and inference;
- action normalization and unnormalization;
- hidden-feature extraction, when a selected ASRR variant uses internal
  features;
- task indexing and suite/task metadata;
- dataset iteration and cache construction;
- environment creation, action queueing, and rollout control;
- paired base/refined evaluation and rescue/regression accounting.

`asrr_core` owns only the residual modules, bounds, gates, losses, and runtime
residual application.

## Method Notes

### ACT

ACT action chunks can be refined directly.  The wrapper normally passes robot
state such as `qpos` as ASRR state context.  Public code includes the minimal
wrapper; complete experiment scripts remain outside this package.

### Diffusion Policy

Diffusion Policy action chunks can be refined with action-only, observation
conditioned, trajectory-mode, or mixture-of-experts variants.  Public code
includes the wrapper and configuration pattern; complete training and rollout
scripts remain method specific.

### Octo

Octo is JAX/Flax based.  The public example builds flat features from base
actions, task ids, and optional Octo internal readout features, then calls the
optional JAX ASRR refiner.  A real Octo experiment still needs an Octo runner
that performs model loading, fresh fine-tuning when required, transformer
execution, token/readout extraction, action statistics, and LIBERO rollout.

### SmolVLA

SmolVLA-style policies commonly produce a long action chunk while the rollout
executes only the next action.  The public example uses executable-step ASRR:
the refiner reads a short prefix of the base action chunk and optional
low-dimensional context, predicts `delta[:, 0]`, and keeps later residual
steps at zero.  A real SmolVLA experiment still needs a LeRobot/SmolVLA runner,
dataset conversion, cache construction, and no-video rollout evaluation.

### pi0.5

The public package includes the same executable-step contract used for
receding-horizon VLA policies.  The package does not include a complete pi0.5
ASRR hidden-state cache or refiner runner.  Production use should treat pi0.5
as supported at the interface level until the method-specific runner provides
base actions, optional context, normalization, and rollout evaluation.

### OpenVLA-OFT

OpenVLA-OFT integrations expose normalized action chunks plus internal action
context.  The public package includes an OpenVLA-OFT-compatible ASRR refiner
and wrapper, but the external OpenVLA-OFT runner must still provide processor
calls, action-head outputs, proprio/state handling, action statistics, task
mapping, cache construction, and environment rollout.

## Reporting Requirement

For ASRR experiments, report paired base/refined metrics whenever possible:

```text
base_success
refined_success
absolute_gain
rescue                 # base failed, refined succeeded
negative/regression    # base succeeded, refined failed
preserve               # base succeeded, refined succeeded
both_fail              # base failed, refined failed
```

This is more informative than reporting only success-rate deltas because ASRR
is intended to rescue selected base-policy failures while preserving already
successful behavior.
