# ASRR: Action Sequence Residual Refinement

ASRR is a lightweight refiner for robot action policies.  It keeps an existing
policy frozen, predicts a bounded residual action sequence, and executes the
refined action:

```text
base_action = frozen_policy(observation, task)      # [B, H, D]
delta_action = asrr_refiner(base_action, context)   # [B, H, D]
refined_action = base_action + alpha * delta_action # [B, H, D]
```

The repository is designed to be policy-agnostic.  ASRR does not replace ACT,
Diffusion Policy, Octo, SmolVLA, pi0.5, or OpenVLA-OFT.  It provides reusable
residual-refinement modules and example wrappers that connect those modules to
external policies.

## Repository Layout

```text
asrr_core/
  models.py          # general action-sequence residual refiner
  executable.py      # first-executable-action residual refiner
  flat.py            # flat-feature residual refiner
  losses.py          # supervised residual losses
  runtime.py         # residual application helpers
  jax_refiners.py    # optional JAX/Flax flat-feature refiners

examples/
  act/
  diffusion_policy/
  octo/
  smolvla/
  pi05/
  openvla_oft/
```

## Installation

Install the core package in editable mode:

```bash
git clone https://github.com/anonymous-0525/ASRR.git
cd ASRR
pip install -e .
```

For Octo-style JAX/Flax refiners, install the optional JAX dependencies:

```bash
pip install -e ".[jax]"
```

The examples for ACT, Diffusion Policy, Octo, SmolVLA, pi0.5, and OpenVLA-OFT
require the corresponding external method repositories, model checkpoints,
datasets, environments, and preprocessing utilities.  This repository provides
the ASRR refiner and integration pattern; it does not bundle external robot
policy repositories or pretrained weights.

For method-by-method reproducibility status, see:

- `METHOD_SUPPORT_MATRIX.md` for what a fresh clone can and cannot run;
- `PATCHES.md` for method-side wrapper and hook requirements.

## Quick Tensor Smoke Test

```python
import torch
from asrr_core import ActionSequenceResidualAdapter, apply_residual

base_action = torch.randn(2, 8, 7)       # [B, H, D]
state_context = torch.randn(2, 16)       # optional context

refiner = ActionSequenceResidualAdapter(
    action_dim=7,
    horizon=8,
    state_context_dim=16,
    fusion_mode="state_add",
    hidden_dim=128,
    head_type="sigmoid_gate",
    max_delta=0.05,
)

delta, info = refiner(base_action, state_context=state_context, return_info=True)
refined_action = apply_residual(base_action, delta, alpha=1.0)

print(refined_action.shape)  # torch.Size([2, 8, 7])
```

## Refiner Families

### `ActionSequenceResidualAdapter`

Use this for policies that predict action chunks and can optionally provide
state, task, or action-context features.

Typical users:

- ACT
- Diffusion Policy
- OpenVLA-OFT sequence-context variants
- action-chunk policies with low-dimensional state context

Supported fusion modes:

```text
action_only
state_add
state_concat
film
action_context_add
state_action_context_add
```

Supported heads:

```text
dense
bounded_dense
sigmoid_gate
lowrank_delta
```

### `ExecutableStepResidualAdapter`

Use this for policies that predict a long chunk but execute only the first
action at each control step.

Typical users:

- SmolVLA-style runners
- pi0.5-style runners
- receding-horizon VLA policies that execute one action per query

The adapter reads `base_action[:, :input_horizon]`, predicts a bounded residual
for `delta[:, 0]`, and returns a full `[B, H, D]` residual with zero residuals
for all later timesteps.

### `FlatFeatureResidualAdapter`

Use this when the method wrapper builds a single flat feature vector before
calling ASRR.

Typical users:

- Octo-style action/readout/task feature adapters
- lightweight wrappers that concatenate action chunks with method-specific
  embeddings

### JAX/Flax Refiners

Octo-style code can use:

```python
from asrr_core.jax_refiners import (
    JaxFlatFeatureResidualAdapter,
    JaxGatedFlatFeatureResidualAdapter,
)
```

These modules are not imported by default, so PyTorch-only users do not need
JAX or Flax installed.

## Connecting ASRR To A Policy

Every integration follows the same pattern:

1. Load a frozen base policy.
2. Run the base policy to obtain `base_action`.
3. Build optional ASRR context tensors.
4. Train ASRR against expert actions or other action targets.
5. During rollout, compute `refined_action = base_action + alpha * delta_action`.
6. Convert `refined_action` back to the environment's action format.
7. Evaluate with paired base/refined seeds or initial states.

The method wrapper owns all method-specific logic:

- image and language preprocessing;
- action normalization and unnormalization;
- hidden-feature extraction;
- task indexing;
- environment creation;
- action queueing and execution horizon.

`asrr_core` owns only the residual module, bounds, gates, losses, and residual
application.

For full VLA experiments, the external method runner must provide the tensors
expected by the relevant `examples/<method>/` wrapper.  The public ASRR package
contains lightweight wrappers and compatibility modules, not the full
OpenVLA-OFT, SmolVLA, Octo, or pi0.5 training/evaluation stacks.

## Method Examples

### ACT

Use `examples/act/asrr_act_wrapper.py`.

ACT produces action chunks and often exposes robot joint state (`qpos`).  A
typical ASRR configuration is:

```python
from examples.act.asrr_act_wrapper import ASRRACTWrapper, build_act_asrr_refiner

refiner = build_act_asrr_refiner(
    action_dim=14,
    horizon=80,
    variant="qpos_add",
    qpos_dim=14,
    hidden_dim=256,
)

policy = ASRRACTWrapper(
    base_policy=frozen_act_policy,
    refiner=refiner,
    alpha=1.0,
)
```

### Diffusion Policy

Use `examples/diffusion_policy/asrr_dp_wrapper.py`.

Diffusion Policy predicts an action chunk from observation history.  The wrapper
can use action-only, observation-conditioned, mode-embedding, observation-mode,
or mixture-of-experts variants.

```python
from examples.diffusion_policy.asrr_dp_wrapper import (
    ASRRDiffusionPolicyWrapper,
    DPASRRAdapter,
)

adapter = DPASRRAdapter(
    action_dim=2,
    horizon=8,
    asrr_variant="asrr_obs_add",
    obs_context_dim=30,
    hidden_dim=256,
)

policy = ASRRDiffusionPolicyWrapper(
    base_policy=frozen_dp_policy,
    adapter=adapter,
    alpha=0.5,
)
```

### Octo

Use `examples/octo/asrr_octo_flat_wrapper.py`.

Octo wrappers usually build a flat feature vector from an action chunk and
optional Octo internal features:

```text
action
action + task_one_hot
action + readout_action
action + readout_action + task_one_hot
```

The Octo runner must provide the base action, task ids, internal features, and
action statistics.

### SmolVLA

Use `examples/smolvla/asrr_smolvla_wrapper.py`.

SmolVLA-style runners commonly predict a long chunk and execute one action.  Use
`ExecutableStepResidualAdapter` to refine the first executable action.

### pi0.5

Use `examples/pi05/asrr_pi05_wrapper.py`.

The pi0.5 example follows the same executable-step pattern as SmolVLA.  A pi0.5
runner must supply the base action chunk, optional low-dimensional context, and
the method-specific action conversion logic.

### OpenVLA-OFT

Use `examples/openvla_oft/asrr_openvla_oft_wrapper.py`.

The OpenVLA-OFT runner should expose normalized action chunks and VLA internal
action context:

```text
base_action_norm: [H, D] or [B, H, D]
action_context:   [H, C] or [B, H, C]
obs_context:      [C_obs] or [B, C_obs]
task_index:       int or [B]
```

For OpenVLA-OFT `learned_pool` and `dimtoken` variants, `action_context` is
`[B, H, D, C]`.  The default `build_openvla_oft_refiner` creates the
local-compatible OpenVLA-OFT refiner, including `sigmoid_gate`,
`bounded_delta_only`, `lowrank_delta`, and `mlp_mixer_gate` heads.

The ASRR wrapper returns refined normalized actions.  The OpenVLA-OFT runner
then unnormalizes them and sends them to the environment.

## What A Fresh Clone Can Run

A fresh clone can:

- import `asrr_core`;
- instantiate ASRR modules;
- run tensor-level smoke tests;
- inspect example wrappers.

A fresh clone cannot run ACT, DP, Octo, SmolVLA, pi0.5, or OpenVLA-OFT rollouts
without the corresponding external method setup.  Real experiments require the
base method repository, checkpoint, dataset, environment, preprocessing, and
action normalization.

## Example Real Integration Flow

For OpenVLA-OFT:

1. Install OpenVLA-OFT and its robot evaluation dependencies.
2. Load a frozen OpenVLA-OFT checkpoint.
3. Implement a function:

```python
def predict_base_and_context(observation, task_text):
    return {
        "base_action_norm": base_action_norm,
        "action_context": action_context,
        "obs_context": obs_context,
        "task_index": task_index,
    }
```

4. Build the ASRR refiner and wrapper:

```python
from examples.openvla_oft.asrr_openvla_oft_wrapper import (
    OpenVLAOFTASRRWrapper,
    build_openvla_oft_refiner,
)

refiner = build_openvla_oft_refiner(
    action_dim=7,
    horizon=8,
    action_context_dim=4096,
    obs_context_dim=8,
    num_task_embeddings=10,
)

policy = OpenVLAOFTASRRWrapper(
    predict_base_and_context=predict_base_and_context,
    refiner=refiner,
    alpha=0.5,
    device="cuda",
    unnormalize_fn=unnormalize_openvla_actions,
)
```

5. Use the original OpenVLA-OFT environment loop, but call `policy` for refined
   actions.

## Evaluation Metrics

Report paired base/refined metrics:

```text
base_success
refined_success
absolute_gain
rescue                 # base failed, refined succeeded
negative/regression    # base succeeded, refined failed
preserve               # base succeeded, refined succeeded
both_fail              # base failed, refined failed
```

Paired evaluation is strongly recommended because ASRR is a residual correction
method.  It is important to know whether gains come from rescuing failed base
episodes or from changing the behavior of already successful episodes.
