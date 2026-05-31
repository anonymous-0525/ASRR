# ASRR Method Interface Guide

ASRR is a policy-agnostic residual refiner.  It does not implement a complete
ACT, Diffusion Policy, Octo, SmolVLA, pi0.5, or OpenVLA-OFT training stack.  It
provides reusable modules and examples for connecting those policies to a shared
residual-refinement interface.

## Core Interface

```text
base_action = frozen_policy(observation, task)      # [B, H, D]
delta_action = asrr_refiner(base_action, context)   # [B, H, D]
refined_action = base_action + alpha * delta_action # [B, H, D]
```

The method wrapper provides the base action and context tensors.  `asrr_core`
computes the residual.

## Package Boundary

```text
asrr_core/
  models.py
  executable.py
  flat.py
  losses.py
  runtime.py
  jax_refiners.py

examples/
  act/
  diffusion_policy/
  octo/
  smolvla/
  pi05/
  openvla_oft/
```

`asrr_core` does not import external robot-policy implementations.  External
method runners import `asrr_core`.

The public repository records this boundary in `METHOD_SUPPORT_MATRIX.md` and
`PATCHES.md`.  Those files are the source of truth for clone-alone capability,
production runner requirements, and method-side hook expectations.

## Entry Point Design

ASRR does not provide a single command such as:

```text
python run_asrr.py --method octo
```

Each base method has different dependencies, policy APIs, datasets,
preprocessors, action statistics, and environment loops.  A robust integration
keeps those details in method-specific wrappers:

```text
method runner -> base action/context tensors -> asrr_core -> refined action
```

The examples show how to build those wrappers.

## Minimal Wrapper Contract

A method integration should implement:

```python
def predict_base_action(batch):
    """Return base_action with shape [B, H, D]."""

def build_asrr_context(batch, base_action):
    """Return context tensors required by the selected ASRR refiner."""

def train_asrr_refiner(cache_or_dataset):
    """Train the refiner against target actions."""

def eval_refined_policy(env, base_policy, refiner):
    """Run paired base/refined evaluation."""
```

Recommended cache schema:

```text
base_action      [N, H, D]
target_action    [N, H, D]
obs_context      [N, C_obs]       optional
action_context   [N, H, C_act]    optional
flat_features    [N, C_flat]      optional
task_index       [N]              optional
metadata         dict
```

## Refiner Selection

| Base policy pattern | Recommended refiner |
|---|---|
| Full action chunk refinement | `ActionSequenceResidualAdapter` |
| First-action receding-horizon execution | `ExecutableStepResidualAdapter` |
| Flat feature vector already built by wrapper | `FlatFeatureResidualAdapter` |
| JAX/Flax flat-feature wrapper | `JaxFlatFeatureResidualAdapter` or `JaxGatedFlatFeatureResidualAdapter` |

## ACT

ACT predicts action chunks and can provide robot state as context.

Recommended variants:

| Variant | Core setup |
|---|---|
| `action_only` | `fusion_mode="action_only"` |
| `qpos_add` | `fusion_mode="state_add"` |
| `qpos_concat` | `fusion_mode="state_concat"` |
| `film` | `fusion_mode="film"` |

Public example:

```text
examples/act/asrr_act_wrapper.py
```

## Diffusion Policy

Diffusion Policy predicts action chunks from observation history.

Recommended variants:

| Variant | Description |
|---|---|
| `asrr_action` | action-only residual |
| `asrr_obs_add` | observation-conditioned residual |
| `asrr_mode_embed` | trajectory-mode context |
| `asrr_obs_mode` | observation plus trajectory mode context |
| `asrr_mode_moe` | mixture of residual experts |

Public example:

```text
examples/diffusion_policy/asrr_dp_wrapper.py
```

## Octo

Octo integrations use the optional JAX/Flax flat-feature refiner.  The Octo
runner constructs feature vectors such as:

```text
flatten(base_action)
flatten(base_action) + task_one_hot
flatten(base_action) + readout_action
flatten(base_action) + readout_action + task_one_hot
```

Public example:

```text
examples/octo/asrr_octo_flat_wrapper.py
```

## SmolVLA

SmolVLA-style runners can use executable-step refinement:

```text
base_action[:, :input_horizon] + obs_context -> delta[:, 0]
delta[:, 1:] = 0
```

Public example:

```text
examples/smolvla/asrr_smolvla_wrapper.py
```

## pi0.5

pi0.5-style runners can use the same executable-step pattern when the policy
predicts a chunk and executes actions in a receding-horizon loop.

Public example:

```text
examples/pi05/asrr_pi05_wrapper.py
```

## OpenVLA-OFT

OpenVLA-OFT runners can provide normalized actions and internal action context:

```text
base_action_norm [B, H, D]
action_context   [B, H, C]
obs_context      [B, C_obs]
task_index       [B]
```

Public example:

```text
examples/openvla_oft/asrr_openvla_oft_wrapper.py
```

## Training Loss

Use `supervised_asrr_loss` for full action-sequence refinement:

```python
from asrr_core import supervised_asrr_loss

delta, info = refiner(base_action, state_context=obs_context, return_info=True)
loss_info = supervised_asrr_loss(
    base_action=base_action,
    delta_action=delta,
    target_action=target_action,
    alpha=train_alpha,
    loss_type="mse",
    delta_l2_weight=1e-3,
    smooth_l2_weight=0.0,
)
loss = loss_info["loss"]
```

For executable-step refinement, train on the first executed action:

```text
loss = distance(base_action[:, 0] + alpha * delta[:, 0], target_action[:, 0])
```

## Evaluation Metrics

Use paired evaluation whenever possible:

```text
base_success
refined_success
absolute_gain
rescue
negative/regression
preserve
both_fail
```

Paired metrics show whether ASRR rescues base-policy failures or changes
already successful behavior.
