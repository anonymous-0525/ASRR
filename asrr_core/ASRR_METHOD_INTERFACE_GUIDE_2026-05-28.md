# ASRR Method Interface Guide

Timestamp: 2026-05-28  
Project root: local development checkout

## 1. Core Idea

ASRR is designed as a small policy-agnostic refiner, not as a replacement for ACT, Diffusion Policy, Octo, SmolVLA, pi0.5, or OpenVLA-OFT.

The shared contract is always:

```text
base_action = frozen_policy(obs, task, context)       # [B, H, D]
delta_action = asrr_refiner(base_action, context)     # [B, H, D]
refined_action = base_action + alpha * delta_action   # [B, H, D]
```

The base policy remains frozen. ASRR only learns a bounded residual correction.

## 2. Code Architecture

The intended open-source architecture is:

```text
asrr_core/
  models.py          # general action-sequence residual core
  executable.py      # first-executable-action refiner
  flat.py            # flat-feature chunk refiner
  losses.py          # supervised residual losses
  runtime.py         # apply_residual, refine_with_adapter
  jax_refiners.py    # optional Flax/JAX flat refiner for Octo-style code

examples/
  act/
  diffusion_policy/
  octo/
  smolvla/
  pi05/
  openvla_oft/
```

Current local project state does not yet have the clean `examples/` release layout. Instead, wrappers currently live in method-specific folders:

```text
third_party/act
third_party/diffusion_policy/dp_residual_adapter
third_party/octo/scripts
vla_residual_adapter
openvla_oft_residual_adapter
```

Important design rule:

```text
asrr_core never imports ACT, DP, Octo, SmolVLA, pi0.5, or OpenVLA-OFT.
```

Method-specific wrappers import `asrr_core`.

## 3. Is There One Entry Point?

There is no single universal runtime entry point such as:

```text
python run_asrr.py --method octo
```

That is intentional for now.

Different robot-learning policies have different environments, dataset formats, action normalization, camera preprocessing, and hidden-feature extraction APIs. A central CLI would either become very fragile or would need to reimplement every method's own runner.

The correct architecture is:

```text
method wrapper decides how to run the base policy
method wrapper extracts tensors
asrr_core refines tensors
method wrapper sends refined action back to its environment
```

In other words, the "entry point" is method-specific:

| Method | Entry point location | ASRR role |
|---|---|---|
| ACT | ACT training/eval script | Wrap ACT policy and refine action chunks |
| Diffusion Policy | DP train/eval adapter scripts | Refine DP predicted action chunks |
| Octo | Octo LIBERO train/eval scripts | Refine Octo action chunks using flat features |
| SmolVLA | SmolVLA train/eval adapter scripts | Refine only the first executable action |
| pi0.5 | pi0.5 wrapper, to be released | Same simple executable-step ASRR pattern |
| OpenVLA-OFT | OpenVLA-OFT train/eval adapter scripts | Refine normalized action chunks with VLA context |

A future release can add a thin registry for convenience, but the registry should only choose a wrapper. It should not put method-specific logic inside `asrr_core`.

## 4. Minimal Wrapper Interface

Any method can connect to ASRR by implementing these four functions:

```python
def predict_base_action(batch) -> torch.Tensor:
    """Return base_action with shape [B, H, D]."""

def build_asrr_context(batch, base_action) -> dict:
    """Return context tensors required by the selected ASRR refiner."""

def train_asrr_refiner(cache_or_dataset) -> None:
    """Train the refiner against expert actions or rollout-derived labels."""

def eval_refined_policy(env, base_policy, refiner) -> dict:
    """Run paired base/refined rollout evaluation."""
```

The cache schema should be as close as possible to:

```text
base_action      [N, H, D]
target_action    [N, H, D]
obs_context      [N, C_obs]       optional
action_context   [N, H, C_act]    optional
flat_features    [N, C_flat]      optional
task_index       [N]              optional
metadata         dict
```

For rollout evaluation, always record paired metrics:

```text
base_success
refined_success
rescue: base fail -> refined success
negative/regression: base success -> refined fail
preserve: base success -> refined success
both_fail: base fail -> refined fail
```

## 5. Choosing The ASRR Refiner

Use this selection rule:

| Base method type | Recommended core refiner |
|---|---|
| Action chunk policy, full chunk executed or meaningful | `ActionSequenceResidualAdapter` |
| Policy executes only the first action from a chunk | `ExecutableStepResidualAdapter` |
| Policy wrapper already builds one flat feature vector | `FlatFeatureResidualAdapter` |
| JAX/Flax VLA wrapper such as Octo | `JaxFlatFeatureResidualAdapter` or `JaxGatedFlatFeatureResidualAdapter` |

### 5.1 General Sequence Refiner

Use:

```python
from asrr_core import ActionSequenceResidualAdapter
```

Typical input:

```text
base_action:    [B, H, D]
state_context:  [B, C_state]       optional
action_context: [B, H, C_action]   optional
task_index:     [B]                optional
```

This is suitable for:

- ACT
- Diffusion Policy
- OpenVLA-OFT sequence-level context modes
- any method that naturally predicts an action chunk and can provide optional context tensors

### 5.2 Executable-Step Refiner

Use:

```python
from asrr_core import ExecutableStepResidualAdapter
```

Typical input:

```text
base_action[:, :input_horizon, :]  # flattened internally
obs_context                        # optional low-dimensional state
```

Typical output:

```text
delta[:, 0, :]  != 0
delta[:, 1:, :] == 0
```

This is suitable for:

- latest SmolVLA simple-ASRR
- pi0.5-style simple ASRR
- any VLA runner that predicts a long chunk but actually executes one action at a time

### 5.3 Flat-Feature Refiner

Use:

```python
from asrr_core import FlatFeatureResidualAdapter
```

Typical input:

```text
features: [B, C_flat]
```

Typical output:

```text
delta_action: [B, H, D]
```

This is suitable when the method wrapper already owns feature construction, for example Octo:

```text
features = concat(action_flat, readout_action, task_one_hot)
```

### 5.4 JAX/Flax Flat Refiner

Use:

```python
from asrr_core.jax_refiners import (
    JaxFlatFeatureResidualAdapter,
    JaxGatedFlatFeatureResidualAdapter,
)
```

This is for Octo-style JAX code. It is not imported by `asrr_core.__init__`, so PyTorch-only users do not need JAX/Flax installed.

## 6. ACT Interface

ACT usually predicts action chunks:

```text
base_action: [B, H, D]
qpos:        [B, C_qpos]
```

Recommended ASRR variants:

| Variant | Core setup |
|---|---|
| action-only | `fusion_mode="action_only"` |
| qpos/state-add | `fusion_mode="state_add"`, `state_context=qpos` |
| qpos/state-concat | `fusion_mode="state_concat"`, `state_context=qpos` |
| FiLM state | `fusion_mode="film"`, `state_context=qpos` |

Wrapper logic:

```python
with torch.no_grad():
    base_action = act_policy(obs)

delta, info = refiner(base_action, state_context=qpos, return_info=True)
refined_action = apply_residual(base_action, delta, alpha=alpha)
```

ACT-specific code should handle:

- camera loading
- qpos extraction
- dataset HDF5 format
- simulation rollout
- selecting the executed chunk/action

## 7. Diffusion Policy Interface

Diffusion Policy predicts an action chunk:

```text
base_action: [B, n_action_steps, D]
obs_context: flatten(obs[:, :n_obs_steps])
```

Recommended ASRR variants:

| Variant | Core setup |
|---|---|
| ASRR-Action-DP | `ActionSequenceResidualAdapter(..., fusion_mode="action_only")` |
| ASRR-State-DP | `ActionSequenceResidualAdapter(..., fusion_mode="state_add")` |
| ASRR-Mode-DP | DP wrapper builds mode context, then passes it as `state_context` |
| ASRR-ObsMode-DP | DP wrapper combines observation and mode context |
| ASRR-MoE-DP | DP wrapper routes among multiple core adapters |

Existing local wrapper:

```text
third_party/diffusion_policy/dp_residual_adapter/asrr_model.py
```

The DP wrapper is the correct place for:

- Hydra config loading
- base checkpoint loading
- base policy EMA selection
- dataset path resolution
- environment runner overrides
- mode embedding and MoE routing

## 8. Octo Interface

Octo is JAX/Flax and uses a different wrapper pattern.

Base Octo output:

```text
base_action: [B, 4, 7]
```

Octo feature options used in this project:

```text
action_only:
  features = flatten(base_action)

action_task:
  features = concat(flatten(base_action), task_one_hot)

readout_action_task:
  features = concat(flatten(base_action), readout_action, task_one_hot)
```

Recommended core refiner:

```python
from asrr_core.jax_refiners import JaxFlatFeatureResidualAdapter
from asrr_core.jax_refiners import JaxGatedFlatFeatureResidualAdapter
```

Wrapper logic:

```python
base_action, internal_features = sample_actions_and_internal_features(...)
features = make_adapter_features(
    observation,
    base_action,
    feature_mode="readout_action_task",
    task_ids=task_ids,
    internal_features=internal_features,
)
delta = refiner.apply(params, features)
refined_action = base_action + alpha * delta
```

Octo-specific code must stay outside `asrr_core`:

- `OctoModel.load_pretrained`
- `model.run_transformer`
- token group pooling
- LIBERO primary/wrist image transforms
- task text to suite-local id mapping
- action statistics and unnormalization

Current local bridge:

```text
third_party/octo/scripts/libero_residual_adapter.py
```

That file now prefers `asrr_core.jax_refiners` when available.

## 9. SmolVLA Interface

The latest SmolVLA result uses simple executable-step ASRR.

Cache fields:

```text
base_action:   [N, 50, 7]
target_action: [N, 50, 7]
obs_context:   [N, 8]
task_index:    [N]
```

Recommended core refiner:

```python
from asrr_core import ExecutableStepResidualAdapter
```

Recommended config:

```text
input_horizon = 5
hidden_dim = 512
max_delta = [0.05, 0.05, 0.05, 0.015, 0.015, 0.015, 0.0]
freeze_last_action_dim = true
gate_bias_init = -2.0
```

Execution:

```text
delta[:, 0, :] is applied
delta[:, 1:, :] is zero
```

Wrapper logic:

```python
base_action = smolvla_policy.predict_action_chunk(batch)
obs_context = flatten_lowdim_state(batch)
delta, info = refiner(base_action, obs_context=obs_context, return_info=True)
refined_action = base_action + alpha * delta
```

SmolVLA-specific code should handle:

- LeRobot policy loading
- LIBERO environment construction
- pre/post processors
- action queueing
- camera mappings
- seed-split evaluation

## 10. pi0.5 Interface

The local project currently has only a result summary for pi0.5:

```text
step10000_all_suites_alpha_0p5_to_4p0_summary.md
```

Based on that record, pi0.5 should be integrated like SmolVLA unless a future pi0.5 wrapper exposes stable internal action features.

Recommended core refiner:

```python
from asrr_core import ExecutableStepResidualAdapter
```

Wrapper requirements:

```text
base_action: [B, H, D]
obs_context or robot state: optional [B, C]
target_action: expert action or offline label [B, H, D]
```

Evaluation should use alpha sweep:

```text
alpha = 0.5, 1.0, 1.5, 2.0, ...
```

and paired rescue/regression accounting.

## 11. OpenVLA-OFT Interface

OpenVLA-OFT refines normalized action chunks:

```text
base_action_norm: [B, 8, 7]
action_context:   [B, 8, 4096] or [B, 8, 7, 4096]
obs_context:      [B, 8]
task_index:       [B]
```

Recommended core mapping:

| Existing OpenVLA-OFT structure | Suggested core mapping |
|---|---|
| `lowrank_delta` | `ActionSequenceResidualAdapter(head_type="lowrank_delta", action_context_dim=...)` |
| `sigmoid_gate` | `ActionSequenceResidualAdapter(head_type="sigmoid_gate", action_context_dim=...)` |
| `bounded_delta_only` | `ActionSequenceResidualAdapter(head_type="bounded_dense", action_context_dim=...)` |
| `dimtoken` | method-specific extension for now |
| `mlp_mixer_gate` | method-specific extension for now |

Wrapper logic:

```python
pred = predict_base_norm_and_context(...)
base = pred["base_action_norm"]
action_context = pred["action_context"]
obs_context = pred["obs_context"]
task_index = suite_local_task_id

delta, info = refiner(
    base,
    action_context=action_context,
    state_context=obs_context,
    task_index=task_index,
    return_info=True,
)
refined_norm = clip(base + alpha * delta, -1, 1)
env_action = unnormalize_for_env(refined_norm)
```

OpenVLA-OFT-specific code should handle:

- processor and VLA loading
- action head and proprio projector
- normalized action clipping
- LIBERO env action conversion
- chunk queueing
- task id resolution

## 12. Training Loss

The shared supervised loss is:

```python
from asrr_core import supervised_asrr_loss
```

Standard use:

```python
delta, info = refiner(base_action, context, return_info=True)
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

For methods that only refine the first executable action, use a first-action loss:

```text
loss = MSE(base_action[:, 0] + alpha * delta[:, 0], target_action[:, 0])
```

and mask non-controlled dimensions such as gripper if needed.

## 13. Evaluation Protocol

Every method should report:

```text
base_success
refined_success
absolute gain
rescue
negative/regression
preserve
both_fail
alpha
episodes
per-task table
```

For fair comparison:

- use paired initial states or paired seeds;
- compare base and refined policy on the same episodes;
- keep the base policy frozen;
- record the exact action horizon and executed horizon;
- record whether gripper is refined or frozen;
- record whether actions are normalized or environment-scale.

## 14. What A New User Must Implement

If someone has a new ACT/DP/VLA policy, they only need to implement:

```text
1. load frozen base policy
2. produce base_action [B,H,D]
3. produce optional context tensor(s)
4. produce target_action [B,H,D] for offline training
5. call ASRR refiner
6. convert refined action back to the environment format
7. run paired evaluation
```

They do not need to modify `asrr_core` unless their method requires a genuinely new residual architecture.

## 15. Release Recommendation

For open source, keep the public package split into:

```text
asrr_core/        # dependency-light refiner library
examples/         # method-specific wrappers
docs/             # interface and experiment protocol notes
```

Do not release historical experiment logs, cached rollouts, or project-specific absolute paths in the package. Use this project as the source for implementation, then clean paths and provide minimal runnable examples.
