# ASRR Refiner Integration Notes

Timestamp: 2026-05-28 17:05 CST  
Project root: local development checkout

## Scope

This note summarizes the latest refiner design used around Octo, SmolVLA, OpenVLA-OFT, and pi0.5, and records how those designs should be folded into a unified `asrr_core` package.

The open-source goal is:

```text
frozen base policy -> base action sequence -> ASRR refiner -> refined action sequence
```

A downstream user should be able to clone `asrr_core`, keep their own ACT, Diffusion Policy, SmolVLA, OpenVLA-OFT, Octo, pi0/pi0.5, or other VLA policy code unchanged, and only write a thin wrapper that provides tensors to the core refiner.

## Unified ASRR Contract

The core contract is:

```text
A_base = frozen_policy(obs, task, context)      # [B, H, D]
Delta_A = refiner(A_base, optional_context)    # [B, H, D]
A_refined = A_base + alpha * Delta_A
```

Where:

- `B`: batch size.
- `H`: predicted action horizon.
- `D`: action dimension.
- `alpha`: evaluation-time residual scale.
- `optional_context`: low-dimensional state, task id, action hidden states, Octo readout features, or a flat feature vector built by the method wrapper.

The core owns the residual module, conservative initialization, gating, action bounds, action masks, and shared supervised losses. Method-specific code owns environment setup, dataset/cache construction, camera preprocessing, language encoding, hidden-feature extraction, and action normalization.

## Latest Octo Refiner Design

Source implementation:

```text
third_party/octo/scripts/libero_residual_adapter.py
third_party/octo/scripts/train_libero_residual_adapter.py
```

Latest paper-facing object 30k three-refiner run:

```text
important_stage/octo_stage/octo_object_30k_3refiner_20260525_194452/OBJECT_30K_3REFINER_RESULTS.md
```

Octo uses a JAX/Flax flat-feature residual MLP. The Octo wrapper first samples an Octo action chunk, optionally extracts transformer token features, then constructs a flat feature vector.

Common dimensions in the LIBERO Octo experiments:

```text
action_horizon = 4
action_dim = 7
action_flat = 4 * 7 = 28
readout_action = 768
task_one_hot = 10
```

The three current refiner structures are:

| Structure | Feature mode | Feature vector | Typical adapter |
|---|---|---|---|
| `action_only` | `action` | `action_flat` | residual MLP, hidden `256,256` |
| `action_task` | `action_task` | `action_flat + task_one_hot` | residual MLP, hidden `256,256` |
| `readout_action_task` | `readout_action_task` | `action_flat + Octo readout_action + task_one_hot` | gated MLP, hidden `512,512` |

Other explored Octo internal modes:

| Mode | Extra information |
|---|---|
| `readout_action` | Octo action readout tokens |
| `readout_action_lang` | action readout + task/language tokens |
| `full_internal` | action readout + language + primary/wrist observation token pools |

Training target:

```text
Delta_A_target = clip(A_expert - A_octo, -max_correction, max_correction)
loss = MSE(Delta_A_pred, Delta_A_target)
```

Current Octo residual application:

```text
A_refined = A_octo + alpha * Delta_A_pred
```

Important Octo wrapper details that should remain outside `asrr_core`:

- physical-intelligence LIBERO loading.
- primary/wrist image transforms.
- Octo action statistics and unnormalization.
- `model.run_transformer(...)` and token pooling.
- LIBERO task text to suite-local task id mapping.

Core integration completed locally:

- Octo feature extraction stays in `third_party/octo/scripts`.
- `third_party/octo/scripts/libero_residual_adapter.py` now prefers `asrr_core.jax_refiners.JaxFlatFeatureResidualAdapter` and `JaxGatedFlatFeatureResidualAdapter` when the project root is visible.
- The old local Octo classes remain as fallback if `asrr_core` is unavailable.
- Octo-specific `make_adapter_features(...)` remains in the Octo wrapper because it depends on Octo token group names.

## Latest SmolVLA Refiner Design

Source implementation:

```text
vla_residual_adapter/train_smolvla_simple_asrr_adapter.py
vla_residual_adapter/eval_smolvla_libero_simple_asrr_no_video.py
vla_residual_adapter/model.py::SimpleExecutableResidualAdapter
```

Latest final simple-ASRR run:

```text
final_logs/smolvla_adapter_logs/simple_asrr_final_20260524_164320
```

Result summary:

```text
final_logs/smolvla_adapter_logs/simple_asrr_final_20260524_164320/results/SUMMARY.md
```

The latest useful SmolVLA structure intentionally removed the more complex prefix/action-trace/internal-feature variants. The final structure is:

```text
input = base_action[:, 0:5, :] + obs_context
obs_context = low-dimensional robot state, dim 8
output = delta for the first executable action only
delta[t > 0] = 0
```

Architecture:

```text
concat(base_action[0:5], obs_context)
  -> LayerNorm
  -> Linear(512) + GELU
  -> Linear(512) + GELU
  -> delta_head + gate_head
  -> tanh(delta_head) * sigmoid(gate_head) * max_delta
```

Bounds used in the final SmolVLA run:

```text
max_delta = [0.05, 0.05, 0.05, 0.015, 0.015, 0.015, 0.0]
gripper frozen
loss_dim_weights = [1, 1, 1, 1, 1, 1, 0]
```

Training:

```text
epochs = 1000
batch_size = 512
lr = 1e-4
delta_l2_weight = 1e-3
gate_loss_weight = 1e-3
```

Core integration completed locally:

```text
asrr_core/executable.py::ExecutableStepResidualAdapter
```

The new class preserves the old state-dict parameter names:

```text
net.*
delta_head.*
gate_head.*
max_delta
action_mask
```

The SmolVLA simple-ASRR train/eval scripts now import:

```text
from asrr_core import ExecutableStepResidualAdapter as SimpleExecutableResidualAdapter
```

## pi0.5 Refiner Information

Available local record:

```text
step10000_all_suites_alpha_0p5_to_4p0_summary.md
```

There is no full pi0.5 implementation directory in this project comparable to Octo or SmolVLA. The available result file indicates the pi0.5-style refiner is operationally closest to the simple executable-step ASRR formulation:

```text
base action sequence + lightweight state/context -> bounded residual
alpha sweep at eval time
paired base/refiner success accounting
```

The best summary in that file reports large suite-level gains at step 10000:

| Suite | Best alpha | Base | Refined |
|---|---:|---:|---:|
| `libero_spatial` | 3.0 | 84.4% | 96.0% |
| `libero_object` | 1.5 | 88.4% | 99.8% |
| `libero_goal` | 3.5 | 70.4% | 84.2% |
| `libero_10` | 2.0 | 68.6% | 84.6% |

Recommendation:

- Treat pi0.5 as another wrapper around `ExecutableStepResidualAdapter` unless its code later exposes useful VLA internal action features.
- Keep the same output metrics as SmolVLA: rescue, negative/regression, preserve, paired seeds, gate mean, delta mean.

## OpenVLA-OFT Refiner Design

Source implementation:

```text
openvla_oft_residual_adapter/model.py
openvla_oft_residual_adapter/train_adapter.py
openvla_oft_residual_adapter/eval_libero_adapter.py
papers/openvla_oft_suite_adapter_results_record_2026-05-24.md
```

Paper-facing OpenVLA-OFT stage used normalized action-space refinement:

```text
base_action_norm: [B, 8, 7]
action_context:   [B, 8, 4096]
obs_context:      [B, 8]
suite_task_id:    [B]
delta_action_norm:[B, 8, 7]
```

Main structures explored:

| Structure | Core idea |
|---|---|
| `lowrank_delta` | low-rank temporal DCT residual head |
| `sigmoid_gate` | dense residual with conservative learned gate |
| `bounded_delta_only` | direct bounded residual without gate |
| `mlp_mixer_gate` | mixer encoder with gated residual |

OpenVLA-OFT has richer per-action internal features than the simple SmolVLA/pi0.5 structure. The general PyTorch `ActionSequenceResidualAdapter` already covers the low-rank/gated/direct sequence-level cases when the wrapper supplies `action_context`. The `dimtoken` and `mlp_mixer_gate` variants are still method-specific extensions and can be promoted later if they remain paper-facing.

## ACT And Diffusion Policy Mapping

ACT and DP already follow the intended core contract.

ACT:

```text
base_action: [B, H, D]
context: qpos or camera-derived wrapper context
core: ActionSequenceResidualAdapter
```

DP:

```text
base_action: [B, n_action_steps, D]
context: flattened low-dimensional observation history
core: ActionSequenceResidualAdapter
optional DP wrapper: mode embedding / mixture of experts
```

The important point is that ACT, DP, and VLA all reduce to the same residual sequence correction. Only the wrapper-side context extractor changes.

## Current Local asrr_core Structure

Implemented files:

```text
asrr_core/
  __init__.py
  README.md
  losses.py
  models.py
  executable.py
  flat.py
  runtime.py
  jax_refiners.py
  ASRR_REFINER_INTEGRATION_NOTES_2026-05-28.md
```

Main exports:

```python
from asrr_core import (
    ActionSequenceResidualAdapter,
    ExecutableStepResidualAdapter,
    FlatFeatureResidualAdapter,
    supervised_asrr_loss,
    apply_residual,
    refine_with_adapter,
    count_trainable_parameters,
)
```

Optional Octo/JAX import:

```python
from asrr_core.jax_refiners import (
    JaxFlatFeatureResidualAdapter,
    JaxGatedFlatFeatureResidualAdapter,
)
```

## Recommended Open-Source Layout

Keep `asrr_core` small and dependency-light:

```text
asrr_core/
  models.py          # general sequence residual core
  executable.py      # first-executed-action refiner
  flat.py            # flat-feature chunk refiner
  losses.py          # shared supervised residual losses
  runtime.py         # apply_residual and wrapper output helpers
  jax_refiners.py    # optional Octo/Flax flat-feature refiner
```

Put policy-specific glue in examples, not the core:

```text
examples/
  act/
  diffusion_policy/
  smolvla/
  openvla_oft/
  octo/
  pi05/
```

Each example should implement the same three steps:

1. Cache or compute `base_action`.
2. Build the method-specific context tensor.
3. Train/evaluate the ASRR refiner with paired base/refined metrics.

## Migration Checklist

Immediate local state:

- SmolVLA simple-ASRR train/eval now use `asrr_core.ExecutableStepResidualAdapter`.
- Octo local flat-feature refiner code now prefers the equivalent optional JAX implementation in `asrr_core.jax_refiners`.
- Existing ACT/DP core usage remains valid.
- Existing OpenVLA-OFT code is not forcibly refactored because its final paper-facing implementation includes extra candidate structures; it should be migrated after deciding which structures remain in the public release.

Recommended next migration steps:

1. Add example wrapper scripts for SmolVLA and Octo that use a common cache schema:

```text
base_action
target_action
obs_context
action_context
flat_features
task_index
metadata
```

2. Add a small unit-test set:

```text
test_executable_step_shape_and_zero_tail
test_flat_feature_shape_and_zero_init
test_general_sequence_state_dict
test_apply_residual_alpha_and_mask
```

3. Keep old experiment logs and checkpoints out of the open-source package. Release only clean examples and a minimal README explaining how to connect a new policy.

## Caveats

- The default shell Python in this workspace does not include `torch`; PyTorch forward tests were run through the `smolvla` conda environment.
- Octo is JAX/Flax, so its full wrapper cannot import PyTorch-only modules. The optional JAX file is intentionally not imported by `asrr_core.__init__`.
- Current Octo experiment scripts still contain local adapter classes as fallback. Runtime import now prefers `asrr_core.jax_refiners` when available.
