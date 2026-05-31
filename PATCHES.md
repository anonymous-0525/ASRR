# Method-Side Integration and Patch Notes

ASRR is designed so that external policy repositories import `asrr_core`; the
core package does not import ACT, Diffusion Policy, Octo, SmolVLA, pi0.5, or
OpenVLA-OFT.  Any method-specific edits, monkey patches, or wrappers should be
kept outside `asrr_core` and documented here.

## Integration Rule

Production integrations should follow this boundary:

```text
external method runner
  -> base action and optional context tensors
  -> asrr_core or examples/<method> wrapper
  -> refined action tensors
  -> external method action conversion and environment step
```

The ASRR repository may include lightweight examples and compatibility wrappers,
but not full external method source trees, checkpoints, datasets, rollout logs,
or machine-specific paths.

## Patch Ledger

| Method | Source-code patch required? | Public ASRR files | Method-side information passed to ASRR | Production notes |
|---|---|---|---|---|
| ACT | Usually no hard patch; wrapper around frozen policy is sufficient | `examples/act/` | `base_action`, optional robot state such as `qpos` | External ACT code owns dataloading, checkpoint loading, and env rollout |
| Diffusion Policy | Usually no hard patch; wrapper around frozen policy is sufficient | `examples/diffusion_policy/` | `base_action`, optional observation context and mode context | External DP code owns normalization, observation history, and rollout |
| Octo | Runner-level integration required; source patch depends on how internal readouts are exposed | `examples/octo/`, `asrr_core/jax_refiners.py` | `base_action`, task ids, optional readout/internal features | External Octo stack owns fine-tuning, token/readout extraction, action statistics, and rollout |
| SmolVLA | Runner-level integration required; source patch may be needed if hidden/prefix features are not public | `examples/smolvla/` | `base_action` prefix, optional low-dimensional state/context | External SmolVLA/LeRobot runner owns cache generation, preprocessing, postprocessing, and no-video evaluation |
| pi0.5 | Runner-level integration required | `examples/pi05/` | `base_action`, optional context | Public package currently provides the contract, not a full pi0.5 hidden-state ASRR runner |
| OpenVLA-OFT | Runner-level integration required; source patch depends on access to action-head context | `examples/openvla_oft/` | normalized `base_action`, `action_context`, optional `obs_context`, `task_index` | External OpenVLA-OFT runner owns processor/action-head calls, action stats, task mapping, queueing, and rollout |

## Information That Must Be Captured

When adding a production integration, document:

- external repository name and tested commit or release;
- environment name and major dependency versions;
- whether any external source files were modified;
- exact function or hook used to extract hidden features;
- tensor shapes and dtypes passed to ASRR;
- action normalization and clipping convention;
- action horizon and executed horizon;
- cache schema;
- evaluation seed policy;
- paired metric definition.

## No-Patch Preferred Pattern

Prefer a no-patch wrapper when the external policy already exposes the required
outputs:

```python
def predict_base_and_context(observation, task):
    base_action = frozen_policy.predict_action(observation, task)
    context = build_context_from_public_outputs(observation, task, base_action)
    return {
        "base_action": base_action,
        "context": context,
    }
```

Use this pattern for ACT and Diffusion Policy where the action chunk and
low-dimensional state are normally public outputs.

## Runner-Level Hook Pattern

Use a runner-level hook when the method exposes the base action but needs
additional intermediate tensors:

```python
def predict_base_and_context(observation, task):
    outputs = frozen_policy.forward_with_intermediates(observation, task)
    return {
        "base_action": outputs["action"],
        "action_context": outputs["action_context"],
        "obs_context": outputs.get("obs_context"),
        "task_index": task_to_index(task),
    }
```

This is the typical pattern for OpenVLA-OFT, SmolVLA, Octo, and pi0.5 style
integrations.  If the method does not expose `forward_with_intermediates`, add a
small, documented method-side hook rather than modifying `asrr_core`.

## What Not To Change

Method integrations must not change:

- ASRR metric definitions;
- dataset labels or task success criteria;
- completed experiment results;
- raw datasets;
- base policy checkpoints;
- `asrr_core` behavior only to satisfy one method.

If a method requires a methodological choice that changes the experiment target
or metric definition, record the issue in that method's experiment log and stop
instead of silently changing the comparison.

## Release Checklist

Before publishing a new method integration:

1. Add or update the corresponding `examples/<method>/README.md`.
2. Add tensor-shape assertions to the wrapper.
3. Document every method-side hook in this file.
4. Add a tensor-level smoke test when the integration can be tested without the
   external method repository.
5. Confirm that no local paths, checkpoints, datasets, or long logs are
   committed.
