# Octo + ASRR Example

Octo is JAX/Flax-based, so this example uses the optional JAX refiners in:

```text
asrr_core/jax_refiners.py
```

This example shows the clean boundary between Octo and ASRR.

## Interface

Octo wrapper responsibilities:

```text
1. load OctoModel
2. preprocess observations and task text
3. sample base_action from Octo
4. optionally run Octo transformer and pool internal features
5. build a flat ASRR feature vector
6. call ASRR JAX refiner
7. unnormalize refined action for the environment
```

ASRR core responsibilities:

```text
features [B, C] -> delta_action [B, 4, 7]
refined_action = base_action + alpha * delta_action
```

## Feature Modes

| Mode | Feature vector |
|---|---|
| `action` | `flatten(base_action)` |
| `action_task` | `flatten(base_action) + task_one_hot` |
| `readout_action` | `flatten(base_action) + readout_action` |
| `readout_action_task` | `flatten(base_action) + readout_action + task_one_hot` |

Additional feature modes such as `readout_action_lang` and `full_internal` can
be implemented in the Octo wrapper when the corresponding Octo token groups are
available.

## Files

```text
asrr_octo_flat_wrapper.py
```

The helper does not import Octo.  It expects the Octo-side wrapper to provide:

```text
base_action
task_ids
internal_features
```

## Minimal Flow

```python
from examples.octo.asrr_octo_flat_wrapper import (
    build_octo_flat_features,
    make_octo_flat_refiner,
    apply_octo_flat_refiner,
)

refiner = make_octo_flat_refiner(config)
params = refiner.init(rng, dummy_features)

features = build_octo_flat_features(
    base_action,
    feature_mode="readout_action_task",
    task_ids=task_ids,
    internal_features={"readout_action": readout_action},
)
delta, refined = apply_octo_flat_refiner(refiner, params, base_action, features, alpha=0.5)
```

## Required External Setup

To run real Octo experiments, users still need:

- Octo installed;
- Octo checkpoint;
- LIBERO or another rollout environment;
- the correct action statistics;
- method-specific image and task preprocessing.
