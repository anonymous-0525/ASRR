# SmolVLA + ASRR Example

This example captures the final simple-ASRR design used for SmolVLA in this
repository.

## Interface

SmolVLA predicts a long action chunk, but evaluation executes one action at a
time.  The final useful ASRR structure only refines the first executable action.

```text
base_action: [B, 50, 7]
obs_context: [B, 8]
delta:       [B, 50, 7]
delta[:, 0]  is learned
delta[:, 1:] is zero
```

Recommended refiner:

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

## Files

```text
asrr_smolvla_wrapper.py
```

The wrapper expects a base policy with:

```python
base_policy.predict_action_chunk(batch) -> base_action
```

It does not import LeRobot.  A LeRobot-specific runner should own policy
loading, env creation, preprocessors, postprocessors, and seed splits.
