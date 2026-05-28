# pi0.5 + ASRR Example

This example provides a pi0.5-style executable-step ASRR wrapper.  It is
intended for pi0.5 runners that predict an action chunk and execute actions in
a receding-horizon loop:

```text
base action sequence + lightweight state/context -> bounded residual
alpha sweep at evaluation time
paired base/refined success accounting
```

## Interface

Recommended refiner:

```python
from asrr_core import ExecutableStepResidualAdapter
```

Expected tensors:

```text
base_action:   [B, H, D]
target_action: [B, H, D]
obs_context:   optional [B, C]
```

Execution:

```text
delta[:, 0] is applied
delta[:, 1:] remains zero
```

## Files

```text
asrr_pi05_wrapper.py
```

The wrapper is intentionally generic.  A real pi0.5 runner must provide model
loading, observation preprocessing, action normalization, and environment action
conversion.

## Example Alpha Sweep Report Format

| Suite | Best alpha | Base success | Refined success |
|---|---:|---:|---:|
| `suite_a` | 1.0 | 70.0% | 78.0% |
| `suite_b` | 2.0 | 65.0% | 74.0% |
