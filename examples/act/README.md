# ACT + ASRR

`ASRRACTWrapper` freezes an ACT-like policy and refines its `[B, H, D]` action
chunk. The refiner may use the chunk alone or condition on `qpos`.

```python
from examples.act.asrr_act_wrapper import ASRRACTWrapper, build_act_asrr_refiner

refiner = build_act_asrr_refiner(
    action_dim=14,
    horizon=80,
    variant="action_only",
    hidden_dim=256,
)
policy = ASRRACTWrapper(base_policy, refiner, alpha=1.0)
```

The base policy must return an action tensor with shape `[B, H, D]`. Dataset,
camera preprocessing, temporal aggregation, and environment rollout remain in
the ACT runner.
