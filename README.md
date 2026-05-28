# ASRR

**Action Sequence Residual Refinement for Frozen Vision-Language-Action Policy Adaptation**

ASRR is an offline adaptation method for robot policies that output action
chunks. A frozen base policy first proposes an action sequence, and a lightweight
Action Sequence Refiner predicts a residual correction in the policy-native
action space:

```text
A_refined = A_base + alpha * M * R_theta(A_base, z)
```

The repository currently publishes a partial snapshot of the policy-agnostic
core package used by the project website. The core is still being adjusted, so
this snapshot should be treated as an early public interface rather than the
final code release. Full training wrappers, experiment scripts, and cleaned
result artifacts will be added as the code release matures.

## Links

- Project page: https://anonymous-0525.github.io/ASRR/
- Core package: [`asrr_core/`](asrr_core/)
- Website source: [`docs/`](docs/)

## Public Code Scope

`asrr_core` contains the reusable parts of ASRR:

- `ActionSequenceResidualAdapter`: a small residual refiner over action chunks.
- `supervised_asrr_loss`: a shared supervised loss for refined action sequences.
- Fusion modes for action-only, state-conditioned, FiLM-conditioned, and
  action-context-conditioned refinement.
- Residual heads for dense, bounded, gated, and low-rank corrections.

The package intentionally stays independent of ACT, Diffusion Policy, OpenVLA,
Octo, SmolVLA, pi0/pi0.5, and other policy-specific code. Wrappers should adapt
those policies into the common tensor contract:

```text
base_action:    [B, H, D]
state_context:  [B, C_s]       optional
action_context: [B, H, C_a]    optional
task_index:     [B]            optional
delta_action:   [B, H, D]
```

## Minimal Example

```python
import torch

from asrr_core import ActionSequenceResidualAdapter, supervised_asrr_loss

adapter = ActionSequenceResidualAdapter(
    action_dim=7,
    horizon=16,
    hidden_dim=256,
    fusion_mode="action_only",
    head_type="bounded_dense",
    max_delta=0.05,
    freeze_last_action_dim=True,
)

base_action = torch.zeros(8, 16, 7)
target_action = torch.randn(8, 16, 7)

delta = adapter(base_action)
metrics = supervised_asrr_loss(
    base_action=base_action,
    delta_action=delta,
    target_action=target_action,
    loss_type="mse",
)
metrics["loss"].backward()
```

## Citation

Citation information will be added after release.
