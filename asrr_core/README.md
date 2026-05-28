# ASRR Core

This package contains policy-agnostic components for Action Sequence Residual
Refinement.

Core contract:

```text
base_action: [B, H, D]
optional context: qpos / obs / mode / VLA action hidden states
adapter(base_action, context) -> delta_action: [B, H, D]
refined_action = base_action + alpha * delta_action
```

The package should stay independent of ACT, Diffusion Policy, VQ-BeT, SmolVLA,
OpenVLA-OFT, Octo, and pi0/pi0.5 code.  Policy-specific wrappers should live in
their own method directories and import this package.

