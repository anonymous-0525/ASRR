# Policy Integration Examples

The examples expose a small tensor boundary between an external frozen policy
and `asrr_core`:

```text
base runner -> action proposal and optional context -> ASRR -> refined proposal
```

Only policy-specific preprocessing and execution remain outside ASRR.

| Directory | Expected inputs |
|---|---|
| `act/` | Action chunk and optional robot state. |
| `diffusion_policy/` | Denoised action chunk, observation history, and optional mode summary. |
| `pi05/` | Continuous action proposal with optional global or step-aligned context. |
| `openvla_oft/` | Normalized action chunk and action-generation features. |

`train_cached_refiner.py` is a runnable policy-independent trainer for cached
tensors. The wrappers do not contain checkpoints, datasets, environment code,
or machine-specific paths.
