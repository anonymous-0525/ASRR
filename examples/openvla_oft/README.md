# OpenVLA-OFT + ASRR

OpenVLA-OFT exposes normalized action chunks and action-generation context. The
wrapper expects a callable that returns:

```text
base_action_norm: [B, H, D]
action_context:   [B, H, C] or [B, H, D, C]
obs_context:      optional [B, C_obs]
task_index:       optional [B]
```

`OpenVLAOFTResidualAdapter` implements the context-conditioned structures used
by the paper experiments. `build_openvla_oft_core_refiner` provides the same
interface through the shared `ActionSequenceResidualAdapter` when checkpoint
compatibility with those specialized refiners is not needed.

The wrapper clips the normalized refined proposal and optionally calls an
external unnormalization function. Model loading, image/language preprocessing,
action statistics, and environment execution stay in the OpenVLA-OFT runner.
