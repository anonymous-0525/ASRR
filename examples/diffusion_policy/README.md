# Diffusion Policy + ASRR

`ASRRDiffusionPolicyWrapper` expects a frozen Diffusion Policy implementation
whose `predict_action` method returns an action chunk. Supported refiners are:

| Variant | Conditioning |
|---|---|
| `asrr_action` | Action proposal only. |
| `asrr_obs_add` | Flattened low-dimensional observation history. |
| `asrr_mode_embed` | Learned summary of proposal and observation. |
| `asrr_obs_mode` | Mode summary with an observation residual path. |

The wrapper owns no Robomimic or environment dependency. The external runner
produces aligned `base_action`, `obs_context`, and demonstration targets for
offline refiner fitting.
