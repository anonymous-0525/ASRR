# Diffusion Policy + ASRR Example

This example demonstrates the reusable ASRR wrapper logic and leaves Hydra,
Robomimic, PushT, and environment-runner details to a method-specific script.

## Interface

Diffusion Policy produces:

```text
base_action: [B, n_action_steps, D]
obs:         [B, n_obs_steps, C]
```

The wrapper flattens observation history when needed:

```text
obs_context = obs.reshape(B, -1)
```

and applies:

```text
delta_action = adapter(base_action, obs_context)
refined_action = base_action + alpha * delta_action
```

## Variants

| Variant | Meaning |
|---|---|
| `asrr_action` | action-only residual baseline |
| `asrr_obs_add` | state/observation-conditioned residual |
| `asrr_mode_embed` | GRU mode embedding from action chunk plus observation |
| `asrr_obs_mode` | mode embedding plus observation residual context |
| `asrr_mode_moe` | mixture of ASRR experts routed by mode context |

## Files

```text
asrr_dp_wrapper.py
```

The wrapper expects a DP-like base policy with:

```python
base_policy.predict_action({"obs": obs}) -> {"action": base_action}
```

## Example Training Pattern

An ASRR training script for Diffusion Policy should:

1. load a frozen DP checkpoint;
2. run the base policy over the dataset;
3. cache `obs_context`, `base_action`, and `target_action`;
4. train the ASRR adapter with `supervised_asrr_loss`;
5. evaluate with the original DP environment runner.

Example command pattern:

```bash
python train_dp_asrr.py \
  --base_checkpoint <frozen-dp-checkpoint> \
  --output_dir <asrr-dp-output-dir> \
  --asrr_variant asrr_obs_mode \
  --epochs 1000 \
  --batch_size 512 \
  --hidden_dim 256 \
  --lr 1e-4

python eval_dp_asrr.py \
  --base_checkpoint <frozen-dp-checkpoint> \
  --adapter_checkpoint <asrr-dp-output-dir>/adapter_best.pt \
  --output_dir <asrr-dp-eval-dir> \
  --residual_alpha 0.5
```

## What To Keep In A Public Release

Keep:

- this wrapper;
- cache schema documentation;
- small tensor smoke tests;
- command templates.

Do not keep:

- Robomimic datasets;
- DP checkpoints;
- generated videos;
- long run logs.
