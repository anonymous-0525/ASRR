# ACT + ASRR Example

This example demonstrates the ASRR boundary for ACT-style policies without
copying a full ACT implementation.

## Interface

ACT produces an action chunk:

```text
base_action: [B, H, D]
qpos:        [B, D] or [B, C_qpos]
```

ASRR refines the action chunk:

```text
delta_action = refiner(base_action, qpos)
refined_action = base_action + alpha * delta_action
```

Recommended variants:

| Variant | ASRR core setting |
|---|---|
| `action_only` | `fusion_mode="action_only"` |
| `qpos_add` | `fusion_mode="state_add"` |
| `qpos_concat` | `fusion_mode="state_concat"` |
| `film` | `fusion_mode="film"` |

## Files

```text
asrr_act_wrapper.py
```

The wrapper expects a frozen ACT-like base policy that returns `[B,H,D]` action
chunks.  It does not depend on a specific ACT repository layout.

## Example Training Command Pattern

An ACT training script with ASRR support can expose arguments like:

```bash
python train_act_with_asrr.py \
  --policy_class ASRRACT \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir <asrr-act-output-dir> \
  --base_ckpt_path <frozen-act-checkpoint> \
  --camera_names top_left_right \
  --base_camera_names top \
  --adapter_variant qpos_add \
  --chunk_size 80 \
  --refine_horizon 80 \
  --adapter_hidden_dim 256 \
  --adapter_head_type dense \
  --num_epochs 1000 \
  --batch_size 8 \
  --seed 0 \
  --lr 1e-4
```

Use method-appropriate task names, dataset paths, camera names, and checkpoint
paths.  The base ACT policy should remain frozen; only ASRR parameters train.

## What To Keep In A Public Release

Keep:

- the wrapper in this example;
- a short command template;
- a small synthetic tensor smoke test.

Do not keep:

- ACT checkpoints;
- generated videos;
- long run directories;
- machine-specific dataset paths.
