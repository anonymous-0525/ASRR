# ASRR: Offline Action Sequence Residual Refinement for Robot Policy Adaptation

ASRR adapts a frozen robot policy by editing the action sequence it already
predicts. A compact refiner is trained offline from cached policy outputs and
demonstration actions, then composed with the base proposal before the policy's
native execution rule is applied.

- Project page: <https://anonymous-0525.github.io/ASRR/>
- Anonymous paper: <https://anonymous-0525.github.io/ASRR/static/files/asrr_paper.pdf>

## Method

```text
base_action = frozen_policy(observation, task)       # [B, H, D]
delta_action = refiner(base_action, optional_context)
refined_action = base_action + alpha * delta_action  # [B, H, D]
```

The public package contains:

- a policy-agnostic PyTorch sequence refiner;
- bounded and coordinate-masked residual composition;
- the supervised offline ASRR objective;
- a tensor-cache dataset and minimal training entry point;
- thin adapters for ACT, Diffusion Policy, pi0.5, and OpenVLA-OFT.

External policy repositories, checkpoints, datasets, and robot environments are
not copied into this repository.

## Installation

Python 3.9 or newer and PyTorch 2.0 or newer are required.

```bash
git clone https://github.com/anonymous-0525/ASRR.git
cd ASRR
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests
```

## Quick Start

### Action-only refinement

```python
import torch
from asrr_core import ActionSequenceResidualAdapter, apply_residual

base_action = torch.randn(2, 16, 7)

refiner = ActionSequenceResidualAdapter(
    action_dim=7,
    horizon=16,
    hidden_dim=256,
    fusion_mode="action_only",
    encoder_type="transformer",
    head_type="bounded_dense",
    max_delta=0.05,
)

delta = refiner(base_action)
refined_action = apply_residual(base_action, delta, alpha=1.0)
```

The residual head is zero-initialized, so the composed controller initially
matches the frozen base policy.

### Context-conditioned and masked refinement

```python
state = torch.randn(2, 14)
editable = [1, 1, 1, 1, 1, 1, 0]

refiner = ActionSequenceResidualAdapter(
    action_dim=7,
    horizon=16,
    state_context_dim=14,
    hidden_dim=256,
    fusion_mode="state_add",
    encoder_type="transformer",
    head_type="bounded_dense",
    max_delta=[0.04, 0.04, 0.04, 0.015, 0.015, 0.015, 0.0],
    action_mask=editable,
)

delta = refiner(base_action, state_context=state)
refined_action = apply_residual(base_action, delta, alpha=1.0)
```

## Offline Cache Training

The generic trainer consumes a `.pt` mapping with the following tensors:

| Key | Shape | Required |
|---|---|---|
| `base_action` | `[N, H, D]` | yes |
| `target_action` | `[N, H, D]` | yes |
| `state_context` | `[N, C]` | no |
| `action_context` | `[N, H, C_step]` | no |
| `task_index` | `[N]` | no |
| `is_pad` | `[N, H]` | no |
| `action_mask` | `[D]` | no |

```bash
python examples/train_cached_refiner.py \
  --cache path/to/cache.pt \
  --output outputs/refiner.pt \
  --encoder-type transformer \
  --epochs 100
```

Cache generation stays with the base-policy runner because observation
preprocessing, action normalization, feature extraction, and demonstration
alignment are policy-specific. The resulting cache contains only tensors and
can be reused across compatible refiner configurations.

## Policy Integrations

| Directory | Public interface |
|---|---|
| `examples/act/` | Frozen ACT action chunks, optionally conditioned on `qpos`. |
| `examples/diffusion_policy/` | Action-only, observation, and trajectory-mode variants. |
| `examples/pi05/` | Policy-native sequence refinement with optional global and step context. |
| `examples/openvla_oft/` | Normalized action chunks and action-generation context. |

Each adapter is intentionally thin. It accepts tensors from an installed base
policy and returns refined tensors; it does not assume a private checkout or an
absolute path.

## Repository Layout

```text
asrr_core/
  cache.py       # offline tensor-cache dataset
  models.py      # action-only and context-conditioned sequence refiner
  losses.py      # demonstration supervision and residual regularization
  runtime.py     # residual composition and standard output helpers

examples/
  act/
  diffusion_policy/
  pi05/
  openvla_oft/
  train_cached_refiner.py

docs/            # anonymous project page served by GitHub Pages
tests/           # core contracts and public-site checks
```

## Scope

A fresh clone can install the package, train from a prepared tensor cache, and
run all tensor-level tests. Reproducing policy rollouts additionally requires
the corresponding public base-policy implementation, benchmark, checkpoint,
and normalization statistics.

## Citation

```bibtex
@misc{asrr2026,
  title  = {ASRR: Offline Action Sequence Residual Refinement
            for Robot Policy Adaptation},
  author = {Anonymous},
  year   = {2026},
  note   = {Anonymous ICRA 2027 submission}
}
```
