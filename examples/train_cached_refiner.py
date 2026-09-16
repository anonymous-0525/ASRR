"""Minimal offline training entry point for an ASRR tensor cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from asrr_core import (
    ActionSequenceResidualAdapter,
    CachedActionDataset,
    supervised_asrr_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--encoder-type", choices=("mlp", "transformer"), default="transformer")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-delta", type=float, default=0.05)
    parser.add_argument("--loss", choices=("l1", "mse", "huber"), default="mse")
    parser.add_argument("--delta-l2-weight", type=float, default=0.01)
    parser.add_argument("--smooth-l2-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CachedActionDataset.from_file(args.cache)
    tensors = dataset.tensors
    _, horizon, action_dim = tensors["base_action"].shape
    state_dim = int(tensors["state_context"].shape[-1]) if "state_context" in tensors else 0
    action_context_dim = int(tensors["action_context"].shape[-1]) if "action_context" in tensors else 0

    if state_dim and action_context_dim:
        fusion_mode = "state_action_context_add"
    elif state_dim:
        fusion_mode = "state_add"
    elif action_context_dim:
        fusion_mode = "action_context_add"
    else:
        fusion_mode = "action_only"

    task_count = 0
    if "task_index" in tensors:
        task_count = int(tensors["task_index"].max().item()) + 1
    action_mask = dataset.action_mask
    if action_mask is not None:
        if action_mask.ndim != 1:
            raise ValueError("action_mask must be a cache-level tensor with shape [D]")
        action_mask = action_mask.tolist()

    model_config = {
        "action_dim": action_dim,
        "horizon": horizon,
        "state_context_dim": state_dim,
        "action_context_dim": action_context_dim,
        "num_task_embeddings": task_count,
        "hidden_dim": args.hidden_dim,
        "fusion_mode": fusion_mode,
        "encoder_type": args.encoder_type,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "head_type": "bounded_dense",
        "max_delta": args.max_delta,
        "action_mask": action_mask,
    }
    refiner = ActionSequenceResidualAdapter(**model_config).to(args.device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.learning_rate)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    refiner.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in loader:
            base = batch["base_action"].to(args.device)
            target = batch["target_action"].to(args.device)
            kwargs = {}
            for key in ("state_context", "action_context", "task_index"):
                if key in batch:
                    kwargs[key] = batch[key].to(args.device)
            delta = refiner(base, **kwargs)
            metrics = supervised_asrr_loss(
                base_action=base,
                delta_action=delta,
                target_action=target,
                is_pad=batch.get("is_pad", None).to(args.device) if "is_pad" in batch else None,
                alpha=args.alpha,
                loss_type=args.loss,
                delta_l2_weight=args.delta_l2_weight,
                smooth_l2_weight=args.smooth_l2_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            metrics["loss"].backward()
            optimizer.step()
            running_loss += float(metrics["loss"].detach()) * base.shape[0]
        print(f"epoch={epoch + 1} loss={running_loss / len(dataset):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": model_config, "state_dict": refiner.state_dict()}, args.output)


if __name__ == "__main__":
    main()
