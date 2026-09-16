from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Union

import torch
from torch.utils.data import Dataset


class CachedActionDataset(Dataset):
    """Tensor dataset for offline ASRR fitting.

    A cache must contain ``base_action`` and ``target_action`` tensors with shape
    ``[N, H, D]``. Optional per-example tensors such as ``state_context``,
    ``action_context``, ``task_index``, and ``is_pad`` are returned unchanged.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor]):
        if "base_action" not in tensors or "target_action" not in tensors:
            raise ValueError("cache requires base_action and target_action")

        self.action_mask = None
        if "action_mask" in tensors:
            self.action_mask = tensors["action_mask"].detach().cpu()
        self.tensors: Dict[str, torch.Tensor] = {
            key: value.detach().cpu()
            for key, value in tensors.items()
            if key != "action_mask"
        }
        base = self.tensors["base_action"]
        target = self.tensors["target_action"]
        if base.ndim != 3 or target.ndim != 3:
            raise ValueError("base_action and target_action must be [N,H,D]")
        if base.shape != target.shape:
            raise ValueError("base_action and target_action must have the same shape")

        self.length = int(base.shape[0])
        if self.length == 0:
            raise ValueError("cache must contain at least one example")
        if self.action_mask is not None:
            if self.action_mask.ndim != 1 or self.action_mask.shape[0] != base.shape[-1]:
                raise ValueError("action_mask must have shape [D]")
        for key, value in self.tensors.items():
            if value.ndim == 0 or value.shape[0] != self.length:
                raise ValueError(f"cache tensor {key!r} must have leading dimension N={self.length}")

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "CachedActionDataset":
        payload = torch.load(Path(path), map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError("cache file must contain a mapping of tensor names to tensors")
        if not all(isinstance(value, torch.Tensor) for value in payload.values()):
            raise ValueError("all cache values must be torch tensors")
        return cls(payload)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}
