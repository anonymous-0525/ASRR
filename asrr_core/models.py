from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


TensorOrNone = Optional[torch.Tensor]


class ActionSequenceResidualAdapter(nn.Module):
    """Policy-agnostic residual adapter over action chunks.

    The module consumes a base action sequence and optional context, then returns
    a same-shape residual.  It is intentionally independent of ACT, DP, VQ-BeT,
    and VLA wrappers.  Policy-specific code should only provide tensors.

    Input conventions:
        base_action:    [B, H, D]
        state_context:  [B, C_s]       e.g. qpos, lowdim obs, pooled obs
        action_context: [B, H, C_a]    e.g. VLA action hidden states
        task_index:     [B]            optional learned task embedding
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        state_context_dim: int = 0,
        action_context_dim: int = 0,
        num_task_embeddings: int = 0,
        hidden_dim: int = 256,
        fusion_mode: str = "action_only",
        encoder_type: str = "mlp",
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        head_type: str = "dense",
        max_delta: Union[None, float, Sequence[float]] = None,
        freeze_last_action_dim: bool = False,
        lowrank_num_basis: int = 4,
        gate_bias_init: float = -5.0,
    ):
        super().__init__()
        valid_fusion_modes = {
            "action_only",
            "state_add",
            "state_concat",
            "film",
            "action_context_add",
            "state_action_context_add",
        }
        valid_encoder_types = {"mlp", "transformer"}
        valid_head_types = {"dense", "bounded_dense", "sigmoid_gate", "lowrank_delta"}
        if fusion_mode not in valid_fusion_modes:
            raise ValueError(f"Unsupported fusion_mode={fusion_mode}")
        if encoder_type not in valid_encoder_types:
            raise ValueError(f"Unsupported encoder_type={encoder_type}")
        if head_type not in valid_head_types:
            raise ValueError(f"Unsupported head_type={head_type}")
        if fusion_mode in {"state_add", "state_concat", "film", "state_action_context_add"} and state_context_dim <= 0:
            raise ValueError(f"state_context_dim must be positive for {fusion_mode}")
        if fusion_mode in {"action_context_add", "state_action_context_add"} and action_context_dim <= 0:
            raise ValueError(f"action_context_dim must be positive for {fusion_mode}")
        if encoder_type == "transformer" and hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads for transformer encoder")
        if lowrank_num_basis <= 0:
            raise ValueError("lowrank_num_basis must be positive")

        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.state_context_dim = int(state_context_dim)
        self.action_context_dim = int(action_context_dim)
        self.num_task_embeddings = int(num_task_embeddings)
        self.hidden_dim = int(hidden_dim)
        self.fusion_mode = str(fusion_mode)
        self.encoder_type = str(encoder_type)
        self.head_type = str(head_type)
        self.lowrank_num_basis = int(lowrank_num_basis)

        self.base_action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.pos_embed = nn.Parameter(torch.zeros(self.horizon, self.hidden_dim))

        self.state_encoder = None
        if self.uses_state:
            self.state_encoder = nn.Sequential(
                nn.Linear(self.state_context_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
            )

        self.action_context_encoder = None
        if self.uses_action_context:
            self.action_context_encoder = nn.Linear(self.action_context_dim, self.hidden_dim)

        self.task_embedding = None
        if self.num_task_embeddings > 0:
            self.task_embedding = nn.Embedding(self.num_task_embeddings, self.hidden_dim)

        self.film_encoder = None
        self.concat_encoder = None
        if self.fusion_mode == "film":
            self.film_encoder = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            )
        elif self.fusion_mode == "state_concat":
            self.concat_encoder = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(inplace=True),
            )

        if self.encoder_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.final_norm = nn.LayerNorm(self.hidden_dim)
        else:
            blocks = []
            for _ in range(max(1, int(num_layers))):
                blocks.extend(
                    [
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                    ]
                )
            self.sequence_encoder = nn.Sequential(*blocks)
            self.final_norm = nn.Identity()

        self.delta_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.gate_head = nn.Linear(self.hidden_dim, 1)
        self.lowrank_coeff_head = nn.Linear(self.hidden_dim, self.lowrank_num_basis * self.action_dim)

        if max_delta is None:
            max_delta_tensor = torch.ones(self.action_dim, dtype=torch.float32)
            self.has_delta_bound = False
        elif isinstance(max_delta, (list, tuple)):
            if len(max_delta) != self.action_dim:
                raise ValueError("max_delta sequence must match action_dim")
            max_delta_tensor = torch.tensor(max_delta, dtype=torch.float32)
            self.has_delta_bound = True
        else:
            max_delta_tensor = torch.full((self.action_dim,), float(max_delta), dtype=torch.float32)
            self.has_delta_bound = True
        self.register_buffer("max_delta", max_delta_tensor)

        action_mask = torch.ones(self.action_dim, dtype=torch.float32)
        if freeze_last_action_dim and self.action_dim > 0:
            action_mask[-1] = 0.0
        self.register_buffer("action_mask", action_mask)
        self.register_buffer("lowrank_basis", self._make_dct_basis(self.horizon, self.lowrank_num_basis))

        self.reset_parameters(gate_bias_init=gate_bias_init)
        self._freeze_inactive_heads()

    @property
    def uses_state(self) -> bool:
        return self.fusion_mode in {"state_add", "state_concat", "film", "state_action_context_add"}

    @property
    def uses_action_context(self) -> bool:
        return self.fusion_mode in {"action_context_add", "state_action_context_add"}

    @staticmethod
    def _make_dct_basis(horizon: int, num_basis: int) -> torch.Tensor:
        steps = torch.arange(horizon, dtype=torch.float32)
        basis = []
        for k in range(num_basis):
            if k == 0:
                row = torch.ones(horizon, dtype=torch.float32)
            else:
                row = torch.cos(math.pi * (steps + 0.5) * float(k) / float(horizon))
            basis.append(row / row.abs().max().clamp_min(1e-6))
        return torch.stack(basis, dim=0)

    def reset_parameters(self, gate_bias_init: float = -5.0) -> None:
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.lowrank_coeff_head.weight)
        nn.init.zeros_(self.lowrank_coeff_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias_init))
        if self.task_embedding is not None:
            nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.02)

    def _set_module_requires_grad(self, module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)

    def _freeze_inactive_heads(self) -> None:
        """Keep trainable-parameter counts aligned with the selected head."""
        self._set_module_requires_grad(self.delta_head, self.head_type in {"dense", "bounded_dense", "sigmoid_gate"})
        self._set_module_requires_grad(self.gate_head, self.head_type == "sigmoid_gate")
        self._set_module_requires_grad(self.lowrank_coeff_head, self.head_type == "lowrank_delta")

    def _encode_tokens(
        self,
        base_action: torch.Tensor,
        state_context: TensorOrNone,
        action_context: TensorOrNone,
        task_index: TensorOrNone,
    ) -> torch.Tensor:
        if base_action.ndim != 3:
            raise ValueError(f"base_action must be [B,H,D], got {tuple(base_action.shape)}")
        batch_size, horizon, action_dim = base_action.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_dim}")
        if horizon > self.horizon:
            raise ValueError(f"Adapter horizon={self.horizon}, got input horizon={horizon}")

        tokens = self.base_action_encoder(base_action)
        tokens = tokens + self.pos_embed[:horizon].unsqueeze(0)

        state_tokens = None
        if self.uses_state:
            if state_context is None:
                raise ValueError(f"state_context is required for fusion_mode={self.fusion_mode}")
            if state_context.shape[-1] != self.state_context_dim:
                raise ValueError(f"Expected state_context_dim={self.state_context_dim}, got {state_context.shape[-1]}")
            state_tokens = self.state_encoder(state_context)

        if self.fusion_mode == "state_add":
            tokens = tokens + state_tokens.unsqueeze(1)
        elif self.fusion_mode == "state_concat":
            expanded = state_tokens.unsqueeze(1).expand(-1, horizon, -1)
            tokens = self.concat_encoder(torch.cat([tokens, expanded], dim=-1))
        elif self.fusion_mode == "film":
            gamma, beta = self.film_encoder(state_tokens).chunk(2, dim=-1)
            tokens = tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        if self.uses_action_context:
            if action_context is None:
                raise ValueError(f"action_context is required for fusion_mode={self.fusion_mode}")
            if action_context.shape[:2] != (batch_size, horizon):
                raise ValueError("action_context must share batch and horizon with base_action")
            if action_context.shape[-1] != self.action_context_dim:
                raise ValueError(f"Expected action_context_dim={self.action_context_dim}, got {action_context.shape[-1]}")
            tokens = tokens + self.action_context_encoder(action_context)
            if self.fusion_mode == "state_action_context_add":
                tokens = tokens + state_tokens.unsqueeze(1)

        if self.task_embedding is not None:
            if task_index is None:
                raise ValueError("task_index is required when num_task_embeddings > 0")
            task_index = task_index.reshape(-1).to(device=base_action.device, dtype=torch.long)
            if task_index.shape[0] != batch_size:
                raise ValueError(f"Expected task_index batch={batch_size}, got {task_index.shape[0]}")
            tokens = tokens + self.task_embedding(task_index).unsqueeze(1)

        return tokens

    def forward(
        self,
        base_action: torch.Tensor,
        state_context: TensorOrNone = None,
        action_context: TensorOrNone = None,
        task_index: TensorOrNone = None,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        tokens = self._encode_tokens(base_action, state_context, action_context, task_index)
        hidden = self.sequence_encoder(tokens)
        hidden = self.final_norm(hidden)
        batch_size, horizon, _ = hidden.shape

        if self.head_type == "lowrank_delta":
            pooled = hidden.mean(dim=1)
            coeff = self.lowrank_coeff_head(pooled).view(batch_size, self.lowrank_num_basis, self.action_dim)
            raw = torch.einsum("bkd,kh->bhd", coeff, self.lowrank_basis[:, :horizon])
        else:
            raw = self.delta_head(hidden)

        if self.head_type in {"bounded_dense", "sigmoid_gate", "lowrank_delta"} or self.has_delta_bound:
            raw_delta = torch.tanh(raw) * self.max_delta.view(1, 1, -1)
        else:
            raw_delta = raw

        if self.head_type == "sigmoid_gate":
            gate = torch.sigmoid(self.gate_head(hidden))
        else:
            gate = torch.ones(batch_size, horizon, 1, dtype=base_action.dtype, device=base_action.device)

        delta = raw_delta * gate * self.action_mask.view(1, 1, -1)
        info = {
            "raw_delta": raw_delta,
            "gate": gate,
            "gate_mean": gate.mean(),
            "delta_abs_mean": delta.abs().mean(),
            "head_type": self.head_type,
            "fusion_mode": self.fusion_mode,
        }
        if return_info:
            return delta, info
        return delta
