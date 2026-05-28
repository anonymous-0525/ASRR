from __future__ import annotations

import math

import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    """Token/channel mixer block used by the OpenVLA-OFT MLP-mixer head."""

    def __init__(self, horizon: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(horizon, horizon),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(horizon, horizon),
        )
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.token_norm(x).transpose(1, 2)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + y
        return x + self.channel_mlp(self.channel_norm(x))


class OpenVLAOFTResidualAdapter(nn.Module):
    """OpenVLA-OFT residual refiner compatible with the local ASRR experiments.

    Inputs are normalized OpenVLA-OFT action chunks plus VLA action-context
    features.  The module predicts a bounded residual in the same normalized
    action space.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        action_context_dim: int,
        obs_context_dim: int = 0,
        num_task_embeddings: int = 0,
        hidden_dim: int = 384,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        max_delta: list[float] | tuple[float, ...] | float = 0.05,
        gate_bias_init: float = -5.0,
        freeze_gripper: bool = True,
        head_type: str = "sigmoid_gate",
        lowrank_num_basis: int = 4,
        context_mode: str = "mean_context",
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if head_type not in {"sigmoid_gate", "bounded_delta_only", "lowrank_delta", "mlp_mixer_gate"}:
            raise ValueError("Unsupported head_type")
        if context_mode not in {"mean_context", "learned_pool", "dimtoken"}:
            raise ValueError("Unsupported context_mode")
        if context_mode == "dimtoken" and head_type == "lowrank_delta":
            raise ValueError("lowrank_delta is only supported for temporal context modes")

        self.context_mode = str(context_mode)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.action_context_dim = int(action_context_dim)
        self.obs_context_dim = int(obs_context_dim)
        self.num_task_embeddings = int(num_task_embeddings)
        self.hidden_dim = int(hidden_dim)
        self.head_type = str(head_type)
        self.lowrank_num_basis = int(lowrank_num_basis)

        self.base_action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        self.base_scalar_encoder = nn.Linear(1, self.hidden_dim)
        self.action_context_encoder = nn.Linear(self.action_context_dim, self.hidden_dim)
        self.raw_action_hidden_encoder = nn.Linear(self.action_context_dim, self.hidden_dim)
        self.raw_action_dim_embed = nn.Parameter(torch.zeros(self.action_dim, self.action_context_dim))
        self.dim_token_embed = nn.Parameter(torch.zeros(self.action_dim, self.hidden_dim))
        self.pool_score = nn.Sequential(
            nn.LayerNorm(self.action_context_dim),
            nn.Linear(self.action_context_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.pos_embed = nn.Parameter(torch.zeros(self.horizon, self.hidden_dim))
        self.obs_encoder = (
            nn.Sequential(
                nn.LayerNorm(self.obs_context_dim),
                nn.Linear(self.obs_context_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            if self.obs_context_dim > 0
            else None
        )
        self.task_embedding = nn.Embedding(self.num_task_embeddings, self.hidden_dim) if self.num_task_embeddings > 0 else None

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
        self.mixer_encoder = nn.Sequential(
            *[MixerBlock(self.horizon, self.hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.delta_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.gate_head = nn.Linear(self.hidden_dim, 1)
        self.dim_delta_head = nn.Linear(self.hidden_dim, 1)
        self.dim_gate_head = nn.Linear(self.hidden_dim, 1)
        self.lowrank_coeff_head = nn.Linear(self.hidden_dim, self.lowrank_num_basis * self.action_dim)

        if isinstance(max_delta, (list, tuple)):
            if len(max_delta) != self.action_dim:
                raise ValueError(f"max_delta length must be action_dim={self.action_dim}")
            max_delta_tensor = torch.tensor(max_delta, dtype=torch.float32)
        else:
            max_delta_tensor = torch.full((self.action_dim,), float(max_delta), dtype=torch.float32)
        self.register_buffer("max_delta", max_delta_tensor)

        mask = torch.ones(self.action_dim, dtype=torch.float32)
        if freeze_gripper and self.action_dim > 0:
            mask[-1] = 0.0
        self.register_buffer("action_mask", mask)
        self.register_buffer("lowrank_basis", self._make_dct_basis(self.horizon, self.lowrank_num_basis))

        self.reset_parameters(gate_bias_init)

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

    def reset_parameters(self, gate_bias_init: float) -> None:
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias_init))
        nn.init.zeros_(self.dim_delta_head.weight)
        nn.init.zeros_(self.dim_delta_head.bias)
        nn.init.zeros_(self.dim_gate_head.weight)
        nn.init.constant_(self.dim_gate_head.bias, float(gate_bias_init))
        nn.init.zeros_(self.lowrank_coeff_head.weight)
        nn.init.zeros_(self.lowrank_coeff_head.bias)
        nn.init.normal_(self.raw_action_dim_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.dim_token_embed, mean=0.0, std=0.02)
        if self.task_embedding is not None:
            nn.init.normal_(self.task_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        base_action: torch.Tensor,
        action_context: torch.Tensor,
        obs_context: torch.Tensor | None = None,
        task_index: torch.Tensor | None = None,
        return_info: bool = False,
    ):
        if base_action.ndim != 3:
            raise ValueError(f"base_action must be [B,H,D], got {tuple(base_action.shape)}")
        batch_size, horizon, action_dim = base_action.shape
        if action_dim != self.action_dim:
            raise ValueError(f"Expected action_dim={self.action_dim}, got {action_dim}")
        if horizon > self.horizon:
            raise ValueError(f"Adapter horizon={self.horizon}, got {horizon}")

        obs_token = None
        if self.obs_encoder is not None:
            if obs_context is None:
                raise ValueError("obs_context is required")
            obs_token = self.obs_encoder(obs_context)

        task_token = None
        if self.task_embedding is not None:
            if task_index is None:
                raise ValueError("task_index is required")
            task_token = self.task_embedding(task_index.reshape(-1).long())

        if self.context_mode == "dimtoken":
            if action_context.ndim != 4:
                raise ValueError(f"dimtoken mode expects action_context [B,H,D,C], got {tuple(action_context.shape)}")
            if action_context.shape[:3] != base_action.shape:
                raise ValueError("dimtoken context must share batch, horizon, and action_dim with base_action")
            dim_tokens = (
                self.raw_action_hidden_encoder(action_context)
                + self.base_scalar_encoder(base_action.unsqueeze(-1))
                + self.pos_embed[:horizon].view(1, horizon, 1, -1)
                + self.dim_token_embed.view(1, 1, self.action_dim, -1)
            )
            if obs_token is not None:
                dim_tokens = dim_tokens + obs_token.view(batch_size, 1, 1, -1)
            if task_token is not None:
                dim_tokens = dim_tokens + task_token.view(batch_size, 1, 1, -1)
            hidden = self.sequence_encoder(dim_tokens.reshape(batch_size, horizon * self.action_dim, self.hidden_dim))
            hidden = self.final_norm(hidden).view(batch_size, horizon, self.action_dim, self.hidden_dim)
            raw = self.dim_delta_head(hidden).squeeze(-1)
            raw_delta = torch.tanh(raw) * self.max_delta.view(1, 1, -1)
            if self.head_type in {"sigmoid_gate", "mlp_mixer_gate"}:
                gate = torch.sigmoid(self.dim_gate_head(hidden)).squeeze(-1)
            else:
                gate = torch.ones(batch_size, horizon, self.action_dim, dtype=base_action.dtype, device=base_action.device)
            delta = raw_delta * gate * self.action_mask.view(1, 1, -1)
        else:
            if self.context_mode == "mean_context":
                if action_context.ndim != 3:
                    raise ValueError(f"mean_context mode expects action_context [B,H,C], got {tuple(action_context.shape)}")
                if action_context.shape[:2] != base_action.shape[:2]:
                    raise ValueError("action_context must share batch and horizon with base_action")
                temporal_context = action_context
            else:
                if action_context.ndim != 4:
                    raise ValueError(f"learned_pool mode expects action_context [B,H,D,C], got {tuple(action_context.shape)}")
                if action_context.shape[:3] != base_action.shape:
                    raise ValueError("learned_pool context must share batch, horizon, and action_dim with base_action")
                raw_tokens = action_context + self.raw_action_dim_embed.view(1, 1, self.action_dim, -1)
                weights = torch.softmax(self.pool_score(raw_tokens), dim=2)
                temporal_context = (weights * raw_tokens).sum(dim=2)

            tokens = self.base_action_encoder(base_action) + self.action_context_encoder(temporal_context)
            tokens = tokens + self.pos_embed[:horizon].unsqueeze(0)
            if obs_token is not None:
                tokens = tokens + obs_token.unsqueeze(1)
            if task_token is not None:
                tokens = tokens + task_token.unsqueeze(1)

            if self.head_type == "mlp_mixer_gate":
                hidden = self.final_norm(self.mixer_encoder(tokens))
            else:
                hidden = self.final_norm(self.sequence_encoder(tokens))
            if self.head_type == "lowrank_delta":
                coeff = self.lowrank_coeff_head(hidden.mean(dim=1)).view(batch_size, self.lowrank_num_basis, self.action_dim)
                raw = torch.einsum("bkd,kh->bhd", coeff, self.lowrank_basis[:, :horizon])
            else:
                raw = self.delta_head(hidden)
            raw_delta = torch.tanh(raw) * self.max_delta.view(1, 1, -1)

            if self.head_type in {"sigmoid_gate", "mlp_mixer_gate"}:
                gate = torch.sigmoid(self.gate_head(hidden))
            else:
                gate = torch.ones(batch_size, horizon, 1, dtype=base_action.dtype, device=base_action.device)
            delta = raw_delta * gate * self.action_mask.view(1, 1, -1)

        info = {
            "gate": gate,
            "gate_mean": gate.mean(),
            "delta_abs_mean": delta.abs().mean(),
            "raw_delta": raw_delta,
        }
        if return_info:
            return delta, info
        return delta
