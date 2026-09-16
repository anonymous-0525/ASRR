from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from asrr_core import ActionSequenceResidualAdapter, apply_residual


def normalize_asrr_variant(variant: str) -> str:
    aliases = {
        "action_only": "asrr_action",
        "obs_add": "asrr_obs_add",
        "mode_embed": "asrr_mode_embed",
        "obs_mode": "asrr_obs_mode",
    }
    return aliases.get(str(variant), str(variant))


class DPASRRAdapter(nn.Module):
    """Diffusion Policy wrapper around the policy-agnostic ASRR core.

    The adapter does not import the Diffusion Policy repository and only assumes
    aligned action and observation tensors.
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        asrr_variant: str = "asrr_obs_add",
        obs_context_dim: int = 0,
        encoder_type: str = "mlp",
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        head_type: str = "dense",
        max_delta: Optional[float] = None,
        freeze_last_action_dim: bool = False,
    ):
        super().__init__()
        asrr_variant = normalize_asrr_variant(asrr_variant)
        valid = {
            "asrr_action",
            "asrr_obs_add",
            "asrr_mode_embed",
            "asrr_obs_mode",
        }
        if asrr_variant not in valid:
            raise ValueError(f"Unsupported asrr_variant={asrr_variant}; valid={sorted(valid)}")
        if asrr_variant != "asrr_action" and obs_context_dim <= 0:
            raise ValueError(f"obs_context_dim must be positive for {asrr_variant}")

        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.asrr_variant = asrr_variant
        self.obs_context_dim = int(obs_context_dim)

        self.obs_encoder = None
        self.action_sequence_encoder = None
        self.mode_encoder = None

        if self.asrr_variant == "asrr_action":
            self.core = ActionSequenceResidualAdapter(
                action_dim=self.action_dim,
                horizon=self.horizon,
                hidden_dim=self.hidden_dim,
                fusion_mode="action_only",
                encoder_type=encoder_type,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                head_type=head_type,
                max_delta=max_delta,
                freeze_last_action_dim=freeze_last_action_dim,
            )
        elif self.asrr_variant == "asrr_obs_add":
            self.core = ActionSequenceResidualAdapter(
                action_dim=self.action_dim,
                horizon=self.horizon,
                state_context_dim=self.obs_context_dim,
                hidden_dim=self.hidden_dim,
                fusion_mode="state_add",
                encoder_type=encoder_type,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                head_type=head_type,
                max_delta=max_delta,
                freeze_last_action_dim=freeze_last_action_dim,
            )
        else:
            self.obs_encoder = nn.Sequential(
                nn.Linear(self.obs_context_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.action_sequence_encoder = nn.GRU(
                input_size=self.action_dim,
                hidden_size=self.hidden_dim,
                batch_first=True,
            )
            self.mode_encoder = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.core = ActionSequenceResidualAdapter(
                action_dim=self.action_dim,
                horizon=self.horizon,
                state_context_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                fusion_mode="state_add",
                encoder_type=encoder_type,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                head_type=head_type,
                max_delta=max_delta,
                freeze_last_action_dim=freeze_last_action_dim,
            )

    @property
    def uses_obs_context(self) -> bool:
        return self.asrr_variant != "asrr_action"

    def _mode_context(self, base_action: torch.Tensor, obs_context: torch.Tensor) -> torch.Tensor:
        _, action_hidden = self.action_sequence_encoder(base_action)
        action_embedding = action_hidden[-1]
        obs_embedding = self.obs_encoder(obs_context)
        mode_context = self.mode_encoder(torch.cat([action_embedding, obs_embedding], dim=-1))
        if self.asrr_variant == "asrr_obs_mode":
            mode_context = mode_context + obs_embedding
        return mode_context

    def forward(
        self,
        base_action: torch.Tensor,
        obs_context: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if self.asrr_variant == "asrr_action":
            return self.core(base_action, return_info=return_info)
        if obs_context is None:
            raise ValueError(f"obs_context is required for {self.asrr_variant}")
        if self.asrr_variant == "asrr_obs_add":
            return self.core(base_action, state_context=obs_context, return_info=return_info)

        mode_context = self._mode_context(base_action, obs_context)
        return self.core(base_action, state_context=mode_context, return_info=return_info)


class ASRRDiffusionPolicyWrapper(nn.Module):
    """Wrap a frozen DP-like policy that implements `predict_action`."""

    def __init__(self, base_policy: nn.Module, adapter: DPASRRAdapter, alpha: float = 1.0):
        super().__init__()
        self.base_policy = base_policy
        self.adapter = adapter
        self.alpha = float(alpha)
        for parameter in self.base_policy.parameters():
            parameter.requires_grad_(False)
        self.base_policy.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_policy.eval()
        return self

    def reset(self) -> None:
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = self.base_policy.predict_action(obs_dict)
        base_action = result["action"]

        obs_context = None
        if self.adapter.uses_obs_context:
            obs = obs_dict["obs"]
            obs_context = obs.reshape(obs.shape[0], -1)

        delta, info = self.adapter(base_action, obs_context=obs_context, return_info=True)
        refined = apply_residual(base_action, delta, alpha=self.alpha)
        output = dict(result)
        output.update(
            {
                "action": refined,
                "base_action": base_action,
                "delta_action": delta,
                "asrr_info": info,
            }
        )
        return output
