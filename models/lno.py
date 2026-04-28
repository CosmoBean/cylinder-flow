# Latent Neural Operator reference: https://arxiv.org/abs/2406.03923
import torch
from torch import nn

from .flare import ResidualMLP


class LatentAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, latent_tokens):
        attended, _ = self.attention(
            self.norm1(latent_tokens),
            self.norm1(latent_tokens),
            self.norm1(latent_tokens),
        )
        latent_tokens = latent_tokens + attended
        latent_tokens = latent_tokens + self.mlp(self.norm2(latent_tokens))
        return latent_tokens


class SimpleLNO(nn.Module):
    def __init__(
        self,
        input_dim,
        output_steps=1,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        num_modes=64,
    ):
        super().__init__()
        self.output_steps = output_steps
        self.num_modes = num_modes

        self.trunk_proj = ResidualMLP(
            in_dim=2 + 4 + 6,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            act="gelu",
            input_residual=False,
            output_residual=True,
        )
        self.branch_proj = ResidualMLP(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            act="gelu",
            input_residual=False,
            output_residual=True,
        )
        self.encode_proj = ResidualMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=num_modes,
            num_layers=2,
            act="gelu",
            input_residual=False,
            output_residual=False,
        )
        self.decode_proj = ResidualMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=num_modes,
            num_layers=2,
            act="gelu",
            input_residual=False,
            output_residual=False,
        )
        self.blocks = nn.ModuleList(
            [LatentAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            ResidualMLP(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                out_dim=output_steps * 3,
                num_layers=2,
                act="gelu",
                input_residual=True,
                output_residual=False,
            ),
        )

    def forward(self, batch):
        global_features = batch["global_features"].view(1, -1).repeat(batch["mesh_pos"].shape[0], 1)
        trunk_input = torch.cat([batch["mesh_pos"], batch["node_type"], global_features], dim=-1)

        trunk_features = self.trunk_proj(trunk_input).unsqueeze(0)
        branch_features = self.branch_proj(batch["node_features"]).unsqueeze(0)

        encode_scores = torch.softmax(self.encode_proj(trunk_features), dim=1)
        decode_scores = torch.softmax(self.decode_proj(trunk_features), dim=-1)

        latent_tokens = torch.einsum("bnm,bnd->bmd", encode_scores, branch_features)
        for block in self.blocks:
            latent_tokens = block(latent_tokens)

        decoded = torch.einsum("bnm,bmd->bnd", decode_scores, latent_tokens)
        output = self.output_proj(decoded).squeeze(0)
        output = output.view(output.shape[0], self.output_steps, 3).permute(1, 0, 2)
        return output + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
