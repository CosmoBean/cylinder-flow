# Latent Neural Operator reference: https://arxiv.org/abs/2406.03923
import math

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.input(x))
        for layer in self.hidden:
            x = x + self.activation(layer(x))
        return self.output(x)


def vanilla_attention(query, key, value):
    scale = math.sqrt(query.shape[-1])
    scores = torch.softmax(torch.einsum("bhid,bhjd->bhij", query, key) / scale, dim=-1)
    return torch.einsum("bhij,bhjd->bhid", scores, value)


class LatentSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.to_query = nn.Linear(hidden_dim, hidden_dim)
        self.to_key = nn.Linear(hidden_dim, hidden_dim)
        self.to_value = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, num_tokens, hidden_dim = x.shape
        query = self.to_query(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        key = self.to_key(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)
        value = self.to_value(x).view(batch_size, num_tokens, self.num_heads, self.head_dim)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attended = vanilla_attention(query, key, value)
        attended = attended.permute(0, 2, 1, 3).contiguous().view(batch_size, num_tokens, hidden_dim)
        return self.proj(attended)


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = LatentSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, latent_tokens):
        latent_tokens = latent_tokens + self.attention(self.norm1(latent_tokens))
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

        trunk_dim = 2 + 4 + 6
        self.trunk_projector = MLP(
            input_dim=trunk_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
        )
        self.branch_projector = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
        )
        self.attention_projector = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_modes,
            num_layers=2,
        )
        self.blocks = nn.ModuleList(
            [AttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.output_projector = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_steps * 3,
            num_layers=2,
        )

    def forward(self, batch):
        num_nodes = batch["mesh_pos"].shape[0]
        global_features = batch["global_features"].view(1, -1).repeat(num_nodes, 1)
        trunk_input = torch.cat([batch["mesh_pos"], batch["node_type"], global_features], dim=-1)

        trunk_features = self.trunk_projector(trunk_input).unsqueeze(0)
        branch_features = self.branch_projector(batch["node_features"]).unsqueeze(0)

        scores = self.attention_projector(trunk_features)
        encode_scores = torch.softmax(scores, dim=1)
        decode_scores = torch.softmax(scores, dim=-1)

        latent_tokens = torch.einsum("bnm,bnd->bmd", encode_scores, branch_features)
        for block in self.blocks:
            latent_tokens = block(latent_tokens)

        decoded = torch.einsum("bnm,bmd->bnd", decode_scores, latent_tokens)
        prediction = self.output_projector(decoded).squeeze(0)
        prediction = prediction.view(prediction.shape[0], self.output_steps, 3)
        prediction = prediction.permute(1, 0, 2)
        return prediction + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
