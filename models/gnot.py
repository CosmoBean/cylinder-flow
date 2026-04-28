import torch
from torch import nn

from .flare import ResidualMLP


class LinearAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query_tokens, key_value_tokens=None):
        if key_value_tokens is None:
            key_value_tokens = query_tokens

        query = self.query(query_tokens)
        key = self.key(key_value_tokens)
        value = self.value(key_value_tokens)

        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        query = torch.softmax(query, dim=-1)
        key = torch.softmax(key, dim=-1)

        context = torch.matmul(key.transpose(-2, -1), value)
        output = torch.matmul(query, context)
        output = output.transpose(1, 2).contiguous().view(query_tokens.shape[0], query_tokens.shape[1], -1)
        return self.output(output)


class GNOTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, num_experts=3):
        super().__init__()
        self.geometry_norm = nn.LayerNorm(hidden_dim)
        self.physics_norm = nn.LayerNorm(hidden_dim)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.post_cross_norm = nn.LayerNorm(hidden_dim)
        self.post_self_norm = nn.LayerNorm(hidden_dim)
        self.post_mlp_norm = nn.LayerNorm(hidden_dim)

        self.cross_attention = LinearAttention(hidden_dim, num_heads=num_heads)
        self.self_attention = LinearAttention(hidden_dim, num_heads=num_heads)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, geometry_tokens, physics_tokens):
        geometry_tokens = geometry_tokens + self.cross_attention(
            self.geometry_norm(geometry_tokens),
            self.physics_norm(physics_tokens),
        )
        geometry_tokens = self.post_cross_norm(geometry_tokens)
        geometry_tokens = geometry_tokens + self.self_attention(self.self_norm(geometry_tokens))
        geometry_tokens = self.post_self_norm(geometry_tokens)

        gate_scores = torch.softmax(self.gate(geometry_tokens), dim=-1)
        expert_outputs = torch.stack([expert(geometry_tokens) for expert in self.experts], dim=-1)
        mixed_output = (gate_scores.unsqueeze(2) * expert_outputs).sum(dim=-1)
        geometry_tokens = geometry_tokens + mixed_output
        return self.post_mlp_norm(geometry_tokens)


class SimpleGNOT(nn.Module):
    def __init__(
        self,
        input_dim,
        output_steps=1,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        num_experts=3,
    ):
        super().__init__()
        self.output_steps = output_steps

        self.geometry_proj = ResidualMLP(
            in_dim=2 + 4 + 6,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            act="gelu",
            input_residual=False,
            output_residual=True,
        )
        self.physics_proj = ResidualMLP(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            act="gelu",
            input_residual=False,
            output_residual=True,
        )
        self.blocks = nn.ModuleList(
            [
                GNOTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_experts=num_experts,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = ResidualMLP(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=output_steps * 3,
            num_layers=2,
            act="gelu",
            input_residual=True,
            output_residual=False,
        )

    def forward(self, batch):
        global_features = batch["global_features"].view(1, -1).repeat(batch["mesh_pos"].shape[0], 1)
        geometry_input = torch.cat([batch["mesh_pos"], batch["node_type"], global_features], dim=-1)

        geometry_tokens = self.geometry_proj(geometry_input).unsqueeze(0)
        physics_tokens = self.physics_proj(batch["node_features"]).unsqueeze(0)

        for block in self.blocks:
            geometry_tokens = block(geometry_tokens, physics_tokens)

        fused_tokens = geometry_tokens + physics_tokens
        output = self.output_proj(fused_tokens).squeeze(0)
        output = output.view(output.shape[0], self.output_steps, 3).permute(1, 0, 2)
        return output + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
