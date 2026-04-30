# GNOT reference: https://arxiv.org/abs/2302.14376
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.input(x))
        for layer in self.hidden:
            x = x + self.activation(layer(x))
        return self.output(x)


class LinearAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens, key_value_tokens=None):
        if key_value_tokens is None:
            key_value_tokens = query_tokens

        batch_size, query_len, hidden_dim = query_tokens.shape
        key_len = key_value_tokens.shape[1]

        query = self.query(query_tokens).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(key_value_tokens).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(key_value_tokens).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        query = query.softmax(dim=-1)
        key = key.softmax(dim=-1)
        key_sum = key.sum(dim=-2, keepdim=True)
        normalizer = 1.0 / (query * key_sum).sum(dim=-1, keepdim=True).clamp_min(1e-6)

        context = key.transpose(-2, -1) @ value
        output = self.dropout((query @ context) * normalizer + query)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.proj(output)


class LinearCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_inputs=1, dropout=0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_inputs = num_inputs
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.keys = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_inputs)])
        self.values = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_inputs)])
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_tokens, branch_tokens_list):
        batch_size, query_len, hidden_dim = query_tokens.shape
        query = self.query(query_tokens).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        query = query.softmax(dim=-1)
        output = query

        for index, branch_tokens in enumerate(branch_tokens_list):
            branch_len = branch_tokens.shape[1]
            key = self.keys[index](branch_tokens).view(batch_size, branch_len, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.values[index](branch_tokens).view(batch_size, branch_len, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.softmax(dim=-1)
            key_sum = key.sum(dim=-2, keepdim=True)
            normalizer = 1.0 / (query * key_sum).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            output = output + (query @ (key.transpose(-2, -1) @ value)) * normalizer

        output = self.dropout(output)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.proj(output)


class GNOTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_experts, inner_ratio=4, dropout=0.0):
        super().__init__()
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.branch_norms = nn.ModuleList([nn.LayerNorm(hidden_dim)])
        self.moe1_norm = nn.LayerNorm(hidden_dim)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.moe2_norm = nn.LayerNorm(hidden_dim)

        self.cross_attention = LinearCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_inputs=1,
            dropout=dropout,
        )
        self.self_attention = LinearAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        inner_dim = hidden_dim * inner_ratio
        self.experts1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, inner_dim),
                    nn.GELU(),
                    nn.Linear(inner_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.experts2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, inner_dim),
                    nn.GELU(),
                    nn.Linear(inner_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Sequential(
            nn.Linear(2, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, num_experts),
        )

    def forward(self, trunk_tokens, branch_tokens_list, positions):
        gate_scores = F.softmax(self.gate(positions), dim=-1).unsqueeze(2)
        normalized_branches = [self.branch_norms[0](branch_tokens_list[0])]

        trunk_tokens = trunk_tokens + self.cross_attention(self.cross_norm(trunk_tokens), normalized_branches)

        expert_outputs1 = torch.stack([expert(trunk_tokens) for expert in self.experts1], dim=-1)
        trunk_tokens = trunk_tokens + self.moe1_norm((gate_scores * expert_outputs1).sum(dim=-1))

        trunk_tokens = trunk_tokens + self.self_attention(self.self_norm(trunk_tokens))

        expert_outputs2 = torch.stack([expert(trunk_tokens) for expert in self.experts2], dim=-1)
        trunk_tokens = trunk_tokens + self.moe2_norm((gate_scores * expert_outputs2).sum(dim=-1))
        return trunk_tokens


class SimpleGNOT(nn.Module):
    def __init__(
        self,
        input_dim,
        output_steps=1,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        num_experts=2,
    ):
        super().__init__()
        self.output_steps = output_steps

        trunk_dim = 2 + 4 + 6
        branch_dim = input_dim - trunk_dim
        self.trunk_mlp = MLP(
            input_dim=trunk_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            activation=nn.GELU(),
        )
        self.branch_mlps = nn.ModuleList(
            [
                MLP(
                    input_dim=branch_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    num_layers=2,
                    activation=nn.GELU(),
                )
            ]
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
        self.output_mlp = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_steps * 3,
            num_layers=2,
            activation=nn.GELU(),
        )

    def forward(self, batch):
        num_nodes = batch["mesh_pos"].shape[0]
        global_features = batch["global_features"].view(1, -1).repeat(num_nodes, 1)
        trunk_input = torch.cat([batch["mesh_pos"], batch["node_type"], global_features], dim=-1).unsqueeze(0)
        positions = batch["mesh_pos"].unsqueeze(0)
        branch_input = batch["inputs"].permute(1, 0, 2).reshape(num_nodes, -1).unsqueeze(0)

        trunk_tokens = self.trunk_mlp(trunk_input)
        branch_tokens_list = [self.branch_mlps[0](branch_input)]

        for block in self.blocks:
            trunk_tokens = block(trunk_tokens, branch_tokens_list, positions)

        prediction = self.output_mlp(trunk_tokens).squeeze(0)
        prediction = prediction.view(prediction.shape[0], self.output_steps, 3)
        prediction = prediction.permute(1, 0, 2)
        return prediction + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
