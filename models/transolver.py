# Transolver reference: https://arxiv.org/abs/2402.02366
import torch
from einops import rearrange
from torch import nn


class PhysicsAttentionIrregularMesh(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_slices, dropout=0.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        head_dim = hidden_dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.slice_temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1) * 0.5)

        self.project_x = nn.Linear(hidden_dim, hidden_dim)
        self.project_fx = nn.Linear(hidden_dim, hidden_dim)
        self.project_slice = nn.Linear(head_dim, num_slices)
        nn.init.orthogonal_(self.project_slice.weight)

        self.to_query = nn.Linear(head_dim, head_dim, bias=False)
        self.to_key = nn.Linear(head_dim, head_dim, bias=False)
        self.to_value = nn.Linear(head_dim, head_dim, bias=False)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_state):
        num_nodes = node_state.shape[0]

        fx_mid = self.project_fx(node_state).view(num_nodes, self.num_heads, self.head_dim)
        x_mid = self.project_x(node_state).view(num_nodes, self.num_heads, self.head_dim)
        fx_mid = fx_mid.permute(1, 0, 2)
        x_mid = x_mid.permute(1, 0, 2)

        temperature = torch.clamp(self.slice_temperature, min=0.1, max=5.0)
        slice_logits = self.project_slice(x_mid) / temperature.squeeze(0)
        slice_weights = self.softmax(slice_logits)
        slice_norm = slice_weights.sum(dim=1, keepdim=True)

        slice_tokens = torch.einsum("hnd,hng->hgd", fx_mid, slice_weights)
        slice_tokens = slice_tokens / slice_norm.transpose(1, 2).clamp_min(1e-6)

        query = self.to_query(slice_tokens)
        key = self.to_key(slice_tokens)
        value = self.to_value(slice_tokens)

        attention_logits = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention = self.softmax(attention_logits)
        attention = self.dropout(attention)
        updated_slice_tokens = torch.matmul(attention, value)

        updated_nodes = torch.einsum("hgd,hng->hnd", updated_slice_tokens, slice_weights)
        updated_nodes = rearrange(updated_nodes, "h n d -> n (h d)")
        return self.output(updated_nodes)


class TransolverBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_slices, dropout=0.0, mlp_ratio=2):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = PhysicsAttentionIrregularMesh(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_slices=num_slices,
            dropout=dropout,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, node_state):
        node_state = node_state + self.attention(self.attention_norm(node_state))
        node_state = node_state + self.mlp(self.mlp_norm(node_state))
        return node_state


class SimpleTransolver(nn.Module):
    def __init__(
        self,
        input_dim,
        output_steps=1,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        num_slices=32,
        dropout=0.0,
    ):
        super().__init__()
        self.output_steps = output_steps
        static_dim = 2 + 4 + 6
        state_dim = input_dim - static_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.placeholder = nn.Parameter(torch.zeros(hidden_dim))
        self.layers = nn.ModuleList(
            [
                TransolverBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_slices=num_slices,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_steps * 3),
        )

    def forward(self, batch):
        num_nodes = batch["mesh_pos"].shape[0]
        state_input = batch["inputs"].permute(1, 0, 2).reshape(num_nodes, -1)
        global_features = batch["global_features"].view(1, -1).repeat(num_nodes, 1)
        geometry_input = torch.cat([batch["mesh_pos"], batch["node_type"], global_features], dim=-1)

        node_state = self.state_encoder(state_input) + self.geometry_encoder(geometry_input)
        node_state = node_state + self.placeholder.view(1, -1)

        for layer in self.layers:
            node_state = layer(node_state)

        prediction = self.output_decoder(node_state)
        prediction = prediction.view(node_state.shape[0], self.output_steps, 3)
        prediction = prediction.permute(1, 0, 2)
        return prediction + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
