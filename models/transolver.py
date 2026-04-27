import torch
from torch import nn


class SliceAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_slices):
        super().__init__()
        self.num_slices = num_slices
        self.slice_assign = nn.Linear(hidden_dim, num_slices)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.slice_norm = nn.LayerNorm(hidden_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, node_state):
        slice_weights = torch.softmax(self.slice_assign(node_state), dim=-1)
        slice_sums = slice_weights.transpose(0, 1) @ node_state
        slice_counts = slice_weights.sum(dim=0, keepdim=True).transpose(0, 1)
        slice_tokens = slice_sums / slice_counts.clamp_min(1e-6)

        attended_tokens, _ = self.attention(
            slice_tokens.unsqueeze(0),
            slice_tokens.unsqueeze(0),
            slice_tokens.unsqueeze(0),
        )
        slice_tokens = self.slice_norm(slice_tokens + attended_tokens.squeeze(0))

        node_update = slice_weights @ slice_tokens
        node_state = self.node_norm(node_state + node_update)
        return node_state + self.feed_forward(node_state)


class SimpleTransolver(nn.Module):
    def __init__(
        self,
        input_dim,
        output_steps=1,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        num_slices=32,
    ):
        super().__init__()
        self.output_steps = output_steps
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                SliceAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_slices=num_slices,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_steps * 3),
        )

    def forward(self, batch):
        hidden = self.encoder(batch["node_features"])
        for layer in self.layers:
            hidden = layer(hidden)
        output = self.decoder(hidden)
        output = output.view(hidden.shape[0], self.output_steps, 3).permute(1, 0, 2)
        return output + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
