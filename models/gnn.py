import torch
from torch import nn


class MeshMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, node_state, edge_index, edge_attr):
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        edge_inputs = torch.cat(
            [node_state[source_nodes], node_state[target_nodes], edge_attr],
            dim=-1,
        )
        messages = self.message_mlp(edge_inputs)

        aggregated = torch.zeros_like(node_state)
        aggregated.index_add_(0, target_nodes, messages)

        degrees = torch.zeros(node_state.shape[0], 1, device=node_state.device)
        degrees.index_add_(
            0,
            target_nodes,
            torch.ones(target_nodes.shape[0], 1, device=node_state.device),
        )
        aggregated = aggregated / degrees.clamp_min(1.0)

        update = self.update_mlp(torch.cat([node_state, aggregated], dim=-1))
        return node_state + update


class SimpleMeshGNN(nn.Module):
    def __init__(self, input_dim, output_steps=1, hidden_dim=128, num_layers=3):
        super().__init__()
        self.output_steps = output_steps
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MeshMessagePassingLayer(hidden_dim) for _ in range(num_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_steps * 3),
        )

    def forward(self, batch):
        hidden = self.encoder(batch["node_features"])
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]

        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_attr)

        output = self.decoder(hidden)
        output = output.view(hidden.shape[0], self.output_steps, 3).permute(1, 0, 2)
        return output + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
