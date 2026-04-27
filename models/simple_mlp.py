from torch import nn


class SimplePerNodeMLP(nn.Module):
    def __init__(self, input_dim, output_steps=1, hidden_dim=128):
        super().__init__()
        self.output_steps = output_steps
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_steps * 3),
        )

    def forward(self, batch):
        node_features = batch["node_features"]
        output = self.network(node_features)
        output = output.view(node_features.shape[0], self.output_steps, 3).permute(1, 0, 2)
        return output + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
