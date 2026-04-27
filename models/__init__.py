from .gnn import SimpleMeshGNN
from .simple_mlp import SimplePerNodeMLP
from .transolver import SimpleTransolver


def build_model(name, input_dim, output_steps, hidden_dim, num_layers, num_heads, num_slices):
    if name == "simple_mlp":
        return SimplePerNodeMLP(
            input_dim=input_dim,
            output_steps=output_steps,
            hidden_dim=hidden_dim,
        )
    if name == "gnn":
        return SimpleMeshGNN(
            input_dim=input_dim,
            output_steps=output_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    if name == "transolver":
        return SimpleTransolver(
            input_dim=input_dim,
            output_steps=output_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_slices=num_slices,
        )
    raise ValueError(f"Unknown model: {name}")


__all__ = ["SimplePerNodeMLP", "SimpleMeshGNN", "SimpleTransolver", "build_model"]
