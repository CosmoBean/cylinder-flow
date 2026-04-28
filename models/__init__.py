from .fno import SimpleFNO
from .flare import SimpleFlare
from .gnot import SimpleGNOT
from .gnn import SimpleMeshGNN
from .lno import SimpleLNO
from .transolver import SimpleTransolver


def build_model(
    name,
    input_dim,
    output_steps,
    hidden_dim,
    num_layers,
    num_heads,
    num_slices,
    fno_modes,
    fno_grid_size,
):
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
    if name == "flare":
        return SimpleFlare(
            input_dim=input_dim,
            output_steps=output_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_slices=num_slices,
        )
    if name == "gnot":
        return SimpleGNOT(
            input_dim=input_dim,
            output_steps=output_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    if name == "lno":
        return SimpleLNO(
            input_dim=input_dim,
            output_steps=output_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_modes=num_slices,
        )
    if name == "fno":
        return SimpleFNO(
            input_dim=input_dim,
            output_steps=output_steps,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            modes=fno_modes,
            grid_size=fno_grid_size,
        )
    raise ValueError(f"Unknown model: {name}")


__all__ = [
    "SimpleMeshGNN",
    "SimpleTransolver",
    "SimpleFlare",
    "SimpleGNOT",
    "SimpleLNO",
    "SimpleFNO",
    "build_model",
]
