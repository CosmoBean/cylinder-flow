import torch
import torch.nn.functional as F
from torch import nn

from .flare import ResidualMLP


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights_top = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )
        self.weights_bottom = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def complex_multiply(self, inputs, weights):
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        output_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        output_ft[:, :, : self.modes, : self.modes] = self.complex_multiply(
            x_ft[:, :, : self.modes, : self.modes],
            self.weights_top,
        )
        output_ft[:, :, -self.modes :, : self.modes] = self.complex_multiply(
            x_ft[:, :, -self.modes :, : self.modes],
            self.weights_bottom,
        )
        return torch.fft.irfft2(output_ft, s=(height, width))


class FNOBlock(nn.Module):
    def __init__(self, hidden_dim, modes):
        super().__init__()
        self.spectral = SpectralConv2d(hidden_dim, hidden_dim, modes)
        self.pointwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.pointwise(x))


class SimpleFNO(nn.Module):
    def __init__(
        self,
        input_dim,
        output_steps=1,
        hidden_dim=64,
        num_layers=4,
        modes=12,
        grid_size=64,
    ):
        super().__init__()
        self.output_steps = output_steps
        self.grid_size = grid_size

        self.input_proj = ResidualMLP(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            act="gelu",
            input_residual=False,
            output_residual=True,
        )
        self.blocks = nn.ModuleList([FNOBlock(hidden_dim, modes) for _ in range(num_layers)])
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_steps * 3, kernel_size=1),
        )

    def scatter_to_grid(self, node_features, mesh_pos):
        num_channels = node_features.shape[-1]
        grid = torch.zeros(
            1,
            num_channels,
            self.grid_size,
            self.grid_size,
            device=node_features.device,
        )
        counts = torch.zeros(1, 1, self.grid_size, self.grid_size, device=node_features.device)

        x_index = torch.clamp(
            torch.round(mesh_pos[:, 0] / 1.6 * (self.grid_size - 1)).long(),
            0,
            self.grid_size - 1,
        )
        y_index = torch.clamp(
            torch.round(mesh_pos[:, 1] / 0.4 * (self.grid_size - 1)).long(),
            0,
            self.grid_size - 1,
        )
        flat_index = y_index * self.grid_size + x_index

        grid = grid.view(1, num_channels, -1)
        counts = counts.view(1, 1, -1)
        grid.index_add_(2, flat_index, node_features.t().unsqueeze(0))
        counts.index_add_(
            2,
            flat_index,
            torch.ones(1, 1, flat_index.shape[0], device=node_features.device),
        )

        grid = grid / counts.clamp_min(1.0)
        return grid.view(1, num_channels, self.grid_size, self.grid_size)

    def sample_from_grid(self, grid_output, mesh_pos):
        x_coord = mesh_pos[:, 0] / 1.6 * 2.0 - 1.0
        y_coord = mesh_pos[:, 1] / 0.4 * 2.0 - 1.0
        sample_grid = torch.stack([x_coord, y_coord], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            grid_output,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
        return sampled

    def forward(self, batch):
        node_features = self.input_proj(batch["node_features"])
        grid_features = self.scatter_to_grid(node_features, batch["mesh_pos"])

        for block in self.blocks:
            grid_features = block(grid_features)

        grid_output = self.output_proj(grid_features)
        node_output = self.sample_from_grid(grid_output, batch["mesh_pos"])
        node_output = node_output.view(node_output.shape[0], self.output_steps, 3).permute(1, 0, 2)
        return node_output + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
