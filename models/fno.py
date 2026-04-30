# Fourier Neural Operator reference: https://arxiv.org/abs/2010.08895
import torch
import torch.nn.functional as F
from torch import nn


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
        width_modes = min(self.modes, width // 2 + 1)
        height_modes = min(self.modes, height)

        x_ft = torch.fft.rfft2(x)
        output_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        output_ft[:, :, :height_modes, :width_modes] = self.complex_multiply(
            x_ft[:, :, :height_modes, :width_modes],
            self.weights_top[:, :, :height_modes, :width_modes],
        )
        output_ft[:, :, -height_modes:, :width_modes] = self.complex_multiply(
            x_ft[:, :, -height_modes:, :width_modes],
            self.weights_bottom[:, :, :height_modes, :width_modes],
        )
        return torch.fft.irfft2(output_ft, s=(height, width))


class FNOBlock(nn.Module):
    def __init__(self, hidden_dim, modes):
        super().__init__()
        self.spectral = SpectralConv2d(hidden_dim, hidden_dim, modes)
        self.pointwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        return F.gelu(self.spectral(x) + self.pointwise(x))


class KernelInterpolator(nn.Module):
    def __init__(self, grid_size, num_neighbors=16, sigma=0.04):
        super().__init__()
        self.grid_size = grid_size
        self.num_neighbors = num_neighbors
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma))))

    def build_grid(self, device):
        x_coords = torch.linspace(0.0, 1.6, self.grid_size, device=device)
        y_coords = torch.linspace(0.0, 0.4, self.grid_size, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)

    def compute_weights(self, target_points, source_points):
        distances = torch.cdist(target_points, source_points)
        num_neighbors = min(self.num_neighbors, source_points.shape[0])
        nearest_distances, nearest_indices = torch.topk(
            distances,
            k=num_neighbors,
            dim=-1,
            largest=False,
        )

        sigma = self.log_sigma.exp().clamp_min(1e-4)
        weights = torch.exp(-(nearest_distances ** 2) / (2.0 * sigma ** 2))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return nearest_indices, weights

    def gather_interpolate(self, source_features, nearest_indices, weights):
        gathered = source_features[nearest_indices]
        return (gathered * weights.unsqueeze(-1)).sum(dim=1)

    def nodes_to_grid(self, node_features, mesh_pos):
        grid_points = self.build_grid(node_features.device)
        nearest_indices, weights = self.compute_weights(grid_points, mesh_pos)
        grid_features = self.gather_interpolate(node_features, nearest_indices, weights)
        return grid_features.view(self.grid_size, self.grid_size, -1).permute(2, 0, 1).unsqueeze(0)

    def grid_to_nodes(self, grid_features, mesh_pos):
        grid_points = self.build_grid(grid_features.device)
        flat_grid = grid_features.squeeze(0).permute(1, 2, 0).reshape(-1, grid_features.shape[1])
        nearest_indices, weights = self.compute_weights(mesh_pos, grid_points)
        return self.gather_interpolate(flat_grid, nearest_indices, weights)


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
        self.interpolator = KernelInterpolator(grid_size=grid_size)

        self.input_proj = nn.Conv2d(input_dim + 2, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(hidden_dim, modes) for _ in range(num_layers)])
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, output_steps * 3, kernel_size=1),
        )

    def build_grid_coords(self, device):
        x_coords = torch.linspace(0.0, 1.6, self.grid_size, device=device)
        y_coords = torch.linspace(0.0, 0.4, self.grid_size, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid = torch.stack([grid_x / 1.6, grid_y / 0.4], dim=0)
        return grid.unsqueeze(0)

    def forward(self, batch):
        grid_features = self.interpolator.nodes_to_grid(batch["node_features"], batch["mesh_pos"])
        grid_coords = self.build_grid_coords(grid_features.device)
        grid_features = torch.cat([grid_features, grid_coords], dim=1)
        grid_features = self.input_proj(grid_features)

        for block in self.blocks:
            grid_features = block(grid_features)

        grid_output = self.output_proj(grid_features)
        node_output = self.interpolator.grid_to_nodes(grid_output, batch["mesh_pos"])
        node_output = node_output.view(node_output.shape[0], self.output_steps, 3).permute(1, 0, 2)
        return node_output + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
