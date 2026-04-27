import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


ACTIVATIONS = {
    "gelu": nn.GELU(approximate="tanh"),
    "silu": nn.SiLU(),
}


class ResidualMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=2,
        act="gelu",
        input_residual=False,
        output_residual=False,
    ):
        super().__init__()
        self.num_layers = num_layers

        if self.num_layers == -1:
            self.fc = nn.Linear(in_dim, out_dim)
            self.residual = input_residual and output_residual and (in_dim == out_dim)
            return

        self.act = ACTIVATIONS[act]
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.input_residual = input_residual and (in_dim == hidden_dim)
        self.output_residual = output_residual and (hidden_dim == out_dim)

    def forward(self, x):
        if self.num_layers == -1:
            return x + self.fc(x) if self.residual else self.fc(x)

        x = x + self.act(self.fc1(x)) if self.input_residual else self.act(self.fc1(x))
        for layer in self.hidden_layers:
            x = x + self.act(layer(x))
        x = x + self.fc2(x) if self.output_residual else self.fc2(x)
        return x


class FlareAttention(nn.Module):
    def __init__(
        self,
        channel_dim,
        num_heads=4,
        num_latents=32,
        attn_scale=1.0,
        act="gelu",
        num_layers_k_proj=2,
        num_layers_v_proj=2,
        k_proj_mlp_ratio=1.0,
        v_proj_mlp_ratio=1.0,
        qk_norm=False,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_heads = num_heads
        self.num_latents = num_latents
        self.head_dim = channel_dim // num_heads
        self.attn_scale = attn_scale

        if channel_dim % num_heads != 0:
            raise ValueError("channel_dim must be divisible by num_heads")

        self.latent_q = nn.Parameter(torch.empty(channel_dim, num_latents))
        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)

        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

        self.k_proj = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=int(channel_dim * k_proj_mlp_ratio),
            out_dim=channel_dim,
            num_layers=num_layers_k_proj,
            act=act,
            input_residual=True,
            output_residual=True,
        )
        self.v_proj = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=int(channel_dim * v_proj_mlp_ratio),
            out_dim=channel_dim,
            num_layers=num_layers_v_proj,
            act=act,
            input_residual=True,
            output_residual=True,
        )
        self.out_proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x):
        q = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim)
        q = self.q_norm(q).unsqueeze(0).expand(x.size(0), -1, -1, -1)

        k = rearrange(self.k_proj(x), "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = self.k_norm(k)

        z = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale)
        y = F.scaled_dot_product_attention(k, q, z, scale=self.attn_scale)
        y = rearrange(y, "b h n d -> b n (h d)")
        return self.out_proj(y)


class FlareBlock(nn.Module):
    def __init__(
        self,
        channel_dim,
        num_heads=4,
        num_latents=32,
        act="gelu",
        num_layers_k_proj=2,
        num_layers_v_proj=2,
        k_proj_mlp_ratio=1.0,
        v_proj_mlp_ratio=1.0,
        num_layers_ffn=2,
        ffn_mlp_ratio=1.0,
        qk_norm=False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(channel_dim)
        self.norm2 = nn.LayerNorm(channel_dim)
        self.attention = FlareAttention(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            act=act,
            num_layers_k_proj=num_layers_k_proj,
            num_layers_v_proj=num_layers_v_proj,
            k_proj_mlp_ratio=k_proj_mlp_ratio,
            v_proj_mlp_ratio=v_proj_mlp_ratio,
            qk_norm=qk_norm,
        )
        self.mlp = ResidualMLP(
            in_dim=channel_dim,
            hidden_dim=int(channel_dim * ffn_mlp_ratio),
            out_dim=channel_dim,
            num_layers=num_layers_ffn,
            act=act,
            input_residual=True,
            output_residual=True,
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SimpleFlare(nn.Module):
    def __init__(
        self,
        input_dim,
        output_steps=1,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        num_slices=32,
    ):
        super().__init__()
        self.output_steps = output_steps

        self.input_proj = ResidualMLP(
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
                FlareBlock(
                    channel_dim=hidden_dim,
                    num_heads=num_heads,
                    num_latents=num_slices,
                    act="gelu",
                    num_layers_k_proj=2,
                    num_layers_v_proj=2,
                    k_proj_mlp_ratio=1.0,
                    v_proj_mlp_ratio=1.0,
                    num_layers_ffn=2,
                    ffn_mlp_ratio=1.0,
                    qk_norm=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            ResidualMLP(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                out_dim=output_steps * 3,
                num_layers=2,
                act="gelu",
                input_residual=True,
                output_residual=False,
            ),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, batch):
        x = batch["node_features"].unsqueeze(0)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        output = self.output_proj(x).squeeze(0)
        output = output.view(output.shape[0], self.output_steps, 3).permute(1, 0, 2)
        return output + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
