# FLARE reference: https://arxiv.org/abs/2508.12594
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


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
        act=None,
        input_residual=False,
        output_residual=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        if self.num_layers < -1:
            raise ValueError(f"num_layers must be at least -1, got {self.num_layers}")

        if self.num_layers == -1:
            self.fc = nn.Linear(in_dim, out_dim)
            self.residual = input_residual and output_residual and (in_dim == out_dim)
            return

        self.act = ACTIVATIONS[act] if act else ACTIVATIONS["gelu"]
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fcs = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.input_residual = input_residual and (in_dim == hidden_dim)
        self.output_residual = output_residual and (hidden_dim == out_dim)

    def forward(self, x):
        if self.num_layers == -1:
            return x + self.fc(x) if self.residual else self.fc(x)

        x = x + self.act(self.fc1(x)) if self.input_residual else self.act(self.fc1(x))
        for fc in self.fcs:
            x = x + self.act(fc(x))
        x = x + self.fc2(x) if self.output_residual else self.fc2(x)
        return x


class FLARE(nn.Module):
    def __init__(
        self,
        channel_dim,
        num_heads=8,
        num_latents=32,
        attn_scale=1.0,
        act=None,
        num_layers_k_proj=3,
        num_layers_v_proj=3,
        k_proj_mlp_ratio=1.0,
        v_proj_mlp_ratio=1.0,
        qk_norm=False,
        rmsnorm=False,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.num_latents = num_latents
        self.num_heads = channel_dim // 8 if num_heads is None else num_heads
        self.head_dim = channel_dim // self.num_heads

        if self.channel_dim % self.num_heads != 0:
            raise ValueError("channel_dim must be divisible by num_heads")
        if attn_scale <= 0.0:
            raise ValueError("attn_scale must be greater than 0")

        self.attn_scale = attn_scale
        self.latent_q = nn.Parameter(torch.empty(self.channel_dim, self.num_latents))
        nn.init.normal_(self.latent_q, mean=0.0, std=0.1)

        norm_cls = nn.RMSNorm if rmsnorm else nn.LayerNorm
        self.q_norm = norm_cls(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_cls(self.head_dim) if qk_norm else nn.Identity()

        self.k_proj = ResidualMLP(
            in_dim=self.channel_dim,
            hidden_dim=int(self.channel_dim * k_proj_mlp_ratio),
            out_dim=self.channel_dim,
            num_layers=num_layers_k_proj,
            act=act,
            input_residual=True,
            output_residual=True,
        )
        self.v_proj = ResidualMLP(
            in_dim=self.channel_dim,
            hidden_dim=int(self.channel_dim * v_proj_mlp_ratio),
            out_dim=self.channel_dim,
            num_layers=num_layers_v_proj,
            act=act,
            input_residual=True,
            output_residual=True,
        )
        self.out_proj = nn.Linear(self.channel_dim, self.channel_dim)

    def forward(self, x, return_scores=False):
        query = self.latent_q.view(self.num_heads, self.num_latents, self.head_dim)
        key = rearrange(self.k_proj(x), "b n (h d) -> b h n d", h=self.num_heads)
        value = rearrange(self.v_proj(x), "b n (h d) -> b h n d", h=self.num_heads)

        query = self.q_norm(query)
        key = self.k_norm(key)
        query = query.unsqueeze(0).expand(x.size(0), -1, -1, -1)

        if not return_scores:
            latent = F.scaled_dot_product_attention(query, key, value, scale=self.attn_scale)
            output = F.scaled_dot_product_attention(key, query, latent, scale=self.attn_scale)
            scores = None
        else:
            scores = query @ key.transpose(-2, -1)
            encode = F.softmax(scores, dim=-1)
            decode = F.softmax(scores.transpose(-2, -1), dim=-1)
            latent = encode @ value
            output = decode @ latent

        output = rearrange(output, "b h n d -> b n (h d)")
        output = self.out_proj(output)
        return output, scores


class FLAREBlock(nn.Module):
    def __init__(
        self,
        channel_dim,
        num_heads=None,
        num_latents=None,
        attn_scale=1.0,
        act=None,
        rmsnorm=False,
        num_layers_k_proj=3,
        num_layers_v_proj=3,
        k_proj_mlp_ratio=1.0,
        v_proj_mlp_ratio=1.0,
        num_layers_ffn=3,
        ffn_mlp_ratio=1.0,
        qk_norm=False,
    ):
        super().__init__()
        norm_cls = nn.RMSNorm if rmsnorm else nn.LayerNorm
        self.norm1 = norm_cls(channel_dim)
        self.norm2 = norm_cls(channel_dim)
        self.att = FLARE(
            channel_dim=channel_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            attn_scale=attn_scale,
            act=act,
            num_layers_k_proj=num_layers_k_proj,
            num_layers_v_proj=num_layers_v_proj,
            k_proj_mlp_ratio=k_proj_mlp_ratio,
            v_proj_mlp_ratio=v_proj_mlp_ratio,
            qk_norm=qk_norm,
            rmsnorm=rmsnorm,
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

    def forward(self, x, return_scores=False):
        attention_output, scores = self.att(self.norm1(x), return_scores=return_scores)
        x = x + attention_output
        x = x + self.mlp(self.norm2(x))
        return x, scores


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

        self.in_proj = ResidualMLP(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=2,
            act="gelu",
            input_residual=False,
            output_residual=True,
        )
        self.out_proj = nn.Sequential(
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
        self.blocks = nn.ModuleList(
            [
                FLAREBlock(
                    channel_dim=hidden_dim,
                    num_heads=num_heads,
                    num_latents=num_slices,
                    attn_scale=1.0,
                    act="gelu",
                    rmsnorm=False,
                    num_layers_k_proj=3,
                    num_layers_v_proj=3,
                    k_proj_mlp_ratio=1.0,
                    v_proj_mlp_ratio=1.0,
                    num_layers_ffn=3,
                    ffn_mlp_ratio=1.0,
                    qk_norm=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, batch):
        x = self.in_proj(batch["node_features"].unsqueeze(0))
        for block in self.blocks:
            x, _ = block(x, return_scores=False)
        prediction = self.out_proj(x).squeeze(0)
        prediction = prediction.view(prediction.shape[0], self.output_steps, 3).permute(1, 0, 2)
        return prediction + batch["inputs"][-1:].repeat(self.output_steps, 1, 1)
