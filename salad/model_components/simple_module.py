import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from salad.model_components.transformer import TimeMLP


class TimePointwiseLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_ctx,
        mlp_ratio=2,
        act=F.leaky_relu,
        dropout=0.0,
        use_time=False,
    ):
        super().__init__()
        self.use_time = use_time
        self.act = act
        self.mlp1 = TimeMLP(
            dim_in, dim_in * mlp_ratio, dim_in, dim_ctx, use_time=use_time
        )
        self.norm1 = nn.LayerNorm(dim_in)

        self.mlp2 = TimeMLP(
            dim_in, dim_in * mlp_ratio, dim_in, dim_ctx, use_time=use_time
        )
        self.norm2 = nn.LayerNorm(dim_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ctx=None):
        res = x
        x = self.mlp1(x, ctx=ctx)
        x = self.norm1(x + res)

        res = x
        x = self.mlp2(x, ctx=ctx)
        x = self.norm2(x + res)
        return x


class TimePointWiseEncoder(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_ctx=None,
        mlp_ratio=2,
        act=F.leaky_relu,
        dropout=0.0,
        use_time=True,
        num_layers=6,
        last_fc=False,
        last_fc_dim_out=None,
    ):
        super().__init__()
        self.last_fc = last_fc
        if last_fc:
            self.fc = nn.Linear(dim_in, last_fc_dim_out)
        self.layers = nn.ModuleList(
            [
                TimePointwiseLayer(
                    dim_in,
                    dim_ctx=dim_ctx,
                    mlp_ratio=mlp_ratio,
                    act=act,
                    dropout=dropout,
                    use_time=use_time,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, ctx=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, ctx=ctx)
        if self.last_fc:
            x = self.fc(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
