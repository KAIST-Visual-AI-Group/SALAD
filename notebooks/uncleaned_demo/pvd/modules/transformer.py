import torch
import torch.nn as nn
import torch.nn.functional as F
from pvd.diffusion.common import ConcatSquashLinear
import numpy as np

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        Input:
            x: [B,N,D]
        """
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class TimeMLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_h,
        dim_out,
        dim_ctx=None,
        act=F.relu,
        dropout=0.0,
        use_time=False,
    ):
        super().__init__()
        self.act = act
        self.use_time = use_time

        dim_h = int(dim_h)
        if use_time:
            self.fc1 = ConcatSquashLinear(dim_in, dim_h, dim_ctx)
            self.fc2 = ConcatSquashLinear(dim_h, dim_out, dim_ctx)
        else:
            self.fc1 = nn.Linear(dim_in, dim_h)
            self.fc2 = nn.Linear(dim_h, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ctx=None):
        if self.use_time:
            x = self.fc1(x=x, ctx=ctx)
        else:
            x = self.fc1(x)

        x = self.act(x)
        x = self.dropout(x)
        if self.use_time:
            x = self.fc2(x=x, ctx=ctx)
        else:
            x = self.fc2(x)

        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim**-0.5
        self.to_queries = nn.Linear(dim_self, dim_self)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        y=None,
        mask=None,
        alpha=None,
    ):
        y = y if y is not None else x
        b_a, n, c = x.shape
        b, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(
            b_a, n, self.num_heads, c // self.num_heads
        )
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(
            b, m, 2, self.num_heads, c // self.num_heads
        )
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        if alpha is not None:
            out, attention = self.forward_interpolation(
                queries, keys, values, alpha, mask
            )
        else:
            attention = torch.einsum("bnhd,bmhd->bnmh", queries, keys) * self.scale
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1)
                attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
            attention = attention.softmax(dim=2)
            out = torch.einsum("bnmh,bmhd->bnhd", attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TimeTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_self,
        dim_ctx=None,
        num_heads=1,
        mlp_ratio=2.0,
        act=F.leaky_relu,
        dropout=0.0,
        use_time=True,
    ):
        super().__init__()
        self.use_time = use_time
        self.act = act
        self.attn = MultiHeadAttention(dim_self, dim_self, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(dim_self)

        mlp_ratio = int(mlp_ratio)
        self.mlp = TimeMLP(dim_self, dim_self * mlp_ratio, dim_self, dim_ctx, use_time=use_time)
        self.norm = nn.LayerNorm(dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ctx=None):
        res = x
        x, attn = self.attn(x)
        x = self.attn_norm(x + res)

        res = x
        x = self.mlp(x, ctx=ctx)
        x = self.norm(x + res)

        return x, attn


class TimeTransformerDecoderLayer(TimeTransformerEncoderLayer):
    def __init__(
        self,
        dim_self,
        dim_ref,
        dim_ctx=None,
        num_heads=1,
        mlp_ratio=2,
        act=F.leaky_relu,
        dropout=0.0,
        use_time=True,
    ):
        super().__init__(
            dim_self=dim_self,
            dim_ctx=dim_ctx,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            act=act,
            dropout=dropout,
            use_time=use_time,
        )
        self.cross_attn = MultiHeadAttention(dim_self, dim_ref, num_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(dim_self)

    def forward(self, x, y, ctx=None):
        res = x
        x, attn = self.attn(x)
        x = self.attn_norm(x + res)

        res = x
        x, attn = self.cross_attn(x, y)
        x = self.cross_attn_norm(x + res)

        res = x
        x = self.mlp(x, ctx=ctx)
        x = self.norm(x + res)

        return x, attn


class TimeTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim_self,
        dim_ctx=None,
        num_heads=1,
        mlp_ratio=2.0,
        act=F.leaky_relu,
        dropout=0.0,
        use_time=True,
        num_layers=3,
        last_fc=False,
        last_fc_dim_out=None,
    ):
        super().__init__()
        self.last_fc = last_fc
        if last_fc:
            self.fc = nn.Linear(dim_self, last_fc_dim_out)
        self.layers = nn.ModuleList(
            [
                TimeTransformerEncoderLayer(
                    dim_self,
                    dim_ctx=dim_ctx,
                    num_heads=num_heads,
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
            x, attn = layer(x, ctx=ctx)

        if self.last_fc:
            x = self.fc(x)
        return x


class TimeTransformerDecoder(nn.Module):
    def __init__(
        self,
        dim_self,
        dim_ref,
        dim_ctx=None,
        num_heads=1,
        mlp_ratio=2.0,
        act=F.leaky_relu,
        dropout=0.0,
        use_time=True,
        num_layers=3,
        last_fc=True,
        last_fc_dim_out=None,
    ):
        super().__init__()
        self.last_fc = last_fc
        if last_fc:
            self.fc = nn.Linear(dim_self, last_fc_dim_out)

        self.layers = nn.ModuleList(
            [
                TimeTransformerDecoderLayer(
                    dim_self,
                    dim_ref,
                    dim_ctx,
                    num_heads,
                    mlp_ratio,
                    act,
                    dropout,
                    use_time,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, y, ctx=None):
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, y=y, ctx=ctx)
        if self.last_fc:
            x = self.fc(x)

        return x
