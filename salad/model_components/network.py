import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap
from salad.model_components.simple_module import TimePointWiseEncoder, TimestepEmbedder


from salad.model_components.transformer import (
    PositionalEncoding,
    TimeTransformerDecoder,
    TimeTransformerEncoder,
)

class UnCondDiffNetwork(nn.Module):
    def __init__(self, input_dim, residual, **kwargs):
        """
        Transformer Encoder.
        """
        super().__init__()
        self.input_dim = input_dim
        self.residual = residual
        self.__dict__.update(kwargs)
        self.hparams = DotMap(self.__dict__)

        self._build_model()

    def _build_model(self):
        self.act = F.leaky_relu
        if self.hparams.get("use_timestep_embedder"):
            self.time_embedder = TimestepEmbedder(self.hparams.timestep_embedder_dim)
            dim_ctx = self.hparams.timestep_embedder_dim
        else:
            dim_ctx = 3

        """
        Encoder part
        """
        enc_dim = self.hparams.embedding_dim
        self.embedding = nn.Linear(self.hparams.input_dim, enc_dim)
        if not self.hparams.get("encoder_type"):
            self.encoder = TimeTransformerEncoder(
                enc_dim,
                dim_ctx=dim_ctx,
                num_heads=self.hparams.num_heads
                if self.hparams.get("num_heads")
                else 4,
                use_time=True,
                num_layers=self.hparams.enc_num_layers,
                last_fc=True,
                last_fc_dim_out=self.hparams.input_dim,
            )
        else:
            if self.hparams.encoder_type == "transformer":
                self.encoder = TimeTransformerEncoder(
                    enc_dim,
                    dim_ctx=dim_ctx,
                    num_heads=self.hparams.num_heads
                    if self.hparams.get("num_heads")
                    else 4,
                    use_time=True,
                    num_layers=self.hparams.enc_num_layers,
                    last_fc=True,
                    last_fc_dim_out=self.hparams.input_dim,
                    dropout=self.hparams.get("attn_dropout", 0.0)
                )
            else:
                raise ValueError

    def forward(self, x, beta):
        """
        Input:
            x: [B,G,D] latent
            beta: B
        Output:
            eta: [B,G,D]
        """
        B, G = x.shape[:2]
        if self.hparams.get("use_timestep_embedder"):
            time_emb = self.time_embedder(beta).unsqueeze(1)
        else:
            beta = beta.view(B, 1, 1)
            time_emb = torch.cat(
                [beta, torch.sin(beta), torch.cos(beta)], dim=-1
            )  # [B,1,3]

        ctx = time_emb
        x_emb = self.embedding(x)

        out = self.encoder(x_emb, ctx=ctx)

        if self.hparams.residual:
            out = out + x
        return out


class CondDiffNetwork(nn.Module):
    def __init__(self, input_dim, residual, **kwargs):
        """
        Transformer Encoder + Decoder.
        """
        super().__init__()
        self.input_dim = input_dim
        self.residual = residual
        self.__dict__.update(kwargs)
        self.hparams = DotMap(self.__dict__)

        self._build_model()

    def _build_model(self):
        self.act = F.leaky_relu
        if self.hparams.get("use_timestep_embedder"):
            self.time_embedder = TimestepEmbedder(self.hparams.timestep_embedder_dim)
            dim_ctx = self.hparams.timestep_embedder_dim
        else:
            dim_ctx = 3
        """
        Encoder part
        """
        enc_dim = self.hparams.context_embedding_dim
        self.context_embedding = nn.Linear(self.hparams.context_dim, enc_dim)
        if self.hparams.encoder_type == "transformer":
            self.encoder = TimeTransformerEncoder(
                enc_dim,
                3,
                num_heads=4,
                use_time=self.hparams.encoder_use_time,
                num_layers=self.hparams.enc_num_layers
                if self.hparams.get("enc_num_layers")
                else 3,
                last_fc=False,
            )

        elif self.hparams.encoder_type == "pointwise":
            self.encoder = TimePointWiseEncoder(
                enc_dim,
                dim_ctx=None,
                use_time=self.hparams.encoder_use_time,
                num_layers=self.hparams.enc_num_layers,
            )
        else:
            raise ValueError

        """ 
        Decoder part
        """
        dec_dim = self.hparams.embedding_dim
        input_dim = self.hparams.input_dim
        self.query_embedding = nn.Linear(self.hparams.input_dim, dec_dim)
        if self.hparams.decoder_type == "transformer_decoder":
            self.decoder = TimeTransformerDecoder(
                dec_dim,
                enc_dim,
                dim_ctx=dim_ctx,
                num_heads=4,
                last_fc=True,
                last_fc_dim_out=input_dim,
                num_layers=self.hparams.dec_num_layers
                if self.hparams.get("dec_num_layers")
                else 3,
            )
        elif self.hparams.decoder_type == "transformer_encoder":
            self.decoder = TimeTransformerEncoder(
                dec_dim,
                dim_ctx=enc_dim + dim_ctx,
                num_heads=4,
                last_fc=True,
                last_fc_dim_out=input_dim,
                num_layers=self.hparams.dec_num_layers
                if self.hparams.get("dec_num_layers")
                else 3,
            )
        else:
            raise ValueError

    def forward(self, x, beta, context):
        """
        Input:
            x: [B,G,D] intrinsic
            beta: B
            context: [B,G,D2] or [B, D2] condition
        Output:
            eta: [B,G,D]
        """
        # print(f"x: {x.shape} context: {context.shape} beta: {beta.shape}")
        B, G = x.shape[:2]

        if self.hparams.get("use_timestep_embedder"):
            time_emb = self.time_embedder(beta).unsqueeze(1)
        else:
            beta = beta.view(B, 1, 1)
            time_emb = torch.cat(
                [beta, torch.sin(beta), torch.cos(beta)], dim=-1
            )  # [B,1,3]
        ctx = time_emb
        """
        Encoding
        """
        cout = self.context_embedding(context)
        cout = self.encoder(cout, ctx=ctx if self.hparams.encoder_use_time else None)

        if cout.ndim == 2:
            cout = cout.unsqueeze(1).expand(-1, G, -1)

        """
        Decoding
        """
        out = self.query_embedding(x)
        if self.hparams.get("use_pos_encoding"):
            out = self.pos_encoding(out)

        if self.hparams.decoder_type == "transformer_encoder":
            try:
                ctx = ctx.expand(-1, G, -1)
                if cout.ndim == 2:
                    cout = cout.unsqueeze(1)
                cout = cout.expand(-1, G, -1)
                ctx = torch.cat([ctx, cout], -1)
            except Exception as e:
                print(e, G, ctx.shape, cout.shape)
            out = self.decoder(out, ctx=ctx)
        else:
            out = self.decoder(out, cout, ctx=ctx)

        # if hasattr(self, "last_fc"):
        # out = self.last_fc(out)

        if self.hparams.residual:
            out = out + x
        return out
