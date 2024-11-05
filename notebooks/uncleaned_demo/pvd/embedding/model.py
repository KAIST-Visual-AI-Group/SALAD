from pvd.utils.train_utils import PolyDecayScheduler
from tqdm import tqdm
from rich.progress import track
from pvd.vae.vae import conv_block
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from dotmap import DotMap

import jutils

from pvd.dataset import *
from pvd.diffusion.network import *

class ContrastiveLearningDistWrapper:
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
        self.hparams = DotMap(self.__dict__)

    def __call__(self, feat_a, feat_b):
        """
        Input:
            feat_a: [B,D_e]
            feat_b: [B,D_e]
        Output:
            dist_mat: [B,B]
        """
        assert feat_a.dim() == feat_b.dim()
        assert feat_a.shape[0] == feat_b.shape[0]
        if feat_a.dim() == 1:
            feat_a = feat_a.unsqueeze(0)
            feat_b = feat_b.unsqueeze(0)


class TFEmbedder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.hparams = DotMap(self.__dict__)
        
        self.embedding = nn.Linear(self.hparams.dim_in, self.hparams.dim_h)
        dim_h = self.hparams.dim_h
        self.encoder = TimeTransformerEncoder(dim_h, dim_ctx=None, num_heads=self.hparams.num_heads, use_time=False, num_layers=self.hparams.num_layers, last_fc=True, last_fc_dim_out=self.hparams.dim_out)

    def forward(self, x):
        """
        Input:
            x: [B,G,D_in]
        Output:
            out: [B,D_out]
        """
        B, G = x.shape[:2]
        # res = x

        x = self.embedding(x)
        out = self.encoder(x)
        out = out.max(1)[0]
        return out
        


class CLIPEmbedder(pl.LightningModule):
    def __init__(self, ext_embedder, int_embedder, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.ext_embedder = ext_embedder
        self.int_embedder = int_embedder

    def forward(self, intrinsics, extrinsics):
        """
        Input:
            intrinsics: [Bi,16,512]
            extrinsics: [Be,16,16]
        Output:
            cossim_mat: [Bi,Be]
        """

        int_f = self.int_embedder(intrinsics) #[B,D_emb]
        ext_f = self.ext_embedder(extrinsics) #[B,D_emb]
        
        int_e = int_f / int_f.norm(dim=-1, keepdim=True)
        ext_e = ext_f / ext_f.norm(dim=-1, keepdim=True)
        
        sim_mat = torch.einsum("nd,md->nm", int_e, ext_e)
        
        return sim_mat, int_e, ext_e
    
    def info_nce_loss(self, sim_mat):
        """
        Input:
            sim_mat: [N,N]
        """
        assert sim_mat.dim() == 2 and sim_mat.shape[0] == sim_mat.shape[1]
        N = sim_mat.shape[0]
        logits = sim_mat * np.exp(self.hparams.temperature)
        labels = torch.arange(N).to(logits).long()
        loss = F.cross_entropy(logits, labels)
        return loss
        
    def step(self, batch, batch_idx, stage: str):
        intrinsics, extrinsics = batch

        sim_mat, int_e, ext_e = self(intrinsics, extrinsics)
        loss = self.info_nce_loss(sim_mat)
        self.log(f"{stage}/loss", loss, on_step=stage == "train", prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")
    
    def get_intrinsic_extrinsic_distance_matrix(self):
        """
        [num_i, num_e]
        """
        ds = self._build_dataset("val")
        intr = ds.data["s_j_affine"]
        extr = ds.data["g_js_affine"]
        intr, extr = list(map(lambda x: jutils.nputil.np2th(x).to(self.device), [intr, extr]))
        dist_mat = self(intr, extr)[0].cpu().numpy()

        return dist_mat



    def _build_dataset(self, stage):
        if stage == "train":
            ds = SpaghettiLatentDataset(**self.hparams.dataset_kwargs)
        else:
            dataset_kwargs = self.hparams.dataset_kwargs.copy()
            dataset_kwargs["repeat"] = 1
            ds = SpaghettiLatentDataset(**dataset_kwargs)
        setattr(self, f"data_{stage}", ds)
        return ds

    def _build_dataloader(self, stage, batch_size=None):
        try:
            ds = getattr(self, f"data_{stage}")
        except:
            ds = self._build_dataset(stage)

        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size if batch_size is not None else self.hparams.batch_size,
            shuffle=stage == "train",
            drop_last=stage == "train",
            num_workers=4,
        )
    def train_dataloader(self):
        return self._build_dataloader("train")

    def val_dataloader(self):
        return self._build_dataloader("val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = PolyDecayScheduler(optimizer, self.hparams.lr, power=0.999)
        return [optimizer], [scheduler]

