import torch
import pytorch_lightning as pl
from pytorch3d.loss import chamfer_distance
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
from pvd.utils.ldm_utils import *
from pvd.dataset import *
import sys
from eval3d import emdModule


def conv_block(in_channels, out_channels, bn=True):
    conv = nn.Conv1d(in_channels, out_channels, 1)
    if not bn:
        return conv
    return nn.Sequential(conv, nn.BatchNorm1d(out_channels))


class PointNet2Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.enc_modules = nn.ModuleList([])
        self.enc_modules.append(
            PointnetSAModuleMSG(
                npoint=512,  # number of output pts
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[3, 32, 64], [3, 64, 64, 128], [3, 64, 96, 128]],
                use_xyz=True,
            )
        )
        input_channels = 64 + 128 + 128

        self.enc_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 64, 64, 128],
                    [input_channels, 64, 64, 128],
                ],
                use_xyz=True,
            )
        )
        input_channels = 128 + 128 + 128
        self.enc_modules.append(PointnetSAModule(mlp=[input_channels, 512, 512]))

    def forward(self, x):
        """
        Input:
            x: [B,N,3]
        Output:
            features: [B,512]
        """
        xyz = x
        features = x.transpose(1, 2).contiguous()

        for module in self.enc_modules:
            xyz, features = module(xyz, features)

        return features.squeeze()


class PointNetEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128)
        self.conv3 = conv_block(128, 1024)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = x.max(-1)[0]
        return x


class PointNet2VEncoder(PointNet2Encoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.to_mu_and_sigma = nn.Linear(512, 512 * 2)

    def forward(self, x):
        """
        Input:
            x: [B,2048,3]
        Output:
            posterior
        """
        feat = super().forward(x)
        feat2 = self.to_mu_and_sigma(feat)
        posterior = DiagonalGaussianDistribution(feat2)

        return posterior


class PointNetVEncoder(PointNetEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.to_mu_and_sigma = nn.Linear(1024, 1024 * 2)

    def forward(self, x):
        feat = super().forward(x)
        feat2 = self.to_mu_and_sigma(feat)
        posterior = DiagonalGaussianDistribution(feat2)
        return posterior


class PointNetDecoder(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.conv1 = conv_block(in_channels, in_channels)
        self.conv2 = conv_block(in_channels, in_channels * 2)
        self.conv3 = conv_block(in_channels * 2, 2048)
        self.conv4 = conv_block(2048, 2048 * 3, bn=False)

    def forward(self, feat):
        """
        Input:
            feat: [B,512]
        Output:
            xyz: [B,2048,3]
        """
        x = feat.unsqueeze(-1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        B = x.shape[0]
        x = x.reshape(B, 2048, 3)
        return x


class PointNetVAE(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        if self.hparams.encoder == "pointnet":
            self.encoder = PointNetVEncoder()
        elif self.hparams.encoder == "pointnet2":
            self.encoder = PointNet2VEncoder()
        else:
            raise AssertionError()

        self.decoder = PointNetDecoder(in_channels=self.get_latent_dim())
        self.emd_criterion = emdModule()

    def forward(self, x, sample_posterior=True):
        posterior = self.encoder(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decoder(z)
        return dec, posterior

    def step(self, batch, batch_idx, stage):
        pts = batch
        pred, posterior = self(pts, self.hparams.kl_weight > 0)

        if self.current_epoch < 100 or stage != "train":
            loss = chamfer_distance(pts, pred)[0]
        else:
            loss = self.emd_criterion(pts, pred, 0.005, 50)[0].mean()
        kl_loss = posterior.kl().mean() * self.hparams.kl_weight
        loss += kl_loss

        self.log(f"{stage}/loss", loss, on_epoch=True, on_step=True, prog_bar=True)

        if stage == "train":
            self.log(
                f"{stage}/kl_loss", kl_loss, on_epoch=True, on_step=True, prog_bar=True
            )

        if stage == "val" and batch_idx == 0:
            wandb_logger = self.get_wandb_logger()
            camPos = np.array([2, 2, -2])
            camUp = np.array([0, 1, 0])
            if wandb_logger is not None:
                vis_gt = jutils.thutil.th2np(pts[:4])
                vis_pred = jutils.thutil.th2np(pred[:4])
                images = []
                for i in range(len(vis_pred)):
                    img_gt = jutils.visutil.render_pointcloud(
                        vis_gt[i],
                        camPos=camPos,
                        camUp=camUp,
                        cloudR=0.04,
                    )
                    img_pred = jutils.visutil.render_pointcloud(
                        vis_pred[i],
                        camPos=camPos,
                        camUp=camUp,
                        cloudR=0.04,
                    )
                    img = jutils.imageutil.merge_images([img_gt, img_pred])
                    images.append(img)
                wandb_logger.log_image("validation_samples", images)

                sample_pred = self.sample(4)
                images = []
                for i in range(len(sample_pred)):
                    img = jutils.visutil.render_pointcloud(
                        sample_pred[i], camPos=camPos, camUp=camUp, cloudR=0.04
                    )
                    images.append(img)
                wandb_logger.log_image("random_samples", images)

        return loss

    def get_latent_dim(self):
        if self.hparams.encoder == "pointnet":
            return 1024
        elif self.hparams.encoder == "pointnet2":
            return 512

    def sample(self, num_samples=None, return_zs=False):
        num_samples = num_samples if num_samples is not None else 1
        z = torch.randn(num_samples, self.get_latent_dim()).to(self.device)
        dec = self.decoder(z)
        if return_zs:
            return dec, z
        return dec

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    @torch.no_grad()
    def save_latents(self, save_dir=None):
        if self.training:
            self.eval()

        model_mode = "train" if self.training else "eval"
        print(f"[*] Current model mode: {model_mode}")
        ds = PointCloudDataset(2048, "train")
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )
        latents = []
        print("[*] Start extracting latents")
        for i, batch in enumerate(dl):
            pts = batch
            pts = pts.to(self.device)
            _, posterior = self(pts, sample_posterior=False)
            z = posterior.mode().detach().cpu().numpy()
            latents.append(z)
        latents = np.concatenate(latents, 0)
        print(f"latents shape: {latents.shape}")
        save_path = (
            f"{self.hparams.save_dir}/{len(latents)}-latents.h5"
            if save_dir is None
            else f"{save_dir}/{len(latents)}-latents.h5"
        )
        with h5py.File(save_path, "w") as f:
            f["data"] = latents.astype(np.float32)
        print(f"[*] Saved latents at {save_path}")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def _build_dataloader(self, stage, batch_size=None, shuffle=None, drop_last=None):
        ds = PointCloudDataset(2048, stage)
        setattr(self, f"data_{stage}", ds)
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size
            if batch_size is not None
            else self.hparams.batch_size,
            shuffle=shuffle if shuffle is not None else stage == "train",
            num_workers=4,
            drop_last=drop_last if drop_last is not None else stage == "train",
        )

    def train_dataloader(self):
        return self._build_dataloader("train")

    def val_dataloader(self):
        return self._build_dataloader("val")

    def test_dataloader(self):
        return self._build_dataloader("test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        # optimizer, milestones=[30, 80], gamma=0.5
        # )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        return [optimizer], [scheduler]

    def get_wandb_logger(self):
        for logger in self.logger:
            if isinstance(logger, pl.loggers.wandb.WandbLogger):
                return logger
        return None
