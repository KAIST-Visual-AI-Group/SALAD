import io
from typing import Optional
from pathlib import Path
from PIL import Image
from pvd.utils.sde_utils import add_noise, gaus_denoising_tweedie, tweedie_approach
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
from torch.utils.data.dataset import Subset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from pvd.dataset import LatentDataset, SpaghettiLatentDataset, split_eigens
from pvd.utils.train_utils import PolyDecayScheduler, get_dropout_mask
from pvd.utils.spaghetti_utils import *
# from pvd.vae.vae import PointNetVAE
from pvd.diffusion.network import *
from eval3d import Evaluator
from dotmap import DotMap
from pvd.diffusion.common import *
import jutils


class LDM(pl.LightningModule):
    def __init__(
        self,
        network,
        variance_schedule,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = network
        self.var_sched = variance_schedule

    def forward(self, x):
        # t = torch.randint(
            # 0, self.hparams.num_timesteps, (x.shape[0],), device=self.device
        # ).long()
        return self.get_loss(x)

    def step(self, x, stage: str):
        loss = self(x)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        x = batch
        return self.step(x, "train")

    # def validation_step(self, batch, batch_idx):
    # x = batch
    # return self.step(x, "val")

    # def test_step(self, batch, batch_idx):
    # x = batch
    # return self.step(x, "test")

    def add_noise(self, x, t):
        """
        Input:
            x: [B,D] or [B,G,D]
            t: list of size B
        Output:
            x_noisy: [B,D]
            beta: [B]
            e_rand: [B,D]
        """
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1)  # [B,1]
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1)

        e_rand = torch.randn_like(x)
        if e_rand.dim() == 3:
            c0 = c0.unsqueeze(1)
            c1 = c1.unsqueeze(1)

        x_noisy = c0 * x + c1 * e_rand

        return x_noisy, beta, e_rand

    def get_loss(
        self,
        x0,
        t=None,
        noisy_in=False,
        beta_in=None,
        e_rand_in=None,
    ):
        if x0.dim() == 2:
            B, D = x0.shape
        else:
            B, G, D = x0.shape
        if not noisy_in:
            if t is None:
                t = self.var_sched.uniform_sample_t(B)
            x_noisy, beta, e_rand = self.add_noise(x0, t)
        else:
            x_noisy = x0
            beta = beta_in
            e_rand = e_rand_in

        e_theta = self.net(x_noisy, beta=beta)
        self.log(f"e_theta_norm:", e_theta.norm(-1).mean(), prog_bar=True, on_step=True)
        loss = F.mse_loss(e_theta.flatten(), e_rand.flatten(), reduction="mean")
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size=0,
        return_traj=False,
    ):
        if self.hparams.flatten:
            x_T = torch.randn([batch_size, 16 * 512]).to(self.device)
        else:
            x_T = torch.randn([batch_size, 16, 512]).to(self.device)

        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility=0)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]

            beta = self.var_sched.betas[[t] * batch_size]
            e_theta = self.net(x_t, beta=beta)
            # print(e_theta.norm(-1).mean())

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()

            traj[t] = traj[t].cpu()

            if not return_traj:
                del traj[t]
        if return_traj:
            return traj
        else:
            return traj[0]

    def validation_step(self, batch, batch_idx):
        return

    def validation_epoch_end(self, outputs):
        if self.hparams.no_run_validation:
            return
        if not self.trainer.sanity_checking:
            if (self.current_epoch) % self.hparams.validation_step == 0:
                self.validation()

    def validation(self):
        batch_size = self.hparams.batch_size
        vae = self.load_vae()
        pc_dataloader = vae._build_dataloader(
            stage="train", batch_size=batch_size, shuffle=False
        )
        pc_data_iter = iter(pc_dataloader)
        pc_dataset = vae.data_train
        latent_data_iter = iter(self.val_dataloader())

        num_shapes_for_eval = 256
        num_shapes_for_visualization = 8

        vae_samples = []
        ldm_samples = []
        gt_samples = []

        vae_zs = []
        ldm_zs = []
        gt_zs = []
        print("[*] Start sampling")
        for i in range(int(np.ceil(num_shapes_for_eval / batch_size))):
            batch_vae_samples, batch_vae_zs = list(
                map(
                    lambda x: jutils.thutil.th2np(x),
                    vae.sample(batch_size, return_zs=True),
                )
            )

            batch_ldm_zs = self.sample(batch_size)
            batch_ldm_samples = jutils.thutil.th2np(vae.decoder(batch_ldm_zs))

            batch_gt_samples = jutils.thutil.th2np(next(pc_data_iter))
            batch_gt_zs = jutils.thutil.th2np(next(latent_data_iter))

            vae_samples.append(batch_vae_samples)
            ldm_samples.append(batch_ldm_samples)
            gt_samples.append(batch_gt_samples)

            vae_zs.append(batch_vae_zs)
            ldm_zs.append(jutils.thutil.th2np(batch_ldm_zs))
            gt_zs.append(batch_gt_zs)
        print("[*] Finished sampling")

        vae_samples, ldm_samples, gt_samples = list(
            map(
                lambda x: np.concatenate(x)[:num_shapes_for_eval],
                [vae_samples, ldm_samples, gt_samples],
            )
        )
        vae_zs, ldm_zs, gt_zs = list(
            map(
                lambda x: np.concatenate(x)[:num_shapes_for_eval],
                [vae_zs, ldm_zs, gt_zs],
            )
        )

        assert (
            len(vae_samples)
            == len(ldm_samples)
            == len(gt_samples)
            == len(vae_zs)
            == len(ldm_zs)
            == len(gt_zs)
            == num_shapes_for_eval
        )

        """ Log visualization """
        wandb_logger = self.get_wandb_logger()
        camPos = np.array([2, 2, -2])
        camUp = np.array([0, 1, 0])

        vae_images = []
        ldm_images = []
        for i in range(num_shapes_for_visualization):
            vae_images.append(
                jutils.visutil.render_pointcloud(
                    vae_samples[i], camPos=camPos, camUp=camUp, cloudR=0.04
                )
            )
            ldm_images.append(
                jutils.visutil.render_pointcloud(
                    ldm_samples[i], camPos=camPos, camUp=camUp, cloudR=0.04
                )
            )
        wandb_logger.log_image("vae vis", vae_images)
        wandb_logger.log_image("ldm_vis", ldm_images)
        """================="""

        """ Run evaluations """
        evaluator = Evaluator(
            gt_set=gt_samples, pred_set=vae_samples, batch_size=128, device=self.device
        )
        vae_res = evaluator.compute_all_metrics(verbose=False)
        vae_res = {f"{k}/vae": v for k, v in vae_res.items()}
        self.log_dict(vae_res, prog_bar=True)

        evaluator.update_pred(ldm_samples)
        ldm_res = evaluator.compute_all_metrics(verbose=False)
        ldm_res = {f"{k}/ldm": v for k, v in ldm_res.items()}
        self.log_dict(ldm_res, prog_bar=True)
        """==============="""

        """ Run latent evaluations """
        gt_zs = gt_zs[:, None]
        vae_zs = vae_zs[:, None]
        ldm_zs = ldm_zs[:, None]
        evaluator = Evaluator(
            gt_set=gt_zs, pred_set=vae_zs, batch_size=128, device=self.device
        )
        vae_latent_res = evaluator.compute_all_metrics(verbose=False)
        vae_latent_res = {f"{k}/latent-vae": v for k, v in vae_latent_res.items()}
        self.log_dict(vae_latent_res, prog_bar=True)

        evaluator.update_pred(ldm_zs)
        ldm_latent_res = evaluator.compute_all_metrics(verbose=False)
        ldm_latent_res = {f"{k}/latent-ldm": v for k, v in ldm_latent_res.items()}
        self.log_dict(ldm_latent_res, prog_bar=True)
        """========================"""

    def _build_dataset(self, stage):
        if not Path(self.hparams.latent_path).is_file():
            vae = self.load_vae()
            vae.save_latents()
            del vae
            jutils.sysutil.clean_gpu()

        if stage == "train":
            ds = LatentDataset(
                data_path=self.hparams.latent_path,
                repeat=self.hparams.get("data_repeat"),
            )
        else:
            ds = LatentDataset(data_path=self.hparams.latent_path)
        setattr(self, f"data_{stage}", ds)
        return ds

    def _build_dataloader(self, stage):
        try:
            ds = getattr(self, f"data_{stage}")
        except:
            ds = self._build_dataset(stage)

        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=stage == "train",
            drop_last=stage == "train",
            num_workers=4,
        )

    def train_dataloader(self):
        return self._build_dataloader("train")

    def val_dataloader(self):
        return self._build_dataloader("val")

    def test_dataloader(self):
        return self._build_dataloader("test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = PolyDecayScheduler(optimizer, self.hparams.lr, power=0.999)
        return [optimizer], [scheduler]

    def load_vae(self, to_self_device=True):
        ckpt_path = Path(self.hparams.latent_path).parent / f"checkpoints/last.ckpt"
        vae = PointNetVAE.load_from_checkpoint(ckpt_path)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        if to_self_device:
            vae = vae.to(self.device)
        self.vae = vae
        return vae

    def get_wandb_logger(self):
        if isinstance(self.logger, list):
            for logger in self.logger:
                if isinstance(logger, pl.loggers.wandb.WandbLogger):
                    return logger
        else:
            if isinstance(self.logger, pl.loggers.wandb.WandbLogger):
                return self.logger
        return None


class SpaghettiLDM(LDM):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)
        # self.spaghetti = load_spaghetti(self.device, self.hparams.spaghetti_tag if self.hparams.get("spaghetti_tag") else "chairs_large")
        # self.mesher = load_mesher(self.device)

    def validation(self):
        batch_size = self.hparams.batch_size
        latent_data_iter = iter(self.val_dataloader())

        num_shapes_for_eval = 512

        ldm_zs = []
        gt_zs = []
        print("[*] Start sampling")
        for i in range(int(np.ceil(num_shapes_for_eval / batch_size))):
            batch_ldm_zs = self.sample(batch_size)
            batch_gt_zs = jutils.thutil.th2np(next(latent_data_iter))

            ldm_zs.append(jutils.thutil.th2np(batch_ldm_zs))
            gt_zs.append(batch_gt_zs)

        print("[*] Finish sampling")

        ldm_zs, gt_zs = list(
            map(lambda x: np.concatenate(x)[:num_shapes_for_eval], [ldm_zs, gt_zs])
        )

        self.log(
            "GT norm",
            np.linalg.norm(gt_zs.reshape(num_shapes_for_eval, -1), axis=1).mean(),
        )
        self.log(
            "LDM norm",
            np.linalg.norm(ldm_zs.reshape(num_shapes_for_eval, -1), axis=1).mean(),
        )
        """ Run latent evaluation """

        ldm_zs = ldm_zs[:, None]
        gt_zs = gt_zs[:, None]
        evaluator = Evaluator(
            gt_set=gt_zs, pred_set=ldm_zs, batch_size=128, device=self.device
        )

        latent_res = evaluator.compute_all_metrics(
            verbose=False, return_distance_matrix=True
        )
        latent_log_res = {
            k: v for k, v in latent_res.items() if "distance_matrix" not in k
        }
        self.log_dict(latent_log_res, prog_bar=True)

        Mxx, Mxy, Myy = (
            latent_res["distance_matrix_xx"],
            latent_res["distance_matrix_xy"],
            latent_res["distance_matrix_yy"],
        )

        d_prime, min_idx = torch.tensor(Mxy).topk(k=1, dim=0, largest=False)  # num_pred
        d_prime = d_prime[0]
        min_idx = min_idx[0]  # [num_preds]

        closest_gt_zs = gt_zs[min_idx]
        d_mat = evaluator.compute_chamfer_distance(
            closest_gt_zs, gt_zs
        )  # [num_preds, num_gt]
        d, min_idx2 = torch.tensor(d_mat).topk(k=2, dim=1, largest=False)
        min_idx2 = min_idx2[:, 1]
        assert torch.all(min_idx != min_idx2), f"{d_mat}, {min_idx}, {min_idx2}"
        d = d[:, 1]
        assert torch.all(d > 0), f"d cannot be zero, {d}."

        rel_dist = d_prime.flatten() / d.flatten()
        self.log("rel_dist", rel_dist.mean(), prog_bar=True)

        """ ===================== """

        """ tSNE """
        if self.current_epoch % 10 == 0:
            eval_zs = np.concatenate([gt_zs[:, 0], ldm_zs[:, 0]], 0)
            tsne = TSNE(n_iter=500, verbose=2)
            embeddings = tsne.fit_transform(eval_zs)
            gt_embs, ldm_embs = np.split(embeddings, 2)

            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.scatter(gt_embs[:, 0], gt_embs[:, 1], label="GT", alpha=0.8, c="red")
            ax.scatter(
                ldm_embs[:, 0], ldm_embs[:, 1], label="LDM", alpha=0.8, c="skyblue"
            )
            ax.legend()

            buf = io.BytesIO()
            plt.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)

            wandb_logger = self.get_wandb_logger()
            wandb_logger.log_image("tSNE", [img])
        """ ==== """

    def _build_dataset(self, stage):
        if hasattr(self, f"data_{stage}"):
            return getattr(self, f"data_{stage}")
        if stage == "train":
            ds = SpaghettiLatentDataset(**self.hparams.dataset_kwargs)
        else:
            dataset_kwargs = self.hparams.dataset_kwargs.copy()
            dataset_kwargs["repeat"] = 1
            ds = SpaghettiLatentDataset(**dataset_kwargs)
        setattr(self, f"data_{stage}", ds)
        return ds


class SpaghettiSALDM(SpaghettiLDM):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        batch_size=0,
        return_traj=False,
    ):
        x_T = torch.randn([batch_size, 16, 512]).to(self.device)

        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility=0)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]

            beta = self.var_sched.betas[[t] * batch_size]
            e_theta = self.net(x_t, beta=beta)
            # print(e_theta.norm(-1).mean())

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()

            traj[t] = traj[t].cpu()

            if not return_traj:
                del traj[t]
        if return_traj:
            return traj
        else:
            return traj[0]

    def validation(self):
        batch_size = self.hparams.batch_size
        latent_data_iter = iter(self.val_dataloader())

        num_shapes_for_eval = 512

        ldm_zs = []
        gt_zs = []
        print("[*] Start sampling")
        for i in range(int(np.ceil(num_shapes_for_eval / batch_size))):
            batch_ldm_zs = self.sample(batch_size)
            batch_gt_zs = jutils.thutil.th2np(next(latent_data_iter))

            ldm_zs.append(jutils.thutil.th2np(batch_ldm_zs))
            gt_zs.append(batch_gt_zs)

        print("[*] Finish sampling")

        ldm_zs, gt_zs = list(
            map(lambda x: np.concatenate(x)[:num_shapes_for_eval], [ldm_zs, gt_zs])
        )

        self.log(
            "GT norm",
            np.linalg.norm(gt_zs, axis=-1).mean(),
        )
        self.log(
            "LDM norm",
            np.linalg.norm(ldm_zs, axis=-1).mean(),
        )
        """ Run latent evaluation """

        evaluator = Evaluator(
            gt_set=gt_zs,
            pred_set=ldm_zs,
            batch_size=128,
            device=self.device,
            metric="l2",
        )

        latent_res = evaluator.compute_all_metrics(
            verbose=False, return_distance_matrix=True
        )
        latent_log_res = {
            k: v for k, v in latent_res.items() if "distance_matrix" not in k
        }
        self.log_dict(latent_log_res, prog_bar=True)

        Mxx, Mxy, Myy = (
            latent_res["distance_matrix_xx"],
            latent_res["distance_matrix_xy"],
            latent_res["distance_matrix_yy"],
        )

        d_prime, min_idx = torch.tensor(Mxy).topk(k=1, dim=0, largest=False)  # num_pred
        d_prime = d_prime[0]
        min_idx = min_idx[0]  # [num_preds]

        closest_gt_zs = gt_zs[min_idx]
        d_mat = evaluator.compute_pairwise_distance(
            closest_gt_zs, gt_zs
        )  # [num_preds, num_gt]
        d, min_idx2 = torch.tensor(d_mat).topk(k=2, dim=1, largest=False)
        min_idx2 = min_idx2[:, 1]
        assert torch.all(min_idx != min_idx2), f"{d_mat}, {min_idx}, {min_idx2}"
        d = d[:, 1]
        assert torch.all(d > 0), f"d cannot be zero, {d}."

        rel_dist = d_prime.flatten() / d.flatten()
        self.log("rel_dist", rel_dist.mean(), prog_bar=True)

        """ ===================== """

        """ tSNE """
        if self.hparams.get("draw_tsne"):
            tsne_num_shapes = slice(None)
            tsne_gt_zs = gt_zs.reshape(-1, 512)[tsne_num_shapes]
            tsne_ldm_zs = ldm_zs.reshape(-1, 512)[tsne_num_shapes]
            if self.current_epoch % 10 == 0:
                eval_zs = np.concatenate([tsne_gt_zs, tsne_ldm_zs], 0)
                tsne = TSNE(n_iter=500, verbose=2)
                embeddings = tsne.fit_transform(eval_zs)
                gt_embs, ldm_embs = np.split(embeddings, 2)

                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.scatter(gt_embs[:, 0], gt_embs[:, 1], label="GT", alpha=0.6, c="red")
                ax.scatter(
                    ldm_embs[:, 0], ldm_embs[:, 1], label="LDM", alpha=0.6, c="skyblue"
                )
                ax.legend()

                buf = io.BytesIO()
                plt.savefig(buf)
                buf.seek(0)
                img = Image.open(buf)

                wandb_logger = self.get_wandb_logger()
                wandb_logger.log_image("tSNE", [img])
        """ ==== """


class SpaghettiConditionSALDM(SpaghettiSALDM):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)
        if self.hparams.get("augment_condition") == "tweedie":
            if self.hparams.get("phase1_ckpt_path"):
                ckpt_path = self.hparams.phase1_ckpt_path
            else:
                ckpt_path = "/home/juil/pvddir/results/phase1-sym/final_0212/0212_125142/checkpoints/last.ckpt"
            
            if "phase1-sym" in str(ckpt_path):
                from pvd.diffusion.phase1_sym import GaussianSymSALDM
                phase1_class = GaussianSymSALDM
            else:
                from pvd.diffusion.phase1 import GaussianSALDM
                phase1_class = GaussianSALDM
            self.phase1_model = phase1_class.load_from_checkpoint(
                ckpt_path, strict=False
                    ).eval()
            for p in self.phase1_model.parameters(): p.requires_grad_(False)
            if self.phase1_model.hparams.get("global_normalization"):
                self.global_gaus_mean = self.phase1_model._build_dataset("val").global_mean
                self.global_gaus_std = self.phase1_model._build_dataset("val").global_std
            else:
                self.global_gaus_mean = None
                self.global_gaus_std = None

    def forward(self, x, cond):
        if self.hparams.get("classifier_free_guidance") and self.training:
            B = cond.shape[0]
            if self.hparams.conditioning_dropout_level == "shape":
                mask_shape = (B,) + (1,) * (cond.dim() - 1)  # masking in a shape level
            elif self.hparams.conditioning_dropout_level == "part":
                mask_shape = (B, cond.shape[1]) + (1,) * (
                    cond.dim() - 2
                )  # masking in a part level
            else:
                raise AssertionError()
            random_dp_mask = get_dropout_mask(
                mask_shape, self.hparams.conditioning_dropout_prob, self.device
            )
            cond = cond * random_dp_mask

        # t = torch.randint(
            # 0, self.hparams.num_timesteps, (x.shape[0],), device=self.device
        # ).long()
        # t = self.var_sched.uniform_sample_t(B)
        if self.hparams.get("augment_condition") and self.training:
            if self.hparams.get("augment_condition") == "forward":
                B = cond.shape[0]
                t = np.random.choice(np.arange(1, self.hparams.augment_timestep+1), B).tolist()
                cond_noisy, beta, e_rand = self.add_noise(cond, t)
                cond = cond_noisy
            elif self.hparams.get("augment_condition") == "tweedie":
                cond_tweedie = gaus_denoising_tweedie(self.phase1_model, cond, self.hparams.augment_timestep, mean=self.global_gaus_mean, std=self.global_gaus_std)
                cond = cond_tweedie

        return self.get_loss(x, cond)

    def step(self, batch, stage: str):
        x, cond = batch
        # loss = self.get_loss(x, cond)
        loss = self(x, cond)
        self.log(f"{stage}/loss", loss, on_step=stage == "train", prog_bar=True)
        return loss

    def get_loss(self, x0, cond, t=None, noisy_in=False, beta_in=None, e_rand_in=None):
        if x0.dim() == 2:
            B, D = x0.shape
        elif x0.dim() == 3:
            B, G, D = x0.shape
        else:
            raise AssertionError(f"x dim should be 2 or 3. {x0.shape}")

        if not noisy_in:
            if t is None:
                t = self.var_sched.uniform_sample_t(B)
            x_noisy, beta, e_rand = self.add_noise(x0, t)
        else:
            x_noisy = x0
            beta = beta_in
            e_rand = e_rand_in

        e_theta = self.net(x_noisy, beta, cond)
        self.log(f"e_theta_norm:", e_theta.norm(-1).mean(), prog_bar=True, on_step=True)
        loss = F.mse_loss(e_theta.flatten(), e_rand.flatten(), reduction="mean")
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples_or_gaus,
        return_traj=False,
        classifier_free_guidance=None,
        free_guidance_weight=-0.7,
        augment_condition_in_test=False,
        return_cond=False,
    ):
        classifier_free_guidance = (
            self.hparams.get("classifier_free_guidance")
            if classifier_free_guidance is None
            else classifier_free_guidance
        )
        if isinstance(num_samples_or_gaus, int):
            batch_size = num_samples_or_gaus
            ds = self._build_dataset("val")
            cond = torch.stack([ds[i][1] for i in range(batch_size)], 0)

        elif isinstance(num_samples_or_gaus, np.ndarray) or isinstance(
            num_samples_or_gaus, torch.Tensor
        ):
            cond = jutils.nputil.np2th(num_samples_or_gaus)
            if cond.dim() == 2:
                cond = cond[None]
            batch_size = len(cond)
            
            #### < Seungwoo >
            # Detect the number of parts in the given GMM condition
            # and use it when initializing the per-part intrinsics.
            n_part = cond.shape[1]
            ####

        #### < Seungwoo >
        # Refer to the description above
        # x_T = torch.randn([batch_size, 16, 512]).to(self.device)
        x_T = torch.randn([batch_size, n_part, 512]).to(self.device)
        ####
        cond = cond.to(self.device)

        if augment_condition_in_test:
            assert self.hparams.get("augment_condition")
            # print(f"[!] Augment condition in test time as [{self.hparams.get('augment_condition')}]")
            if self.hparams.get("augment_condition") == "forward":
                B = cond.shape[0]
                t = np.random.choice(np.arange(1, self.hparams.augment_timestep+1), B).tolist()
                cond_noisy, beta, e_rand = self.add_noise(cond, t)
                cond = cond_noisy
            elif self.hparams.get("augment_condition") == "tweedie":
                # cond_tweedie = gaus_denoising_tweedie(self.phase1_model, cond, self.hparams.augment_timestep)
                cond_tweedie = gaus_denoising_tweedie(self.phase1_model, cond, self.hparams.augment_timestep, mean=self.global_gaus_mean, std=self.global_gaus_std)
                cond = cond_tweedie

        # classifier free guidance sampling.
        if classifier_free_guidance:
            null_cond = torch.zeros_like(cond)

        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility=0)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]

            beta = self.var_sched.betas[[t] * batch_size]
            e_theta = self.net(x_t, beta=beta, context=cond)

            if classifier_free_guidance:
                null_e_theta = self.net(x_t, beta=beta, context=null_cond)
                w = free_guidance_weight
                e_theta = (1 + w) * e_theta - w * null_e_theta

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()

            traj[t] = traj[t].cpu()

            if not return_traj:
                del traj[t]

        if self.hparams.get("sj_global_normalization"):
            # print("[*] Unnormalize intrinsics.")
            if not hasattr(self, "data_val"): self._build_dataset("val")
            out = jutils.thutil.th2np(traj[0])
            out = self.data_val.unnormalize_sj_global_static(traj[0])
            traj[0] = jutils.nputil.np2th(out).to(self.device)

        if return_traj:
            if return_cond:
                return traj, cond
            return traj
        else:
            if return_cond:
                return traj[0], cond
            return traj[0]

    def validation(self):
        # batch_size = self.hparams.batch_size
        # latent_data_iter = iter(self.val_dataloader())
        latent_ds = self._build_dataset("val")
        vis_num_shapes = 3
        num_variations = 3
        num_shapes_for_eval = vis_num_shapes
        # num_shapes_for_eval = 512
        jutils.sysutil.clean_gpu()

        if not hasattr(self, "spaghetti"):
            spaghetti = load_spaghetti(self.device, self.hparams.spaghetti_tag if self.hparams.get("spaghetti_tag") else "chairs_large")
            self.spaghetti = spaghetti
        else:
            spaghetti = self.spaghetti

        if not hasattr(self, "mesher"):
            mesher = load_mesher(self.device)
            self.mesher = mesher
        else:
            mesher = self.mesher


        """======== Sampling ========"""
        ldm_zs = []
        gt_zs = []
        gt_gaus = []
        # print("[*] Start sampling")
       
        gt_zs, gt_gaus = zip(*[latent_ds[i+3] for i in range(vis_num_shapes)])
        gt_zs, gt_gaus = list(map(lambda x : torch.stack(x), [gt_zs, gt_gaus]))
        if self.hparams.get("sj_global_normalization"):
            gt_zs = jutils.thutil.th2np(gt_zs)
            gt_zs = latent_ds.unnormalize_sj_global_static(gt_zs)
            gt_zs = jutils.nputil.np2th(gt_zs).to(self.device)

        gt_gaus_repeated = gt_gaus.repeat_interleave(num_variations, 0)
        clean_ldm_zs, clean_gaus = self.sample(gt_gaus_repeated, return_cond=True)
        clean_gaus = project_eigenvectors(clip_eigenvalues(clean_gaus))
        clean_zcs = generate_zc_from_sj_gaus(spaghetti, clean_ldm_zs, clean_gaus)
        gt_zcs = generate_zc_from_sj_gaus(spaghetti, gt_zs, gt_gaus)
        jutils.sysutil.clean_gpu()

        if self.hparams.get("augment_condition"):
            noisy_ldm_zs, noisy_gaus = self.sample(gt_gaus_repeated, augment_condition_in_test=True, return_cond=True)
            noisy_gaus = project_eigenvectors(clip_eigenvalues(noisy_gaus))
            noisy_zcs = generate_zc_from_sj_gaus(spaghetti, noisy_ldm_zs, noisy_gaus)

        # print("[*] Finish sampling")
        """=========================="""

        """ Spaghetti Decoding """

        wandb_logger = self.get_wandb_logger()
        resolution = (256,256)
        for i in range(vis_num_shapes):
            img_per_shape = []
            gaus_img = jutils.visutil.render_gaussians(gt_gaus[i], resolution=resolution)
            vert, face = get_mesh_from_spaghetti(spaghetti, mesher, gt_zcs[i], res=128)
            gt_mesh_img = jutils.visutil.render_mesh(vert, face, resolution=resolution)
            gt_img = jutils.imageutil.merge_images([gaus_img, gt_mesh_img])
            gt_img = jutils.imageutil.draw_text(gt_img, "GT", font_size=24)
            img_per_shape.append(gt_img)
            for j in range(num_variations):
                try:
                    gaus_img = jutils.visutil.render_gaussians(clean_gaus[i * num_variations + j], resolution=resolution)
                    vert, face = get_mesh_from_spaghetti(spaghetti, mesher, clean_zcs[i * num_variations + j], res=128)
                    mesh_img = jutils.visutil.render_mesh(vert, face, resolution=resolution)
                    pred_img = jutils.imageutil.merge_images([gaus_img, mesh_img])
                    pred_img = jutils.imageutil.draw_text(pred_img, f"{j}-th clean gaus", font_size=24)
                    img_per_shape.append(pred_img)
                except Exception as e:
                    print(e)
            
            if self.hparams.get("augment_condition"):
                for j in range(num_variations):
                    try:
                        gaus_img = jutils.visutil.render_gaussians(noisy_gaus[i * num_variations + j], resolution=resolution)
                        vert, face = get_mesh_from_spaghetti(spaghetti, mesher, noisy_zcs[i * num_variations + j], res=128)
                        mesh_img = jutils.visutil.render_mesh(vert, face, resolution=resolution)
                        pred_img = jutils.imageutil.merge_images([gaus_img, mesh_img])
                        pred_img = jutils.imageutil.draw_text(pred_img, f"{j}-th noisy gaus", font_size=24)
                        img_per_shape.append(pred_img)
                    except Exception as e:
                        print(e)
                       
            try:
                image = jutils.imageutil.merge_images(img_per_shape)
                wandb_logger.log_image("visualization", [image])
            except Exception as e:
                print(e)

        """ ================== """

        """ Run latent evaluation """
        # self.log(
            # "GT norm",
            # np.linalg.norm(gt_zs, axis=-1).mean(),
        # )
        # self.log(
            # "LDM norm",
            # np.linalg.norm(ldm_zs, axis=-1).mean(),
        # )
        # evaluator = Evaluator(
            # gt_set=gt_zs,
            # pred_set=ldm_zs,
            # batch_size=128,
            # device=self.device,
            # metric="l2",
        # )

        # latent_res = evaluator.compute_all_metrics(
            # verbose=False, return_distance_matrix=True, compute_jsd_together=False
        # )
        # latent_log_res = {
            # k: v for k, v in latent_res.items() if "distance_matrix" not in k
        # }
        # self.log_dict(latent_log_res, prog_bar=True)

        # Mxx, Mxy, Myy = (
            # latent_res["distance_matrix_xx"],
            # latent_res["distance_matrix_xy"],
            # latent_res["distance_matrix_yy"],
        # )

        # d_prime, min_idx = torch.tensor(Mxy).topk(k=1, dim=0, largest=False)  # num_pred
        # d_prime = d_prime[0]
        # min_idx = min_idx[0]  # [num_preds]

        # closest_gt_zs = gt_zs[min_idx]
        # d_mat = evaluator.compute_pairwise_distance(
            # closest_gt_zs, gt_zs
        # )  # [num_preds, num_gt]
        # d, min_idx2 = torch.tensor(d_mat).topk(k=2, dim=1, largest=False)
        # min_idx2 = min_idx2[:, 1]
        # # assert torch.all(min_idx != min_idx2), f"{d_mat}, {min_idx}, {min_idx2}"
        # d = d[:, 1]
        # # assert torch.all(d > 0), f"d cannot be zero, {d}."

        # rel_dist = d_prime.flatten() / d.flatten()
        # self.log("rel_dist", rel_dist.mean(), prog_bar=True)

        """ ===================== """

        """ tSNE """
        # if self.hparams.get("draw_tsne"):
            # tsne_num_shapes = slice(None)  # slice(1024)
            # tsne_gt_zs = gt_zs.reshape(-1, 512)[tsne_num_shapes]
            # tsne_ldm_zs = ldm_zs.reshape(-1, 512)[tsne_num_shapes]
            # if self.current_epoch % 10 == 0:
                # eval_zs = np.concatenate([tsne_gt_zs, tsne_ldm_zs], 0)
                # tsne = TSNE(n_iter=500, verbose=2)
                # embeddings = tsne.fit_transform(eval_zs)
                # gt_embs, ldm_embs = np.split(embeddings, 2)

                # fig = plt.figure(figsize=(10, 10))
                # ax = fig.add_subplot(111)
                # ax.scatter(gt_embs[:, 0], gt_embs[:, 1], label="GT", alpha=0.6, c="red")
                # ax.scatter(
                    # ldm_embs[:, 0], ldm_embs[:, 1], label="LDM", alpha=0.6, c="skyblue"
                # )
                # ax.legend()

                # buf = io.BytesIO()
                # plt.savefig(buf)
                # buf.seek(0)
                # img = Image.open(buf)

                # wandb_logger = self.get_wandb_logger()
                # wandb_logger.log_image("tSNE", [img])
        """ ==== """


