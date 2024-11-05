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
from pvd.vae.vae import PointNetVAE
from pvd.diffusion.network import *
from eval3d import Evaluator
from dotmap import DotMap
from pvd.diffusion.common import *
from pvd.diffusion.ldm import *
import jutils


class ReverseConditionSALDM(SpaghettiConditionSALDM):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    def forward(self, x, cond):
        return self.get_loss(x, cond)

    def sample(
        self,
        num_samples_or_sjs,
        return_traj=False,
        return_cond=False
    ):
        if isinstance(num_samples_or_sjs, int):
            batch_size = num_samples_or_sjs
            ds = self._build_dataset("val")
            cond = torch.stack([ds[i][1] for i in range(batch_size)], 0)
        elif isinstance(num_samples_or_sjs, np.ndarray) or isinstance(num_samples_or_sjs, torch.Tensor):
            cond = jutils.nputil.np2th(num_samples_or_sjs)
            if cond.dim() == 2:
                cond = cond[None]
            batch_size = len(cond)
        x_T = torch.randn([batch_size, 16, 16]).to(self.device)
        cond = cond.to(self.device)

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

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()

            traj[t] = traj[t].cpu()

            if not return_traj:
                del traj[t]
        
        if return_traj:
            if return_cond:
                return traj, cond
            return traj
        else:
            if return_cond:
                return traj[0], cond
            return traj[0]

    def validation(self):
        latent_ds = self._build_dataset("val")
        vis_num_shapes = 3
        num_variations = 3
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


        gt_gaus, gt_zs = zip(*[latent_ds[i+3] for i in range(vis_num_shapes)])
        gt_gaus, gt_zs = list(map(lambda x : torch.stack(x), [gt_gaus, gt_zs]))

        gt_zs_repeated = gt_zs.repeat_interleave(num_variations, 0)
        ldm_gaus, ldm_zs = self.sample(gt_zs_repeated, return_cond=True)
        ldm_gaus = clip_eigenvalues(ldm_gaus)
        ldm_zcs = generate_zc_from_sj_gaus(spaghetti, ldm_zs, ldm_gaus)
        gt_zcs = generate_zc_from_sj_gaus(spaghetti, gt_zs, gt_gaus)
        jutils.sysutil.clean_gpu()

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
                    gaus_img = jutils.visutil.render_gaussians(ldm_gaus[i * num_variations + j], resolution=resolution)
                    vert, face = get_mesh_from_spaghetti(spaghetti, mesher, ldm_zcs[i * num_variations + j], res=128)
                    mesh_img = jutils.visutil.render_mesh(vert, face, resolution=resolution)
                    pred_img = jutils.imageutil.merge_images([gaus_img, mesh_img])
                    pred_img = jutils.imageutil.draw_text(pred_img, f"{j}-th predicted gaus", font_size=24)
                    img_per_shape.append(pred_img)
                except Exception as e:
                    print(e)
            
            try:
                image = jutils.imageutil.merge_images(img_per_shape)
                wandb_logger.log_image("visualization", [image])
            except Exception as e:
                print(e)



