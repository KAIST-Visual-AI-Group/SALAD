from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from salad.models.base_model import BaseModel
from salad.utils import imageutil, nputil, sysutil, thutil, visutil
from salad.utils.spaghetti_util import (clip_eigenvalues,
                                        generate_zc_from_sj_gaus,
                                        get_mesh_from_spaghetti, load_mesher,
                                        load_spaghetti, project_eigenvectors)


class Phase2Model(BaseModel):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    def forward(self, x, cond):
        return self.get_loss(x, cond)

    def step(self, batch, stage: str):
        x, cond = batch
        loss = self(x, cond)
        self.log(f"{stage}/loss", loss, on_step=stage == "train", prog_bar=True)
        return loss

    def get_loss(self, x0, cond, t=None, noisy_in=False, beta_in=None, e_rand_in=None):
        B, G, D = x0.shape

        if not noisy_in:
            if t is None:
                t = self.var_sched.uniform_sample_t(B)
            x_noisy, beta, e_rand = self.add_noise(x0, t)
        else:
            x_noisy = x0
            beta = beta_in
            e_rand = e_rand_in

        e_theta = self.net(x_noisy, beta, cond)
        loss = F.mse_loss(e_theta.flatten(), e_rand.flatten(), reduction="mean")
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples_or_gaus: Union[torch.Tensor, np.ndarray, int],
        return_traj=False,
        classifier_free_guidance=None,
        free_guidance_weight=-0.7,
        augment_condition_in_test=False,
        return_cond=False,
    ):
        if isinstance(num_samples_or_gaus, int):
            batch_size = num_samples_or_gaus
            ds = self._build_dataset("val")
            cond = torch.stack([ds[i][1] for i in range(batch_size)], 0)

        elif isinstance(num_samples_or_gaus, np.ndarray) or isinstance(
            num_samples_or_gaus, torch.Tensor
        ):
            cond = nputil.np2th(num_samples_or_gaus)
            if cond.dim() == 2:
                cond = cond[None]
            batch_size = len(cond)
        else:
            raise ValueError(
                "'num_samples_or_gaus' should be int, torch.Tensor or np.ndarray."
            )

        x_T = torch.randn([batch_size, 16, 512]).to(self.device)
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
        sysutil.clean_gpu()

        if not hasattr(self, "spaghetti"):
            spaghetti = load_spaghetti(
                self.device,
                self.hparams.spaghetti_tag
                if self.hparams.get("spaghetti_tag")
                else "chairs_large",
            )
            self.spaghetti = spaghetti
        else:
            spaghetti = self.spaghetti

        if not hasattr(self, "mesher"):
            mesher = load_mesher(self.device)
            self.mesher = mesher
        else:
            mesher = self.mesher

        """======== Sampling ========"""
        gt_zs = []
        gt_gaus = []

        gt_zs, gt_gaus = zip(*[latent_ds[i + 3] for i in range(vis_num_shapes)])
        gt_zs, gt_gaus = list(map(lambda x: torch.stack(x), [gt_zs, gt_gaus]))
        if self.hparams.get("sj_global_normalization"):
            gt_zs = thutil.th2np(gt_zs)
            gt_zs = latent_ds.unnormalize_sj_global_static(gt_zs)
            gt_zs = nputil.np2th(gt_zs).to(self.device)

        gt_gaus_repeated = gt_gaus.repeat_interleave(num_variations, 0)
        clean_ldm_zs, clean_gaus = self.sample(gt_gaus_repeated, return_cond=True)
        clean_gaus = project_eigenvectors(clip_eigenvalues(clean_gaus))
        clean_zcs = generate_zc_from_sj_gaus(spaghetti, clean_ldm_zs, clean_gaus)
        gt_zcs = generate_zc_from_sj_gaus(spaghetti, gt_zs, gt_gaus)
        sysutil.clean_gpu()

        """=========================="""

        """ Spaghetti Decoding """
        wandb_logger = self.get_wandb_logger()
        resolution = (256, 256)
        for i in range(vis_num_shapes):
            img_per_shape = []
            gaus_img = visutil.render_gaussians(gt_gaus[i], resolution=resolution)
            vert, face = get_mesh_from_spaghetti(spaghetti, mesher, gt_zcs[i], res=128)
            gt_mesh_img = visutil.render_mesh(vert, face, resolution=resolution)
            gt_img = imageutil.merge_images([gaus_img, gt_mesh_img])
            gt_img = imageutil.draw_text(gt_img, "GT", font_size=24)
            img_per_shape.append(gt_img)
            for j in range(num_variations):
                try:
                    gaus_img = visutil.render_gaussians(
                        clean_gaus[i * num_variations + j], resolution=resolution
                    )
                    vert, face = get_mesh_from_spaghetti(
                        spaghetti, mesher, clean_zcs[i * num_variations + j], res=128
                    )
                    mesh_img = visutil.render_mesh(vert, face, resolution=resolution)
                    pred_img = imageutil.merge_images([gaus_img, mesh_img])
                    pred_img = imageutil.draw_text(
                        pred_img, f"{j}-th clean gaus", font_size=24
                    )
                    img_per_shape.append(pred_img)
                except Exception as e:
                    print(e)

            try:
                image = imageutil.merge_images(img_per_shape)
                wandb_logger.log_image("visualization", [image])
            except Exception as e:
                print(e)

        """ ================== """
