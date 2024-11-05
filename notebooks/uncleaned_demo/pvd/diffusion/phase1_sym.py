import io
from pathlib import Path
from PIL import Image
from pvd.dataset import batch_split_eigens
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pvd.diffusion.ldm import SpaghettiSALDM
import jutils
from pvd.utils.spaghetti_utils import *


class GaussianSymSALDM(SpaghettiSALDM):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    def forward(self, x):
        assert x.shape[1] == 8, f"x.shape: {x.shape}"
        return self.get_loss(x)

    @torch.no_grad()
    def sample(
        self,
        batch_size=0,
        return_traj=False,
    ):
        if self.hparams.get("use_scaled_eigenvectors"):
            x_T = torch.randn([batch_size, 8, 13]).to(self.device)
        else:
            x_T = torch.randn([batch_size, 8, 16]).to(self.device)

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

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()

            traj[t] = traj[t].cpu()

            if not return_traj:
                del traj[t]

        if self.hparams.get("eigen_value_clipping"):
            traj[0][:, :, 13:16] = torch.clamp_min(
                traj[0][:, :, 13:16], min=self.hparams["eigen_value_clip_val"]
            )

        ####
        if return_traj:
            return traj
        else:
            return traj[0]

    def sampling_gaussians(self, num_shapes):
        """
        Return:
            ldm_gaus: np.ndarray
        """
        ldm_gaus = self.sample(num_shapes)

        if self.hparams.get("use_scaled_eigenvectors"):
            ldm_gaus = jutils.thutil.th2np(batch_split_eigens(jutils.nputil.np2th(ldm_gaus)))

        half = ldm_gaus
        full = reflect_and_concat_gmms(half)
        ldm_gaus = full
        if self.hparams.get("global_normalization"):
            if not hasattr(self, "data_val"):
                self._build_dataset("val")
            print(f"[!] Unnormalize samples as {self.hparams.get('global_normalization')}")
            if self.hparams.get("global_normalization") == "partial":
                ldm_gaus = self.data_val.unnormalize_global_static(ldm_gaus, slice(12,None)) 
            elif self.hparams.get("global_normalization") == "all":
                ldm_gaus = self.data_val.unnormalize_global_static(ldm_gaus, slice(None))

        return ldm_gaus

    def validation(self):
        vis_num_shapes = 16
        ldm_gaus = self.sampling_gaussians(vis_num_shapes)

        pred_images = []
        for i in range(vis_num_shapes):

            def draw_gaus(gaus):
                gaus = clip_eigenvalues(gaus)
                img = jutils.visutil.render_gaussians(gaus, resolution=(256, 256))
                return img

            try:
                pred_img = draw_gaus(ldm_gaus[i])
                pred_images.append(pred_img)
            except:
                pass

        wandb_logger = self.get_wandb_logger()

        try:
            pred_images = jutils.imageutil.merge_images(pred_images)
            wandb_logger.log_image("Pred", [pred_images])
        except:
            pass
