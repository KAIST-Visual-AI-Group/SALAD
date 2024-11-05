import io
from pathlib import Path
from PIL import Image
from pvd.dataset import batch_split_eigens
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pvd.diffusion.ldm import *


class SingleSALDM(SpaghettiSALDM):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        batch_size=0,
        return_traj=False,
    ):
        x_T = torch.randn([batch_size, 16, 528]).to(self.device)

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
            raise NotImplementedError("Not implemented eig clip for single phase.")
            # traj[0][:, :, 13:16] = torch.clamp_min(
                # traj[0][:, :, 13:16], min=self.hparams["eigen_value_clip_val"]
            # )
        if return_traj:
            return traj
        else:
            return traj[0]

    def validation(self):
        vis_num_shapes = 4
        # ldm_gaus, gt_gaus = self.sampling_gaussians(vis_num_shapes)
        ldm_feats = self.sample(vis_num_shapes)
        ldm_sj, ldm_gaus = ldm_feats.split(split_size=[512,16], dim=-1)
        if self.hparams.get("global_normalization") == "partial":
            ldm_gaus = self._build_dataset("val").unnormalize_global_static(ldm_gaus, slice(12,None))
        elif self.hparams.get("global_normalization") == "all":
            ldm_gaus = self._build_dataset("val").unnormalize_global_static(ldm_gaus, slice(None))
        
        jutils.sysutil.clean_gpu()
        """ Spaghetti Decoding """
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

        ldm_zcs = generate_zc_from_sj_gaus(spaghetti, ldm_sj, ldm_gaus)

        camera_kwargs = dict(
            camPos = np.array([-2,2,-2]),
            camLookat=np.array([0,0,0]),
            camUp=np.array([0,1,0]),
            resolution=(256,256),
            samples=16,
                )

        wandb_logger = self.get_wandb_logger()
        for i in range(vis_num_shapes):
            try:
                vert, face = get_mesh_from_spaghetti(spaghetti, mesher, ldm_zcs[i])
                mesh_img = jutils.visutil.render_mesh(vert, face, **camera_kwargs)
                gaus_img = jutils.visutil.render_gaussians(clip_eigenvalues(ldm_gaus[i]), resolution=camera_kwargs["resolution"])
                img = jutils.imageutil.merge_images([gaus_img, mesh_img])
                wandb_logger.log_image("Pred", [img])
            except Exception as e:
                print(f"Error occured at visualizing, {e}")

class SingleZbSALDM(SpaghettiSALDM):
    def validation(self):
        vis_num_shapes = 4
        ldm_zbs = self.sample(vis_num_shapes)

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

        ldm_sjs, ldm_gmms = spaghetti.decomposition_control.forward_mid(ldm_zbs)
        ldm_gaus = batch_gmms_to_gaus(ldm_gmms)
        ldm_zb_hat = spaghetti.merge_zh_step_a(ldm_sjs, ldm_gmms)
        ldm_zcs, _ = spaghetti.mixing_network.forward_with_attention(ldm_zb_hat)

        camera_kwargs = dict(
            camPos = np.array([-2,2,-2]),
            camLookat=np.array([0,0,0]),
            camUp=np.array([0,1,0]),
            resolution=(256,256),
            samples=16,
                )

        wandb_logger = self.get_wandb_logger()
        for i in range(vis_num_shapes):
            try:
                vert, face = get_mesh_from_spaghetti(spaghetti, mesher, ldm_zcs[i])
                mesh_img = jutils.visutil.render_mesh(vert, face, **camera_kwargs)
                gaus_img = jutils.visutil.render_gaussians(clip_eigenvalues(ldm_gaus[i]), resolution=camera_kwargs["resolution"])
                img = jutils.imageutil.merge_images([gaus_img, mesh_img])
                wandb_logger.log_image("Pred", [img])
            except Exception as e:
                print(f"Error occured at visualizing, {e}")
