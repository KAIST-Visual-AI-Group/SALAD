import io
from pathlib import Path
from PIL import Image
from pvd.dataset import batch_split_eigens
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pvd.diffusion.ldm import *
from pvd.utils.spaghetti_utils import *


class GaussianSALDM(SpaghettiSALDM):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        batch_size=0,
        return_traj=False,
    ):
        if self.hparams.get("use_scaled_eigenvectors"):
            x_T = torch.randn([batch_size, 16, 13]).to(self.device)
        else:
            x_T = torch.randn([batch_size, 16, 16]).to(self.device)

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
        if return_traj:
            return traj
        else:
            return traj[0]

    def sampling_gaussians(self, num_shapes):
        """
        Return:
            ldm_gaus: np.ndarray
            gt_gaus: np.ndarray
        """
        ldm_gaus = self.sample(num_shapes)

        if self.hparams.get("use_scaled_eigenvectors"):
            ldm_gaus = jutils.thutil.th2np(batch_split_eigens(jutils.nputil.np2th(ldm_gaus)))

        if self.hparams.get("global_normalization"):
            if not hasattr(self, "data_val"):
                self._build_dataset("val")
            if self.hparams.get("global_normalization") == "partial":
                ldm_gaus = self.data_val.unnormalize_global_static(ldm_gaus, slice(12,None)) 
            elif self.hparams.get("global_normalization") == "all":
                ldm_gaus = self.data_val.unnormalize_global_static(ldm_gaus, slice(None))

        # return ldm_gaus, gt_gaus
        ldm_gaus = clip_eigenvalues(ldm_gaus)
        return ldm_gaus

    def validation(self):
        vis_num_shapes = 16
        # ldm_gaus, gt_gaus = self.sampling_gaussians(vis_num_shapes)
        ldm_gaus = self.sampling_gaussians(vis_num_shapes)

        pred_images = []
        for i in range(vis_num_shapes):

            def draw_gaus(gaus):
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

class ZaSALDM(SpaghettiSALDM):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        batch_size=0,
        return_traj=False,
    ):
        x_T = torch.randn([batch_size, 256]).to(self.device)

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
        if return_traj:
            return traj
        else:
            return traj[0]

    def sampling_za(self, num_shapes):
        """
        Return: torch.Tensor

        """
        # batch_size = self.hparams.batch_size

        # ldm_zas = []
        # print("[*] Start sampling")
        # for i in range(int(np.ceil(num_shapes / batch_size))):
            # batch_ldm_za = self.sample(batch_size)
            # ldm_zas.append(batch_ldm_za)

        # print("[*] Finished sampling")

        # ldm_zas = torch.stack(ldm_zas, 0)
        ldm_zas = self.sample(num_shapes)
        return ldm_zas

    def validation(self):
        vis_num_shapes = 4
        vis_ldm_zas = self.sampling_za(vis_num_shapes)

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

        wandb_logger = self.get_wandb_logger()

        camera_kwargs = dict(
            camPos=np.array([-2, 2, -2]),
            camLookat=np.array([0, 0, 0]),
            camUp=np.array([0, 1, 0]),
            resolution=(256, 256),
            samples=32,
        )

        vis_ldm_zhs, vis_ldm_gmms = spaghetti.decomposition_control(vis_ldm_zas)
        vis_ldm_gaus = jutils.thutil.th2np(batch_gmms_to_gaus(vis_ldm_gmms))
        vis_ldm_zcs, _ = spaghetti.merge_zh(vis_ldm_zhs, vis_ldm_gmms)

        for i in range(vis_num_shapes):
            gaus_img = jutils.visutil.render_gaussians(
                vis_ldm_gaus[i], resolution=camera_kwargs["resolution"]
            )
            mesh = mesher.occ_meshing(
                decoder=get_occ_func(spaghetti, vis_ldm_zcs[i]),
                res=128,
                get_time=False,
                verbose=False,
            )
            try:
                vert, face = list(map(lambda x: jutils.thutil.th2np(x), mesh))
                pred_img = jutils.fresnelvis.renderMeshCloud(
                    mesh={"vert": vert / 2, "face": face}, **camera_kwargs
                )
                pred_img = Image.fromarray(pred_img)
            except:
                pass

            try:
                image = jutils.imageutil.merge_images([gaus_img, pred_img])
                wandb_logger.log_image("visualization", [image])
            except:
                pass
