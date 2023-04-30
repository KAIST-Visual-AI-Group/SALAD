import torch
import numpy as np
from salad.models.base_model import BaseModel
from salad.utils import nputil, thutil
from salad.utils.spaghetti_util import clip_eigenvalues, project_eigenvectors

class Phase1Model(BaseModel):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        batch_size=0,
        return_traj=False,
    ):
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

    def sampling_gaussians(self, num_shapes):
        """
        Return:
            ldm_gaus: np.ndarray
            gt_gaus: np.ndarray
        """
        ldm_gaus = self.sample(num_shapes)

        if self.hparams.get("global_normalization"):
            if not hasattr(self, "data_val"):
                self._build_dataset("val")
            if self.hparams.get("global_normalization") == "partial":
                ldm_gaus = self.data_val.unnormalize_global_static(ldm_gaus, slice(12,None)) 
            elif self.hparams.get("global_normalization") == "all":
                ldm_gaus = self.data_val.unnormalize_global_static(ldm_gaus, slice(None))

        ldm_gaus = clip_eigenvalues(ldm_gaus)
        ldm_gaus = project_eigenvectors(ldm_gaus)
        return ldm_gaus
