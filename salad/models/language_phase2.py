import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

from salad.model_components.lstm import LSTM
from salad.models.language_phase1 import LangPhase1Model
from salad.utils import imageutil, nputil, visutil
from salad.utils.spaghetti_util import (generate_zc_from_sj_gaus,
                                        get_mesh_from_spaghetti, load_mesher,
                                        load_spaghetti)
from salad.utils.train_util import get_dropout_mask


class LangPhase2Model(LangPhase1Model):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    def random_mask_gaus_text(self, gaus, text):
        if self.hparams.get("classifier_free_guidance"):
            text = list(text)
            B = gaus.shape[0]
            random_dp_mask = get_dropout_mask(
                B, self.hparams.conditioning_dropout_prob, self.device
            )
            gaus = gaus * random_dp_mask.unsqueeze(1).unsqueeze(2)
            for i in range(B):
                if random_dp_mask[i] == 0:
                    text[i] = ""

        return gaus, text

    def forward(self, x, gaus, text):
        """
        Input:
            x: [B,G,512]
            gaus: [B,G,16]
            text: list of [B]
        """
        B, G = x.shape[:2]
        gaus, text = self.random_mask_gaus_text(gaus, text)
        lang_emb = self.text_to_embedding(text)
        cond = self.cond_from_gaus_lang_f(gaus, lang_emb)

        return self.get_loss(x, cond)

    def step(self, batch, stage):
        x, gaus, text = batch
        loss = self(x, gaus, text)
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

    def cond_from_gaus_lang_f(self, gaus, lang_f):
        gaus = nputil.np2th(gaus).to(self.device)
        G = gaus.shape[1]
        lang_f = nputil.np2th(lang_f).to(self.device)
        assert gaus.ndim == 3
        if lang_f.ndim == 2:
            lang_f = lang_f.unsqueeze(1)
        lang_f = lang_f.expand(-1, G, -1)
        return torch.cat([gaus, lang_f], -1)

    def generate_null_cond(self, B, G):
        text = ["" for _ in range(B)]
        lang_emb = self.text_to_embedding(text)
        gaus = torch.zeros(B, G, 16, dtype=torch.float, device=self.device)
        return self.cond_from_gaus_lang_f(gaus, lang_emb)

    @torch.no_grad()
    def sample(
        self,
        num_samples_or_cond,
        return_traj=False,
        return_cond=False,
        classifier_free_guidance=False,
        free_guidance_weight=0.7,
    ):

        if isinstance(num_samples_or_cond, int):
            batch_size = num_samples_or_cond
            ds = self._build_dataset("val")
            batch_gaus = []
            batch_text = []
            for i in range(batch_size):
                _, gaus, text = ds[i]
                batch_gaus.append(gaus)
                batch_text.append(text)

            batch_gaus = torch.stack(batch_gaus, 0)
            lang_emb = self.text_to_embedding(batch_text)
            cond = self.cond_from_gaus_lang_f(batch_gaus, lang_emb).to(self.device)

        elif isinstance(num_samples_or_cond, np.ndarray) or isinstance(
            num_samples_or_cond, torch.Tensor
        ):
            cond = nputil.np2th(num_samples_or_cond).to(self.device)
            batch_size = len(cond)

        G = cond.shape[1]
        if classifier_free_guidance:
            null_cond = self.generate_null_cond(batch_size, G)

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

        if return_traj:
            if return_cond:
                return traj, cond
            return traj
        else:
            if return_cond:
                return traj[0], cond
            return traj[0]

    def validation(self):
        vis_num_shapes = 4
        vis_gt_sj = []
        vis_gaus = []
        vis_texts = []
        ds = self._build_dataset("val")
        vis_indices = [18453, 13036, 13204, 48244]
        for i in vis_indices:
            sj, gaus, text = ds[i]
            vis_gt_sj.append(sj)
            vis_gaus.append(gaus)
            vis_texts.append(text)

        vis_gt_sj = torch.stack(vis_gt_sj, 0)
        vis_gaus = torch.stack(vis_gaus, 0).to(self.device)
        vis_lang_f = self.text_to_embedding(vis_texts)
        vis_cond = self.cond_from_gaus_lang_f(vis_gaus, vis_lang_f)
        pred_sj = self.sample(vis_cond)

        if not hasattr(self, "spaghetti"):
            self.spaghetti = load_spaghetti(self.device, self.hparams.spaghetti_tag)
        spaghetti = self.spaghetti

        if not hasattr(self, "mesher"):
            self.mesher = load_mesher(self.device)
        mesher = self.mesher

        gt_zcs = generate_zc_from_sj_gaus(spaghetti, vis_gt_sj, vis_gaus)
        pred_zcs = generate_zc_from_sj_gaus(spaghetti, pred_sj, vis_gaus)

        wandb_logger = self.get_wandb_logger()
        for i in range(vis_num_shapes):
            gaus_img = visutil.render_gaussians(vis_gaus[i], resolution=(256, 256))
            vert, face = get_mesh_from_spaghetti(spaghetti, mesher, gt_zcs[i], res=128)
            gt_mesh_img = visutil.render_mesh(vert, face, resolution=(256, 256))
            img = [gaus_img, gt_mesh_img]
            try:
                vert, face = get_mesh_from_spaghetti(spaghetti, mesher, pred_zcs[i])
                pred_mesh_img = visutil.render_mesh(vert, face, resolution=(256, 256))
                img.append(pred_mesh_img)
            except Exception as e:
                print(e)
            img = imageutil.merge_images(img)
            img = imageutil.draw_text(
                img, vis_texts[i], font_size=14, max_seq_length=50
            )
            wandb_logger.log_image("vis", [img])
