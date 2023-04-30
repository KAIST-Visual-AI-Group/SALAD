import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

from salad.model_components.lstm import LSTM
from salad.models.phase1 import Phase1Model
from salad.utils import imageutil, nputil, visutil
from salad.utils.spaghetti_util import (clip_eigenvalues,
                                        generate_zc_from_sj_gaus,
                                        get_mesh_from_spaghetti, load_mesher,
                                        load_spaghetti, project_eigenvectors)
from salad.utils.train_util import get_dropout_mask
from salad.data.dataset import LangSALADDataset


class LangPhase1Model(Phase1Model):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if self.hparams.get("use_lstm"):
            self.bertmodel = LSTM(
                text_dim=768, embedding_dim=768, vocab_size=30522, padding_idx=0
            )
        else:
            self.bertmodel = BertModel.from_pretrained("bert-base-uncased")
        if self.hparams.get("text_encoder_freeze"):
            for p in self.bertmodel.parameters():
                p.requires_grad_(False)

    def forward(self, x, text):
        """
        Input:
            x: [B,G,16]
            text: list of length [B]
        """
        B, G = x.shape[:2]
        text = self.random_mask_text(text)
        lang_emb = self.text_to_embedding(text)
        return self.get_loss(x, lang_emb)

    def tokenizing(self, text):
        tokenized = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        return tokenized

    def text_to_embedding(self, text):
        """
        text: list of length [B]
        return [B,768]
        """
        tokenized = self.tokenizing(text)
        if self.hparams.get("use_lstm"):
            lang_emb, _ = self.bertmodel(tokenized.input_ids)
        else:
            if self.hparams.get("text_encoder_return_seq"):
                lang_emb = self.bertmodel(**tokenized).last_hidden_state
            else:
                lang_emb = self.bertmodel(**tokenized).pooler_output
        if lang_emb.ndim == 2:
            lang_emb = lang_emb.unsqueeze(1)
        return lang_emb

    def random_mask_text(self, text):
        text = list(text)
        B = len(text)
        if self.hparams.get("classifier_free_guidance"):
            random_dp_mask = get_dropout_mask(
                B, self.hparams.conditioning_dropout_prob, self.device
            )
            for i in range(B):
                if random_dp_mask[i] == 0:
                    text[i] = ""
        return text

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

    def step(self, batch, stage: str):
        x, text = batch
        loss = self(x, text)
        self.log(f"{stage}/loss", loss, on_step=stage == "train", prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(
        self,
        num_samples_or_text,
        return_traj=False,
        return_cond=False,
        classifier_free_guidance=True,
        free_guidance_weight=2.0,
    ):
        if isinstance(num_samples_or_text, str):
            num_samples_or_text = [num_samples_or_text]
        if isinstance(num_samples_or_text, int):
            batch_size = num_samples_or_text
            ds = self._build_dataset("val")
            texts = [ds[i][1] for i in range(batch_size)]
        elif isinstance(num_samples_or_text, list):
            texts = num_samples_or_text
            batch_size = len(num_samples_or_text)
        if self.hparams.get("use_zc"):
            x_T = torch.randn([batch_size, 16, 512]).to(self.device)
        else:
            x_T = torch.randn([batch_size, 16, 16]).to(self.device)
        G = x_T.shape[1]
        lang_emb = self.text_to_embedding(texts)

        if classifier_free_guidance:
            null_texts = ["" for _ in range(batch_size)]
            null_lang_emb = self.text_to_embedding(null_texts)

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
            e_theta = self.net(x_t, beta=beta, context=lang_emb)

            if classifier_free_guidance:
                null_e_theta = self.net(x_t, beta=beta, context=null_lang_emb)
                w = free_guidance_weight
                e_theta = (1 + w) * e_theta - w * null_e_theta

            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()

            traj[t] = traj[t].cpu()

            if not return_traj:
                del traj[t]

        if return_traj:
            if return_cond:
                return traj, lang_emb
            return traj
        else:
            if return_cond:
                return traj[0], lang_emb
            return traj[0]

    def sampling_gaussians(
        self,
        num_samples_or_text,
        classifier_free_guidance=True,
        free_guidance_weight=2.0,
        return_cond=False,
    ):
        gaus = self.sample(
            num_samples_or_text,
            classifier_free_guidance=classifier_free_guidance,
            free_guidance_weight=free_guidance_weight,
            return_cond=return_cond,
        )
        if isinstance(gaus, tuple):
            text = gaus[1]
            gaus = gaus[0]
        # gaus = reflect_and_concat_gmms(raw_gaus)
        if self.hparams.get("global_normalization"):
            if not hasattr(self, "data_val"):
                self._build_dataset("val")
            if self.hparams.get("global_normalization") == "partial":
                gaus = self.data_val.unnormalize_global_static(gaus, slice(12, None))
            elif self.hparams.get("global_normalization") == "all":
                gaus = self.data_val.unnormalize_global_static(gaus, slice(None))

        gaus = project_eigenvectors(clip_eigenvalues(gaus))
        if return_cond:
            return gaus, text
        return gaus

    def _build_dataset(self, stage):
        if hasattr(self, f"data_{stage}"):
            return getattr(self, f"data_{stage}")

        ds_class = (
            LangSALADDataset
        )
        if stage == "train":
            ds = ds_class(**self.hparams.dataset_kwargs)
        else:
            dataset_kwargs = self.hparams.dataset_kwargs.copy()
            dataset_kwargs["repeat"] = 1
            ds = ds_class(**dataset_kwargs)
        setattr(self, f"data_{stage}", ds)
        return ds

    def validation_zc(self):
        vis_num_shapes = 4
        vis_zcs = []
        vis_texts = []
        ds = self._build_dataset("val")
        for i in [0, 1, 2, 3]:
            zcs, text = ds[i]
            vis_zcs.append(zcs)
            vis_texts.append(text)
        vis_zcs = torch.stack(vis_zcs, 0)
        ldm_zcs = self.sample(vis_texts)

        if not hasattr(self, "spaghetti"):
            self.spaghetti = load_spaghetti(self.device, self.hparams.spaghetti_tag)
        spaghetti = self.spaghetti

        if not hasattr(self, "mesher"):
            self.mesher = load_mesher(self.device)
        mesher = self.mesher

        wandb_logger = self.get_wandb_logger()
        images = []
        for i in range(vis_num_shapes):
            try:
                v, f = get_mesh_from_spaghetti(spaghetti, mesher, vis_zcs[i], res=128)
                gt_img = visutil.render_mesh(v, f, resolution=(256, 256))
            except:
                pass
            try:
                v, f = get_mesh_from_spaghetti(spaghetti, mesher, ldm_zcs[i], res=128)
                pred_img = visutil.render_mesh(v, f, resolution=(256, 256))
            except:
                pass

            img = imageutil.merge_images([gt_img, pred_img])
            img = imageutil.draw_text(
                img,
                f"Left: GT | Right: Pred \n{vis_texts[i]}",
                font_size=14,
                max_seq_length=50,
            )
            images.append([img])

        images = imageutil.merge_images(images)
        wandb_logger.log_image("vis", [images])

    def validation(self):
        if self.hparams.get("use_zc"):
            self.validation_zc()
            return

        vis_num_shapes = 4
        vis_gaus = []
        vis_texts = []
        ds = self._build_dataset("val")
        vis_indices = [18453, 13036, 13204, 48244]
        for i in vis_indices:
            gaus, text = ds[i]
            vis_gaus.append(gaus)
            vis_texts.append(text)

        vis_gaus = torch.stack(vis_gaus, 0)
        if self.hparams.get("global_normalization"):
            if self.hparams.get("global_normalization") == "partial":
                vis_gaus = self.data_val.unnormalize_global_static(
                    vis_gaus, slice(12, None)
                )
            elif self.hparams.get("global_normalization") == "all":
                vis_gaus = self.dataval.unnormalize_global_static(vis_gaus, slice(None))

        # vis_gaus = reflect_and_concat_gmms(vis_gaus)
        pred_gaus = self.sampling_gaussians(vis_texts)

        if not hasattr(self, "spaghetti"):
            self.spaghetti = load_spaghetti(self.device, self.hparams.spaghetti_tag)
        spaghetti = self.spaghetti

        if not hasattr(self, "mesher"):
            self.mesher = load_mesher(self.device)
        mesher = self.mesher

        """ get intrinsics """
        # TODO change the ckpt path.
        if not hasattr(self, "phase2_model"):
            phase2_ckpt = "/home/juil/pvddir/results/phase2/augment_final_0214/0214_202607/checkpoints/epoch=4999-val_loss=0.0000.ckpt"
            self.phase2_model = SpaghettiConditionSALDM.load_from_checkpoint(
                phase2_ckpt, strict=False
            ).to(self.device)
            self.phase2_model.eval()
            for p in self.phase2_model.parameters():
                p.requires_grad_(False)

        phase2_model = self.phase2_model

        gt_sj = phase2_model.sample(vis_gaus)
        pred_sj = phase2_model.sample(pred_gaus)

        gt_zcs = generate_zc_from_sj_gaus(spaghetti, gt_sj, vis_gaus)
        pred_zcs = generate_zc_from_sj_gaus(spaghetti, pred_sj, pred_gaus)

        wandb_logger = self.get_wandb_logger()
        images = []
        for i in range(vis_num_shapes):
            gt_img = visutil.render_gaussians(vis_gaus[i], resolution=(256, 256))
            try:
                v, f = get_mesh_from_spaghetti(spaghetti, mesher, gt_zcs[i], res=128)
                gt_mesh_img = visutil.render_mesh(v, f, resolution=(256, 256))
                gt_img = imageutil.merge_images([gt_img, gt_mesh_img])
            except:
                pass

            pred_img = visutil.render_gaussians(pred_gaus[i], resolution=(256, 256))
            try:
                v, f = get_mesh_from_spaghetti(spaghetti, mesher, pred_zcs[i], res=128)
                pred_mesh_img = visutil.render_mesh(v, f, resolution=(256, 256))
                pred_img = imageutil.merge_images([pred_img, pred_mesh_img])
            except:
                pass

            img = imageutil.merge_images([gt_img, pred_img])
            img = imageutil.draw_text(
                img,
                f"Left: GT | Right: Pred \n{vis_texts[i]}",
                font_size=14,
                max_seq_length=50,
            )
            images.append([img])

        images = imageutil.merge_images(images)
        wandb_logger.log_image("vis", [images])
