from pvd.dataset import T5SpaghettiLatentDataset, ShapeglotSpaghettiDataset, ShapeglotSpaghettiDataset2
import torch
from pathlib import Path
from pvd.diffusion.ldm import *
import jutils
import numpy as np
from transformers import BertTokenizer, BertModel
from partglot.modules.encoders import LSTM


class LanguageGaussianSymSALDM(SpaghettiSALDM):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    def forward(self, x, cond):
        """
        Input:
            x: [B,G,16]
            cond: [B,D]
        """
        B, G = x.shape[:2]
        lang_f = cond
        lang_f = lang_f.unsqueeze(1).repeat(1, G, 1)  # [B,G,D]

        if self.hparams.get("classifier_free_guidance") and self.training:
            mask_shape = (B, 1, 1)
            random_dp_mask = get_dropout_mask(
                mask_shape, self.hparams.conditioning_dropout_prob, self.device
            )
            cond = cond * random_dp_mask

        return self.get_loss(x, lang_f)

    def step(self, batch, stage: str):
        x, cond = batch
        lang_f = cond
        loss = self(x, lang_f)
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
    def sample(self, num_samples_or_lang_f, return_traj=False, return_cond=False,
            classifier_free_guidance=False, free_guidance_weight=0.7):
        if isinstance(num_samples_or_lang_f, int):
            batch_size = num_samples_or_lang_f
            ds = self._build_dataset("val")
            cond = torch.stack([ds[i][1] for i in range(batch_size)], 0).to(self.device)
        elif isinstance(num_samples_or_lang_f, np.ndarray) or isinstance(
            num_samples_or_lang_f, torch.Tensor
        ):
            cond = jutils.nputil.np2th(num_samples_or_lang_f).to(self.device)
            batch_size = len(cond)

        x_T = torch.randn([batch_size, 8, 16]).to(self.device)
        if cond.ndim == 2:
            B, G = x_T.shape[:2]
            cond = cond.unsqueeze(1).repeat(1, G, 1)
        
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

        if return_traj:
            if return_cond:
                return traj, cond
            return traj
        else:
            if return_cond:
                return traj[0], cond
            return traj[0]

    def sampling_gaussians(self, num_samples_or_lang_f, classifier_free_guidance=False, free_guidance_weight=0.7):
        gaus = self.sample(num_samples_or_lang_f, classifier_free_guidance=classifier_free_guidance, free_guidance_weight=free_guidance_weight)

        half = gaus
        full = reflect_and_concat_gmms(half)
        gaus = full
        if self.hparams.get("global_normalization"):
            if not hasattr(self, "data_val"):
                self._build_dataset("val")
            print(
                f"[!] Unnormalize samples as {self.hparams.get('global_normalization')}"
            )
            if self.hparams.get("global_normalization") == "partial":
                gaus = self.data_val.unnormalize_global_static(gaus, slice(12, None))
            elif self.hparams.get("global_normalization") == "all":
                gaus = self.dataval.unnormalize_global_static(gaus, slice(None))

        gaus = project_eigenvectors(clip_eigenvalues(gaus))
        return gaus

    def _build_dataset(self, stage):
        if hasattr(self, f"data_{stage}"):
            return getattr(self, f"data_{stage}")

        ds = T5SpaghettiLatentDataset(**self.hparams.dataset_kwargs)
        setattr(self, f"data_{stage}", ds)
        return ds

    def validation(self):
        vis_num_shapes = 4
        vis_gaus = []
        vis_lang_f = []
        vis_texts = []
        ds = self._build_dataset("val")
        for i in range(vis_num_shapes):
            gaus, lang_f, text = ds.getitem_with_text(i)
            vis_gaus.append(gaus)
            vis_lang_f.append(lang_f)
            vis_texts.append(text)

        vis_gaus = torch.stack(vis_gaus, 0)
        vis_gaus = reflect_and_concat_gmms(vis_gaus)
        vis_lang_f = torch.stack(vis_lang_f, 0)
        pred_gaus = self.sampling_gaussians(vis_lang_f)

        if not hasattr(self, "spaghetti"):
            self.spaghetti = load_spaghetti(self.device, self.hparams.spaghetti_tag)
        spaghetti = self.spaghetti

        if not hasattr(self, "mesher"):
            self.mesher = load_mesher(self.device)
        mesher = self.mesher

        """ get intrinsics """
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
            gt_img = jutils.visutil.render_gaussians(vis_gaus[i], resolution=(256, 256))
            try:
                v, f = get_mesh_from_spaghetti(spaghetti, mesher, gt_zcs[i], res=128)
                gt_mesh_img = jutils.visutil.render_mesh(v, f, resolution=(256, 256))
                gt_img = jutils.imageutil.merge_images([gt_img, gt_mesh_img])
            except:
                pass

            pred_img = jutils.visutil.render_gaussians(
                pred_gaus[i], resolution=(256, 256)
            )
            try:
                v, f = get_mesh_from_spaghetti(spaghetti, mesher, pred_zcs[i], res=128)
                pred_mesh_img = jutils.visutil.render_mesh(v, f, resolution=(256, 256))
                pred_img = jutils.imageutil.merge_images([pred_img, pred_mesh_img])
            except:
                pass

            img = jutils.imageutil.merge_images([gt_img, pred_img])
            img = jutils.imageutil.draw_text(
                img,
                f"Left: GT | Right: Pred \n{vis_texts[i]}",
                font_size=14,
                max_seq_length=50,
            )
            images.append([img])

        images = jutils.imageutil.merge_images(images)
        wandb_logger.log_image("vis", [images])


class LanguageConditionIntrinsicLDM(SpaghettiSALDM):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    def forward(self, x, gaus, lang_f):
        (
            B,
            G,
        ) = x.shape[:2]
        cond = self.cond_from_gaus_lang_f(gaus, lang_f)

        if self.hparams.get("classifier_free_guidance") and self.training:
            mask_shape = (B, 1, 1)
            random_dp_mask = get_dropout_mask(
                mask_shape, self.hparams.conditioning_dropout_prob, self.device
            )
            cond = cond * random_dp_mask

        return self.get_loss(x, cond)

    def cond_from_gaus_lang_f(self, gaus, lang_f):
        assert gaus.ndim == 3  # [B,G,16]
        if lang_f.ndim == 2:
            G = gaus.shape[1]
            lang_f = lang_f.unsqueeze(1).repeat(1, G, 1)

        return torch.cat([gaus, lang_f], -1)

    def step(self, batch, stage: str):
        x, gaus, lang_f = batch
        loss = self(x, gaus, lang_f)
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
    def sample(self, num_samples_or_cond, return_traj=False, return_cond=False):
        if isinstance(num_samples_or_cond, int):
            batch_size = num_samples_or_cond
            ds = self._build_dataset("val")
            batch_gaus = []
            batch_lang_f = []
            for i in range(batch_size):
                _, gaus, lang_f = ds[i]
                batch_gaus.append(gaus)
                batch_lang_f.append(lang_f)

            batch_gaus = torch.stack(batch_gaus, 0)
            batch_lang_f = (
                torch.stack(batch_lang_f, 0)
                .unsqueeze(1)
                .repeat(1, batch_gaus.shape[1], 1)
            )
            cond = self.cond_from_gaus_lang_f(batch_gaus, batch_lang_f).to(self.device)

        elif isinstance(num_samples_or_cond, np.ndarray) or isinstance(
            num_samples_or_cond, torch.Tensor
        ):
            cond = jutils.nputil.np2th(num_samples_or_cond).to(self.device)
            batch_size = len(cond)

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
            # print(x_t.shape, cond.shape)
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

    def _build_dataset(self, stage):
        if hasattr(self, f"data_{stage}"):
            return getattr(self, f"data_{stage}")

        ds = T5SpaghettiLatentDataset(**self.hparams.dataset_kwargs)
        setattr(self, f"data_{stage}", ds)
        return ds

    def validation(self):
        vis_num_shapes = 8
        vis_gt_sj = []
        vis_gaus = []
        vis_lang_f = []
        vis_texts = []
        ds = self._build_dataset("val")
        for i in range(vis_num_shapes):
            sj, gaus, lang_f, text = ds.getitem_with_text(i)
            vis_gt_sj.append(sj)
            vis_gaus.append(gaus)
            vis_lang_f.append(lang_f)
            vis_texts.append(text)

        vis_gt_sj = torch.stack(vis_gt_sj, 0)
        vis_gaus = torch.stack(vis_gaus, 0)
        vis_lang_f = torch.stack(vis_lang_f, 0)
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
            gaus_img = jutils.visutil.render_gaussians(
                vis_gaus[i], resolution=(256, 256)
            )
            vert, face = get_mesh_from_spaghetti(spaghetti, mesher, gt_zcs[i], res=128)
            gt_mesh_img = jutils.visutil.render_mesh(vert, face, resolution=(256, 256))
            img = [gaus_img, gt_mesh_img]
            try:
                vert, face = get_mesh_from_spaghetti(spaghetti, mesher, pred_zcs[i])
                pred_mesh_img = jutils.visutil.render_mesh(
                    vert, face, resolution=(256, 256)
                )
                img.append(pred_mesh_img)
            except Exception as e:
                print(e)
            img = jutils.imageutil.merge_images(img)
            img = jutils.imageutil.draw_text(
                img, vis_texts[i], font_size=14, max_seq_length=50
            )
            wandb_logger.log_image("vis", [img])

class ShapeglotPhase1(SpaghettiSALDM):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if self.hparams.get("use_lstm"):
            self.bertmodel = LSTM(text_dim=768, embedding_dim=768, vocab_size=30522, padding_idx=0)
        else:
            self.bertmodel = BertModel.from_pretrained("bert-base-uncased")
        if self.hparams.get("text_encoder_freeze"):
            for p in self.bertmodel.parameters(): p.requires_grad_(False)


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
        tokenized = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
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
            random_dp_mask = get_dropout_mask(B, self.hparams.conditioning_dropout_prob, self.device)
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
    def sample(self, num_samples_or_text, return_traj=False, return_cond=False, classifier_free_guidance=False, free_guidance_weight=0.7):
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
        # lang_emb = lang_emb.unsqueeze(1).repeat(1,G,1)
        
        if classifier_free_guidance:
            null_texts = ["" for _ in range(batch_size)]
            null_lang_emb = self.text_to_embedding(null_texts)
            # null_lang_emb = null_lang_emb.unsqueeze(1).repeat(1,G,1)
        
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
    
    def sampling_gaussians(self, num_samples_or_text, classifier_free_guidance=False, free_guidance_weight=0.7, return_cond=False):
        gaus = self.sample(num_samples_or_text, classifier_free_guidance=classifier_free_guidance, free_guidance_weight=free_guidance_weight, return_cond=return_cond)
        if isinstance(gaus, tuple):
            text = gaus[1]
            gaus = gaus[0]
        # gaus = reflect_and_concat_gmms(raw_gaus)
        if self.hparams.get("global_normalization"):
            if not hasattr(self, "data_val"):
                self._build_dataset("val")
            if self.hparams.get("global_normalization") == "partial":
                gaus = self.data_val.unnormalize_global_static(gaus, slice(12,None)) 
            elif self.hparams.get("global_normalization") == "all":
                gaus = self.data_val.unnormalize_global_static(gaus, slice(None))

        gaus = project_eigenvectors(clip_eigenvalues(gaus))
        if return_cond:
            return gaus, text
        return gaus

    def _build_dataset(self, stage):
        if hasattr(self, f"data_{stage}"):
            return getattr(self, f"data_{stage}")

        ds_class = ShapeglotSpaghettiDataset2 if self.hparams.get("use_partglot_data") else ShapeglotSpaghettiDataset
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
        for i in [0,1,2,3]:
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
                gt_img = jutils.visutil.render_mesh(v, f, resolution=(256, 256))
            except:
                pass
            try:
                v, f = get_mesh_from_spaghetti(spaghetti, mesher, ldm_zcs[i], res=128)
                pred_img = jutils.visutil.render_mesh(v, f, resolution=(256, 256))
            except:
                pass

            img = jutils.imageutil.merge_images([gt_img, pred_img])
            img = jutils.imageutil.draw_text(
                img,
                f"Left: GT | Right: Pred \n{vis_texts[i]}",
                font_size=14,
                max_seq_length=50,
            )
            images.append([img])

        images = jutils.imageutil.merge_images(images)
        wandb_logger.log_image("vis", [images])


    def validation(self):
        if self.hparams.get("use_zc"):
            self.validation_zc()
            return

        vis_num_shapes = 4
        vis_gaus = []
        vis_texts = []
        ds = self._build_dataset("val")
        if self.hparams.get("use_partglot_data"):
            vis_indices = [0,1,2,3]
        else:
            vis_indices = [18453,13036,13204,48244]
        for i in vis_indices:
            gaus, text = ds[i]
            vis_gaus.append(gaus)
            vis_texts.append(text)

        vis_gaus = torch.stack(vis_gaus, 0)
        if self.hparams.get("global_normalization"):
            if self.hparams.get("global_normalization") == "partial":
                vis_gaus = self.data_val.unnormalize_global_static(vis_gaus, slice(12, None))
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
            gt_img = jutils.visutil.render_gaussians(vis_gaus[i], resolution=(256, 256))
            try:
                v, f = get_mesh_from_spaghetti(spaghetti, mesher, gt_zcs[i], res=128)
                gt_mesh_img = jutils.visutil.render_mesh(v, f, resolution=(256, 256))
                gt_img = jutils.imageutil.merge_images([gt_img, gt_mesh_img])
            except:
                pass

            pred_img = jutils.visutil.render_gaussians(
                pred_gaus[i], resolution=(256, 256)
            )
            try:
                v, f = get_mesh_from_spaghetti(spaghetti, mesher, pred_zcs[i], res=128)
                pred_mesh_img = jutils.visutil.render_mesh(v, f, resolution=(256, 256))
                pred_img = jutils.imageutil.merge_images([pred_img, pred_mesh_img])
            except:
                pass

            img = jutils.imageutil.merge_images([gt_img, pred_img])
            img = jutils.imageutil.draw_text(
                img,
                f"Left: GT | Right: Pred \n{vis_texts[i]}",
                font_size=14,
                max_seq_length=50,
            )
            images.append([img])

        images = jutils.imageutil.merge_images(images)
        wandb_logger.log_image("vis", [images])

        
class ShapeglotPhase2(ShapeglotPhase1):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)
        
    def random_mask_gaus_text(self, gaus, text):
        if self.hparams.get("classifier_free_guidance"):
            text = list(text)
            B = gaus.shape[0]
            random_dp_mask = get_dropout_mask(B, self.hparams.conditioning_dropout_prob, self.device)
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
        self.log(f"{stage}/loss", loss, on_step=stage=="train", prog_bar=True)
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
        gaus = jutils.nputil.np2th(gaus).to(self.device)
        G = gaus.shape[1]
        lang_f = jutils.nputil.np2th(lang_f).to(self.device)
        assert gaus.ndim == 3
        if lang_f.ndim == 2:
            lang_f = lang_f.unsqueeze(1)
        lang_f = lang_f.expand(-1,G,-1)
        return torch.cat([gaus, lang_f], -1)

    def generate_null_cond(self, B, G):
        text = ["" for _ in range(B)]
        lang_emb = self.text_to_embedding(text)
        gaus = torch.zeros(B,G,16,dtype=torch.float, device=self.device)
        return self.cond_from_gaus_lang_f(gaus, lang_emb)

    @torch.no_grad()
    def sample(self, num_samples_or_cond, return_traj=False, return_cond=False,
            classifier_free_guidance=False, free_guidance_weight=0.7):
    
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
            cond = jutils.nputil.np2th(num_samples_or_cond).to(self.device)
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
        if self.hparams.get("use_partglot_data"):
            vis_indices = [0,1,2,3]
        else:
            vis_indices = [18453,13036,13204,48244]
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
            gaus_img = jutils.visutil.render_gaussians(
                vis_gaus[i], resolution=(256, 256)
            )
            vert, face = get_mesh_from_spaghetti(spaghetti, mesher, gt_zcs[i], res=128)
            gt_mesh_img = jutils.visutil.render_mesh(vert, face, resolution=(256, 256))
            img = [gaus_img, gt_mesh_img]
            try:
                vert, face = get_mesh_from_spaghetti(spaghetti, mesher, pred_zcs[i])
                pred_mesh_img = jutils.visutil.render_mesh(
                    vert, face, resolution=(256, 256)
                )
                img.append(pred_mesh_img)
            except Exception as e:
                print(e)
            img = jutils.imageutil.merge_images(img)
            img = jutils.imageutil.draw_text(
                img, vis_texts[i], font_size=14, max_seq_length=50
            )
            wandb_logger.log_image("vis", [img])


class ShapeglotSinglePhase(ShapeglotPhase1):
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    @torch.no_grad()
    def sample(self, num_samples_or_text, return_traj=False, return_cond=False, classifier_free_guidance=False, free_guidance_weight=0.7):
        if isinstance(num_samples_or_text, str):
            num_samples_or_text = [num_samples_or_text]
        if isinstance(num_samples_or_text, int):
            batch_size = num_samples_or_text
            ds = self._build_dataset("val")
            texts = [ds[i][1] for i in range(batch_size)]
        elif isinstance(num_samples_or_text, list):
            texts = num_samples_or_text
            batch_size = len(num_samples_or_text)
        
        x_T = torch.randn([batch_size, 16, 528]).to(self.device)
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

    
    def validation(self):
        vis_num_shapes = 4
        gt_sj_gaus = []
        texts = []
        ds = self._build_dataset("val")
        if self.hparams.get("use_partglot_data"):
            vis_indices = [0,1,2,3]
        else:
            vis_indices = [18453,13036,13204,48244]
        for i in vis_indices:
            x, text = ds[i]
            gt_sj_gaus.append(x)
            texts.append(text)

        gt_sj_gaus = torch.stack(gt_sj_gaus, 0).to(self.device)
        ldm_sj_gaus = self.sample(texts)
        ldm_sj_gaus = jutils.nputil.np2th(ldm_sj_gaus).to(self.device)

        gt_sj, gt_gaus = gt_sj_gaus.split(split_size=[512, 16], dim=-1)
        ldm_sj, ldm_gaus = ldm_sj_gaus.split(split_size=[512, 16], dim=-1)
        if self.hparams.get("global_normalization") == "partial":
            ldm_gaus = self._build_dataset("val").unnormalize_global_static(ldm_gaus, slice(12,None))
            gt_gaus = self._build_dataset("val").unnormalize_global_static(gt_gaus, slice(12,None))
        elif self.hparams.get("global_normalization") == "all":
            ldm_gaus = self._build_dataset("val").unnormalize_global_static(ldm_gaus, slice(None))
            gt_gaus = self._build_dataset("val").unnormalize_global_static(gt_gaus, slice(None))

        if not hasattr(self, "spaghetti"):
            self.spaghetti = load_spaghetti(self.device, self.hparams.spaghetti_tag)
        spaghetti = self.spaghetti

        if not hasattr(self, "mesher"):
            self.mesher = load_mesher(self.device)
        mesher = self.mesher
        
        gt_zcs = generate_zc_from_sj_gaus(spaghetti, gt_sj, gt_gaus)
        ldm_zcs = generate_zc_from_sj_gaus(spaghetti, ldm_sj, ldm_gaus)
    
        wandb_logger = self.get_wandb_logger()
        images = []
        for i in range(vis_num_shapes):
            def draw_per_shape(gaus, zc):
                resolution=(256,256)
                gaus_img = jutils.visutil.render_gaussians(clip_eigenvalues(gaus), resolution=resolution)
                try:
                    v, f = get_mesh_from_spaghetti(spaghetti, mesher, zc, res=128)
                    mesh_img = jutils.visutil.render_mesh(v,f, resolution=resolution)
                    return jutils.imageutil.merge_images([gaus_img, mesh_img])
                except:
                    return gaus_img
            
            gt_img = draw_per_shape(gt_gaus[i], gt_zcs[i])
            pred_img = draw_per_shape(ldm_gaus[i], ldm_zcs[i])
            img = jutils.imageutil.merge_images([gt_img, pred_img])
            img = jutils.imageutil.draw_text(img, f"{texts[i]}", font_size=14, max_seq_length=50)
            images.append([img])
        images = jutils.imageutil.merge_images(images)
        wandb_logger.log_image("vis", [images])
