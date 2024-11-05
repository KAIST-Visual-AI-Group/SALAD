import io
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pvd.diffusion.ldm import *

#### < Seungwoo >
# Renamed class.
class SinglePhaseSALDM(SpaghettiSALDM):
# class GaussianSALDM(SpaghettiSALDM):
####
    def __init__(self, network, variance_schedule, **kwargs):
        super().__init__(network, variance_schedule, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        batch_size=0,
        return_traj=False,
    ):
        #### < Seungwoo >
        # Changed the last dimension: 16 -> 528 [gmm (16) | intrinsic (512)]
        x_T = torch.randn([batch_size, 16, 528]).to(self.device)

        # x_T = torch.randn([batch_size, 16, 16]).to(self.device)
        ####

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

        #### < Seungwoo >
        # In the concatenated setup, the first 16 among 528 channels
        # represent the parameters of part Gaussian blobs.
        # We need to fix indexed range to prevent clipping intrinsics.
        if self.hparams.get("eigen_value_clipping"):
            # traj[0][:,:,13:] = torch.clamp_min(traj[0][:,:,13:], min=self.hparams['eigen_value_clip_val'])
            traj[0][:,:,13:16] = torch.clamp_min(traj[0][:,:,13:16], min=self.hparams['eigen_value_clip_val'])
        ####
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
        batch_size = self.hparams.batch_size
        gaus_data_iter = iter(self.val_dataloader())

        ldm_gaus = []
        gt_gaus = []
        print("[*] Start sampling")
        for i in range(int(np.ceil(num_shapes / batch_size))):
            batch_ldm_gaus = self.sample(batch_size)
            batch_gt_gaus = jutils.thutil.th2np(next(gaus_data_iter))

            ldm_gaus.append(jutils.thutil.th2np(batch_ldm_gaus))
            gt_gaus.append(batch_gt_gaus)
        print("[*] Finished sampling")

        ldm_gaus, gt_gaus = list(
            map(lambda x: np.concatenate(x)[:num_shapes], [ldm_gaus, gt_gaus])
        )
        return ldm_gaus, gt_gaus

    def validation(self):
        ####
        batch_size = self.hparams.batch_size
        latent_data_iter = iter(self.val_dataloader())

        num_shapes_for_eval = 512
        # num_shapes_for_eval = 4

        gt_gaus = []
        gt_zs = []
        ldm_gaus = []
        ldm_zs = []

        print("[*] Start sampling")

        #### < Seungwoo >
        # Validation code brought from the conditional LDM model (Phase 2).
        for i in range(int(np.ceil(num_shapes_for_eval / batch_size))):

            batch_in = next(latent_data_iter)
            batch_gt_gaus, batch_gt_zs = torch.split(batch_in, (16, 512), dim=2)
            batch_gt_gaus = jutils.thutil.th2np(batch_gt_gaus)
            batch_gt_zs = jutils.thutil.th2np(batch_gt_zs)
            # print(batch_gt_gaus.shape, batch_gt_zs.shape, type(batch_gt_gaus), type(batch_gt_zs))

            batch_out = self.sample(batch_gt_gaus.shape[0], return_traj=False)
            batch_ldm_gaus, batch_ldm_zs = torch.split(batch_out, (16, 512), dim=2)
            batch_ldm_gaus = jutils.thutil.th2np(batch_ldm_gaus)
            batch_ldm_zs = jutils.thutil.th2np(batch_ldm_zs)
            # print(batch_ldm_gaus.shape, batch_ldm_zs.shape, type(batch_ldm_gaus), type(batch_ldm_zs))

            gt_gaus.append(batch_gt_gaus)
            gt_zs.append(batch_gt_zs)
            ldm_gaus.append(batch_ldm_gaus)
            ldm_zs.append(batch_ldm_zs)

            # batch_gt_zs, batch_gt_gaus = list(
            #     map(lambda x: jutils.thutil.th2np(x), iter(next(latent_data_iter)))
            # )
            # batch_ldm_zs = self.sample(batch_gt_gaus)

            # ldm_zs.append(jutils.thutil.th2np(batch_ldm_zs))
            # gt_zs.append(batch_gt_zs)
            # gt_gaus.append(batch_gt_gaus)
            # gt_gaus.append(batch_gt_gaus)
            # gt_zs.append(batch_gt_zs)

        print("[*] Finish sampling")

        #### < Seungwoo >
        # Listify additional data 'ldm_gaus'
        # ldm_zs, gt_zs, gt_gaus = list(
        #     map(
        #         lambda x: np.concatenate(x)[:num_shapes_for_eval],
        #         [ldm_zs, gt_zs, gt_gaus],
        #     )
        # )

        # Listify all data
        gt_gaus, gt_zs, ldm_gaus, ldm_zs = list(
            map(
                lambda x: np.concatenate(x)[:num_shapes_for_eval],
                [gt_gaus, gt_zs, ldm_gaus, ldm_zs],
            )
        )
        ####

        self.log(
            "GT norm",
            np.linalg.norm(gt_zs, axis=-1).mean(),
        )
        self.log(
            "LDM norm",
            np.linalg.norm(ldm_zs, axis=-1).mean(),
        )
        """=========================="""

        """ Run latent evaluation """
        evaluator = Evaluator(
            gt_set=gt_zs,
            pred_set=ldm_zs,
            batch_size=128,
            device=self.device,
            metric="l2",
        )

        latent_res = evaluator.compute_all_metrics(
            verbose=False, return_distance_matrix=True, compute_jsd_together=False
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
            tsne_num_shapes = slice(None)  # slice(1024)
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

        """ Spaghetti Decoding """
        jutils.sysutil.clean_gpu()

        if not hasattr(self, "spaghetti"):
            #### < Seungwoo >
            # Added 'tag' argument
            # spaghetti = load_spaghetti(self.device)
            spaghetti = load_spaghetti(self.device, self.hparams.spaghetti_tag)
            ####
            self.spaghetti = spaghetti
        else:
            spaghetti = self.spaghetti

        if not hasattr(self, "mesher"):
            mesher = load_mesher(self.device)
            self.mesher = mesher
        else:
            mesher = self.mesher

        vis_num_shapes = 3

        #### < Seungwoo >
        # Make no variation yet
        # The original source is below.

        # num_variations = 2
        # vis_ldm_zs = []
        # for i in range(vis_num_shapes):
        #     gaus = jutils.nputil.np2th(gt_gaus[i])[None].repeat(num_variations, 1, 1)
        #     ith_vis_ldm_zs = self.sample(gaus)  # [3,512]
        #     vis_ldm_zs.append(ith_vis_ldm_zs)
        # vis_ldm_zs = torch.cat(vis_ldm_zs, 0)  # [N * Vari,512]
        # 
        # # vis_ldm_zs = jutils.nputil.np2th(ldm_zs[:vis_num_shapes]).to(self.device)
        # vis_gt_zs = jutils.nputil.np2th(gt_zs[:vis_num_shapes]).to(self.device)
        # vis_gt_gaus = jutils.nputil.np2th(gt_gaus[:vis_num_shapes])
        # vis_gt_gmms = batch_gaus_to_gmms(vis_gt_gaus, self.device)
        # vis_gt_gmms_repeated = batch_gaus_to_gmms(
        #     vis_gt_gaus.repeat_interleave(num_variations, 0), self.device
        # )
        # vis_gt_zcs, _ = spaghetti.merge_zh(vis_gt_zs, vis_gt_gmms)
        # vis_ldm_zcs, _ = spaghetti.merge_zh(vis_ldm_zs, vis_gt_gmms_repeated)
        # 
        # vis_ldm_zs = vis_gt_zs = vis_gt_gmms = vis_gt_gmms_repeated = None
        # jutils.sysutil.clean_gpu()

        vis_gt_gaus = jutils.nputil.np2th(gt_gaus[:vis_num_shapes])
        vis_gt_zs = jutils.nputil.np2th(gt_zs[:vis_num_shapes]).to(self.device)
        vis_ldm_gaus = jutils.nputil.np2th(ldm_gaus[:vis_num_shapes])
        vis_ldm_zs = jutils.nputil.np2th(ldm_zs[:vis_num_shapes]).to(self.device)

        vis_gt_gmms = batch_gaus_to_gmms(vis_gt_gaus, self.device)
        vis_ldm_gmms = batch_gaus_to_gmms(vis_ldm_gaus, self.device)

        vis_gt_zcs, _ = spaghetti.merge_zh(vis_gt_zs, vis_gt_gmms)
        vis_ldm_zcs, _ = spaghetti.merge_zh(vis_ldm_zs, vis_ldm_gmms)
        print(f"[~] {vis_gt_zcs.shape} {vis_ldm_zcs.shape}")

        vis_ldm_zs = vis_gt_zs = vis_gt_gmms = vis_ldm_gmms = None
        jutils.sysutil.clean_gpu()
        ####

        wandb_logger = self.get_wandb_logger()

        camera_kwargs = dict(
            camPos=np.array([-2, 2, -2]),
            camLookat=np.array([0, 0, 0]),
            camUp=np.array([0, 1, 0]),
            resolution=(512, 512),
            samples=32,
        )

        for i in range(vis_num_shapes):
            img_per_shape = []

            # Render: GT GMM
            gt_gaus_img = jutils.visutil.render_gaussians(vis_gt_gaus[i])
            img_per_shape.append(gt_gaus_img)

            # Render: LDM GMM
            ldm_gaus_img = jutils.visutil.render_gaussians(vis_ldm_gaus[i])
            img_per_shape.append(ldm_gaus_img)

            # Render: GT shape
            mesh = mesher.occ_meshing(
                decoder=get_occ_func(spaghetti, vis_gt_zcs[i]),
                res=256,
                get_time=False,
                verbose=False,
            )
            try:
                vert, face = list(map(lambda x: jutils.thutil.th2np(x), mesh))
                gt_img = jutils.fresnelvis.renderMeshCloud(
                    mesh={"vert": vert / 2, "face": face}, **camera_kwargs
                )
                gt_img = Image.fromarray(gt_img)
                img_per_shape.append(gt_img)
            except:
                pass

            # Render: LDM shape
            mesh = mesher.occ_meshing(
                decoder=get_occ_func(spaghetti, vis_ldm_zcs[i]),
                res=256,
                get_time=False,
                verbose=False,
            )
            try:
                vert, face = list(map(lambda x: jutils.thutil.th2np(x), mesh))
                ldm_img = jutils.fresnelvis.renderMeshCloud(
                    mesh={"vert": vert / 2, "face": face}, **camera_kwargs
                )
                ldm_img = Image.fromarray(ldm_img)
                img_per_shape.append(ldm_img)
            except:
                pass

            ####
            # for j in range(num_variations):
            #     mesh = mesher.occ_meshing(
            #         decoder=get_occ_func(
            #             spaghetti, vis_ldm_zcs[i * num_variations + j]
            #         ),
            #         res=128,
            #         get_time=False,
            #         verbose=False,
            #     )
            #     try:
            #         vert, face = list(map(lambda x: jutils.thutil.th2np(x), mesh))
            #         pred_img = jutils.fresnelvis.renderMeshCloud(
            #             mesh={"vert": vert / 2, "face": face}, **camera_kwargs
            #         )
            #         pred_img = Image.fromarray(pred_img)
            #         img_per_shape.append(pred_img)
            #     except:
            #         pass
            ####

            try:
                image = jutils.imageutil.merge_images(img_per_shape)
                wandb_logger.log_image("visualization", [image])
            except:
                pass
            
        return
        ####

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
        batch_size = self.hparams.batch_size

        ldm_zas = []
        print("[*] Start sampling")
        for i in range(int(np.ceil(num_shapes / batch_size))):
            batch_ldm_za = self.sample(batch_size)
            ldm_zas.append(batch_ldm_za)

        print("[*] Finished sampling")
        
        ldm_zas = torch.stack(ldm_zas, 0)
        return ldm_zas

    def validation(self):
        vis_num_shapes = 4
        vis_ldm_zas = self.sampling_za(vis_num_shapes)

        jutils.sysutil.clean_gpu()

        if not hasattr(self, "spaghetti"):
            #### < Seungwoo >
            # Added 'tag' argument
            # spaghetti = load_spaghetti(self.device)
            spaghetti = load_spaghetti(self.device, self.hparams.spaghetti_tag)
            ####
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
            gaus_img = jutils.visutil.render_gaussians(vis_ldm_gaus[i], resolution=camera_kwargs["resolution"])
            mesh = mesher.occ_meshing(
                    decoder=get_occ_func(spaghetti, vis_ldm_zcs[i]),
                    res=128,
                    get_time=False,
                    verbose=False
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


#### < Seungwoo >
# Stolen from 'network.py' to prevent overwriting the definition.
# The config file 'single_phase' is edited to refer to this class.
class LatentSelfAttentionNetwork(nn.Module):
    def __init__(self, input_dim, residual, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.residual = residual
        self.__dict__.update(kwargs)
        self.hparams = DotMap(self.__dict__)

        self._build_model()

    def _build_model(self):
        self.act = F.leaky_relu
        if self.hparams.get("use_timestep_embedder"):
            self.time_embedder = TimestepEmbedder(self.hparams.timestep_embedder_dim)
            dim_ctx = self.hparams.timestep_embedder_dim
        else:
            dim_ctx = 3

        """
        Encoder part
        """
        enc_dim = self.hparams.embedding_dim
        self.embedding = nn.Linear(self.hparams.input_dim, enc_dim)
        self.encoder = TimeTransformerEncoder(
            enc_dim,
            dim_ctx=dim_ctx,
            num_heads=self.hparams.num_heads if self.hparams.get("num_heads") else 4,
            use_time=True,
            num_layers=self.hparams.enc_num_layers,
            last_fc=True,

            #### < Seungwoo >
            # Replaced the hard-coded dim 16 to 528 to resolve dimension mismatch.
            last_fc_dim_out=528, 
            # last_fc_dim_out=16,
            ####
        )

    def forward(self, x, beta):
        """
        Input:
            x: [B,G,D] latent
            beta: B
        Output:
            eta: [B,G,D]
        """
        B, G = x.shape[:2]
        if self.hparams.get("use_timestep_embedder"):
            time_emb = self.time_embedder(beta).unsqueeze(1)
        else:
            beta = beta.view(B, 1, 1)
            time_emb = torch.cat(
                [beta, torch.sin(beta), torch.cos(beta)], dim=-1
            )  # [B,1,3]

        ctx = time_emb
        x_emb = self.embedding(x)

        out = self.encoder(x_emb, ctx=ctx)

        if self.hparams.residual:
            out = out + x
        return out
####