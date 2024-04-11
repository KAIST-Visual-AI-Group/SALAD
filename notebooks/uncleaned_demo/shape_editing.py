import argparse
from PIL import Image
from pathlib import Path

import h5py
import jutils
import numpy as np
from rich.progress import track
import torch

from pvd.diffusion.phase1 import GaussianSALDM
from pvd.diffusion.ldm import SpaghettiConditionSALDM
from pvd.utils.spaghetti_utils import (
    batch_gaus_to_gmms,
    get_occ_func,
    load_mesher,
    load_spaghetti,
)

from utils import *
from utils import decode_gmm_and_intrinsic


camera_kwargs = dict(
    camPos=np.array([-2, 2, -2]),
    camLookat=np.array([0, 0, 0]),
    camUp=np.array([0, 1, 0]),
    resolution=(512, 512),
    samples=32,
)

# GT semantic label for GMMs from SPAGHETTI checkpoint.
# Note that we do NOT assume any ordering, or correspondences
# for GMMs generated using our phase 1 model.
part_label = {
    # Manually found semantic labels for parts
    "back": [2, 10, 6, 14, 7, 15],
    "seat": [5, 13, 3, 11],
    "front_leg": [0, 1, 8, 9],
    "hind_leg": [4, 12],
    "arm": [4, 12],

    # Complements of semantic parts
    "back_compl": [0, 1, 3, 4, 5, 8, 9, 11, 12, 13],
    "seat_compl": [0, 1, 2, 4, 6, 7, 8, 9, 10, 12, 14],
    "front_leg_compl": [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14],
    "hind_leg_compl": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14],
    "arm_compl": [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_shape", type=int)
    parser.add_argument("n_variation", type=int)
    parser.add_argument("start_t", type=int)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_gt_gmm", action="store_true")
    parser.add_argument(
        "--removed_part",
        type=str,
        default=None,
        help="One of 'back', 'seat', 'front_leg', 'hind_leg', 'arm' and their '*_compl's",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    n_shape = args.n_shape
    n_variation = args.n_variation
    start_t = args.start_t
    batch_size = args.batch_size
    device = args.device
    debug = args.debug
    use_gt_gmm = args.use_gt_gmm
    removed_part = args.removed_part

    assert isinstance(n_shape, int), n_shape > 0
    if debug and n_shape > 10:
        print(
            f"[*] In debug mode, it is recommended to use less than 10 samples for speed."
        )
    if use_gt_gmm:
        assert removed_part is not None and isinstance(removed_part, str)
        if debug:
            print(f"Name of removed part(s): {removed_part}")

    # Print out
    print("====================================================================")
    print(f"[*] Number of shapes: {n_shape}")
    print(f"[*] Number of variations per shape: {n_variation}")
    print(f"[*] Starting t: {start_t}")
    print(f"[*] Use GT GMM: {use_gt_gmm}")
    print(f"[*] Device: {device}")
    print(f"[*] Debug: {debug}")
    print("====================================================================")

    # Create output directory
    out_dir = create_out_dir(Path("demo_out/edit"))
    if debug:
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        assert debug_dir.exists(), "Failed to create a debug directory"
        print(f"Created debug directory {str(debug_dir)}")

    # ================================================================================
    # Load phase 1 model
    p1_ckpt_dir = Path(
        "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/phase1/default/0204_204711/checkpoints"
    )
    assert p1_ckpt_dir.exists()

    p1_model = GaussianSALDM.load_from_checkpoint(
        str(p1_ckpt_dir / "last.ckpt"),  # Loads the last checkpoint file by default
        strict=False,
    )
    for p in p1_model.parameters():
        p.requires_grad_(False)
    p1_model.eval()
    p1_model = p1_model.to(device)

    print(f"Loaded model from {str(p1_ckpt_dir)}")
    print(f"Model type: {type(p1_model)}")
    # ================================================================================

    # ================================================================================
    # Load phase 2 model
    p2_ckpt_dir = Path(
        "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/spaghetti_condition_sa/cond_author_enc_sa_dec_sa_big_timestep_embedder/0202_100531/checkpoints"
    )
    assert p2_ckpt_dir.exists()

    p2_model = SpaghettiConditionSALDM.load_from_checkpoint(
        str(p2_ckpt_dir / "last.ckpt"),  # Loads the last checkpoint by default
        strict=False,
    )
    for p in p2_model.parameters():
        p.requires_grad_(False)
    p2_model.eval()
    p2_model = p2_model.to(device)

    print(f"Loaded model from {str(p2_ckpt_dir)}")
    print(f"Model type: {type(p2_model)}")
    # ================================================================================

    # ================================================================================
    # Load SPAGHETTI and mesher
    spaghetti_tag = (
        "chairs_large"  # Which shape category is used for SPAGHETTI training?
    )
    spaghetti = load_spaghetti(device, spaghetti_tag)
    print(f"SPAGHETTI type / tag: {type(spaghetti)} / {spaghetti_tag}")

    mesher = load_mesher(device)
    print(f"Mesher type: {type(mesher)}")
    # ================================================================================

    # ================================================================================
    # Compute batch splits
    batch_start_indices, batch_end_indices = compute_batch_sections(n_shape, batch_size)
    if debug:
        print(f"Batch start: {batch_start_indices}")
        print(f"Batch end: {batch_end_indices}")
    # ================================================================================

    # ================================================================================
    # Sample from Phase 1 model
    if use_gt_gmm:

        ####
        # data_path = Path(p1_model.hparams.dataset_kwargs.data_path)
        # data_keys = p1_model.hparams.dataset_kwargs.data_keys
        # assert data_path.exists(), f"Dataset path {str(data_path)} does not exist"
        # assert len(data_keys) == 1 and data_keys[0] == "g_js_affine", f"{data_keys}"
        ####

        # TODO: Remove the hard-coded path
        data_path = Path(
            "/home/juil/docker_home/projects/3D_CRISPR/crispr/data/pre_trained_spaghetti_latent_params/6755_2023-01-19-15-02-36.hdf5"
        )
        assert data_path.exists(), f"Dataset path {str(data_path)} does not exist"
        print("NOTE: The dataset path is HARD-CODED")

        with h5py.File(str(data_path), "r") as f:
            gmms = np.array(f["g_js_affine"][:])
            gmms = torch.from_numpy(gmms).to(device)
            gmms = gmms[:n_shape]
        print(f"Loaded GMM of shape: {gmms.shape}")
    else:
        gmms = []
        with torch.no_grad():
            for start, end in zip(batch_start_indices, batch_end_indices):
                batch_size = end - start
                batch_gmm = p1_model.sample(batch_size, return_traj=False)
                gmms.append(batch_gmm)
        gmms = torch.cat(gmms, dim=0)
        assert gmms.ndim == 3, f"Expected output to be 3D tensor got {gmms.ndim}D"
        assert (
            gmms.shape[0] == n_shape
        ), f"Expected the number of sampled GMMs to be {n_shape}, got {gmms.shape[0]}"
    assert (
        gmms.shape[0] == n_shape
    ), f"Number of shape do not match. Expected {n_shape}, got {gmms.shape[0]}"
    print(f"GMMs shape: {gmms.shape}")
    # ================================================================================

    # ================================================================================
    # Visualize sampled Gaussians in debug mode
    if debug:
        assert debug_dir.exists(), f"Debug directory {str(debug_dir)} does not exist"
        gmm_vis_dir = debug_dir / "gmm_vis"
        gmm_vis_dir.mkdir(parents=True, exist_ok=True)
        assert gmm_vis_dir.exists(), "Failed to create directory"

        for batch_idx, (start, end) in enumerate(
            zip(batch_start_indices, batch_end_indices)
        ):
            if batch_idx > 0:
                break  # Do NOT iterate over all data

            batch_gmm = gmms[start:end]
            assert batch_gmm.ndim == 3, f"Expected 3D tensor after indexing"

            images = []
            for gmm in batch_gmm:
                img = jutils.visutil.render_gaussians(
                    jutils.thutil.th2np(gmm), is_bspnet=False
                )
                images.append(img)
            images = make_img_grid([images], nrow=5)
            images.save(str(gmm_vis_dir / f"{batch_idx:05}.png"))
    # ================================================================================

    # ================================================================================
    # Debug eigenvalue clipping functionality
    if debug:
        assert debug_dir.exists(), f"Debug directory {str(debug_dir)} does not exist"
        eigval_clip_dir = debug_dir / "eigval_clip"
        eigval_clip_dir.mkdir(parents=True, exist_ok=True)
        assert eigval_clip_dir.exists(), "Failed to create directory"

        if isinstance(gmms, torch.Tensor):
            eigvals = gmms[..., 13:16]
            indices = torch.where(eigvals < 0.0)

            gmms_clipped = clip_gmm_eigvals(gmms)
            clipped_eigvals = gmms_clipped[..., 13:16]
            assert torch.all(clipped_eigvals[indices] == 1e-4)

        elif isinstance(gmms, np.ndarray):
            eigvals = gmms[..., 13:16]
            indices = np.where(eigvals < 0.0)

            gmms_clipped = clip_gmm_eigvals(gmms)
            clipped_eigvals = gmms_clipped[..., 13:16]
            assert np.all(clipped_eigvals[indices] == 1e-4)

        else:
            raise AssertionError(f"Expected tensor or array. Got {type(gmms)}")

        for batch_idx, (start, end) in enumerate(
            zip(batch_start_indices, batch_end_indices)
        ):
            if batch_idx > 0:
                break  # Do NOT iterate over all data

            batch_gmm = gmms[start:end]
            batch_clipped_gmm = gmms_clipped[start:end]
            assert batch_gmm.ndim == 3, "Expected 3D tensor after indexing"
            assert batch_clipped_gmm.ndim == 3, "Expected 3D tensor after indexing"

            orig_images = []
            for gmm in batch_gmm:
                img = jutils.visutil.render_gaussians(
                    jutils.thutil.th2np(gmm), is_bspnet=False
                )
                orig_images.append(img)

            clipped_images = []
            for gmm in batch_clipped_gmm:
                img = jutils.visutil.render_gaussians(
                    jutils.thutil.th2np(gmm), is_bspnet=False
                )
                clipped_images.append(img)

            images = make_img_grid([orig_images, clipped_images], nrow=5)
            images.save(str(eigval_clip_dir / f"{batch_idx:05}.png"))

        # Garbage collection
        _ = gmms_clipped.cpu()
        jutils.sysutil.clean_gpu()
    # ================================================================================

    # ================================================================================
    # Sample from Phase 2 model
    zhs = []
    with torch.no_grad():
        for start, end in zip(batch_start_indices, batch_end_indices):
            batch_gmm = gmms[start:end]
            batch_zh = p2_model.sample(batch_gmm, return_traj=False)
            assert (
                batch_zh.ndim == 3
            ), f"Expected 3D tensor as output, got {batch_zh.shape}"
            assert (
                batch_gmm.shape[0] == batch_zh.shape[0]
            ), "The number of GMM and intrinsic must be the same"
            zhs.append(batch_zh)
        zhs = torch.cat(zhs, dim=0)
    assert (
        zhs.ndim == 3
        and gmms.size(0) == zhs.size(0)
        and gmms.size(1) == zhs.size(1)
        and gmms.size(2) == 16
        and zhs.size(2) == 512
    )
    if debug:
        print(f"Intrinsics shape: {zhs.shape}")
    # ================================================================================

    # ================================================================================
    # Visualize decoded shapes in debug mode
    if debug:
        assert debug_dir.exists(), "Debug directory does not exist"
        vis_shape_dir = debug_dir / "decoded_shape"
        vis_shape_dir.mkdir(parents=True, exist_ok=True)
        assert vis_shape_dir.exists(), "Failed to create directory"

        for batch_idx, (start, end) in enumerate(
            zip(batch_start_indices, batch_end_indices)
        ):
            if batch_idx > 0:
                break
            batch_gmm = gmms[start:end]
            batch_zh = zhs[start:end]

            gmm_images = []
            shape_images = []
            for gmm, zh in zip(batch_gmm, batch_zh):
                vert, face = decode_gmm_and_intrinsic(
                    spaghetti, mesher, gmm[None], zh[None]
                )

                gmm_image = jutils.visutil.render_gaussians(
                    jutils.thutil.th2np(gmm), is_bspnet=False
                )
                gmm_images.append(gmm_image)

                shape_image = jutils.fresnelvis.renderMeshCloud(
                    mesh={"vert": vert / 2, "face": face}, **camera_kwargs
                )
                shape_images.append(Image.fromarray(shape_image))
            image = make_img_grid([gmm_images, shape_images], nrow=5)
            image.save(str(vis_shape_dir / f"{batch_idx:05}.png"))
    # ================================================================================

    # ================================================================================
    # Ablate parts
    if use_gt_gmm:
        gmms_ablate, ablated_indices = ablate_semantic_parts(
            gmms, part_label[removed_part]
        )
    else:
        # NOTE
        # The variable 'gmms_ablate' only includes the part parameters that are NOT ablated.
        # In the editing setup such as SDEdit which adds noises to the original data,
        # use 'gmms' directly. The variable MUST not be modified before and after the function call
        gmms_ablate, ablated_indices = ablate_symmetric_parts(gmms, 0)
    if debug:
        assert debug_dir.exists()
        abl_sym_dir = debug_dir / "abl_sym"
        abl_sym_dir.mkdir(parents=True, exist_ok=True)
        assert abl_sym_dir.exists(), "Failed to create directory"

        print(f"GMMs ablated shape: {gmms_ablate.shape}")
        print(f"Ablated indices shape: {ablated_indices.shape}")

        for batch_idx, (start, end) in enumerate(
            zip(batch_start_indices, batch_end_indices)
        ):
            if batch_idx > 0:
                break
            # For visualization purpose, clip eigenvalues
            batch_gmm = clip_gmm_eigvals(gmms[start:end])
            batch_gmm_abl = clip_gmm_eigvals(gmms_ablate[start:end])

            orig_images = []
            ablate_images = []
            for gmm_orig, gmm_abl in zip(batch_gmm, batch_gmm_abl):
                orig_img = jutils.visutil.render_gaussians(
                    jutils.thutil.th2np(gmm_orig), is_bspnet=False
                )
                abl_img = jutils.visutil.render_gaussians(
                    jutils.thutil.th2np(gmm_abl), is_bspnet=False
                )
                orig_images.append(orig_img)
                ablate_images.append(abl_img)
            images = make_img_grid([orig_images, ablate_images])
            images.save(str(abl_sym_dir / f"{batch_idx:05}.png"))
    # ================================================================================

    # ================================================================================
    # Visualize shapes before and after part ablation
    with torch.no_grad():
        intrinsics = p2_model.sample(gmms, return_traj=False)
        intrinsics_ablate = p2_model.sample(gmms_ablate, return_traj=False)
    if debug:
        print(f"'intrinsics' shape: {intrinsics.shape}")
        print(f"'intrinsics_ablate' shape: {intrinsics_ablate.shape}")
    """
        zcs, _ = spaghetti.merge_zh(
            intrinsics, batch_gaus_to_gmms(gmms, device)
        )
        zcs_ablate, _ = spaghetti.merge_zh(
                intrinsics_ablate, batch_gaus_to_gmms(gmms_ablate, device)
            )
    print(f"'zcs' shape: {zcs.shape}")
    print(f"'zcs_ablate' shape: {zcs_ablate.shape}")
        
    gmm_images = []  # Collect GMM images before ablation
    gmm_abl_images = []  # Collect GMM images after ablation
    shape_images = []  # Collect decoded shape images before ablation
    shape_abl_images = []  # Collect decoded shape images after ablation
    
    for idx, (gmm, zc, gmm_abl, zc_abl) in enumerate(zip(gmms, zcs, gmms_ablate, zcs_ablate)):
        mesh = mesher.occ_meshing(
            decoder=get_occ_func(spaghetti, zc),
            res=256,
            get_time=False,
            verbose=False
        )
        assert mesh is not None, "Marching cube failed"
        vert, face = list(map(lambda x: jutils.thutil.th2np(x), mesh))
        if debug:
            print(f"Vert: {vert.shape} / Face: {face.shape}")
        
        # Garbage collection
        _ = zc.cpu()
        jutils.sysutil.clean_gpu()
    
        gmm_image = jutils.visutil.render_gaussians(clip_gmm_eigvals(gmm[None])[0], is_bspnet=False)
        shape_image = jutils.fresnelvis.renderMeshCloud(
                mesh={"vert": vert / 2, "face": face}, **camera_kwargs
        )
        gmm_images.append(gmm_image)
        shape_images.append(Image.fromarray(shape_image))
    
        mesh_abl = mesher.occ_meshing(
            decoder=get_occ_func(spaghetti, zc_abl),
            res=256,
            get_time=False,
            verbose=False,
        )
        assert mesh_abl is not None, "Marching cube failed"
        vert_abl, face_abl = list(map(lambda x: jutils.thutil.th2np(x), mesh_abl))
        if debug:
            print(f"Vert: {vert_abl.shape} / Face: {face_abl.shape}")
    
        # Garbage collection
        _ = zc_abl.cpu()
        jutils.sysutil.clean_gpu()
    
        gmm_abl_image = jutils.visutil.render_gaussians(clip_gmm_eigvals(gmm_abl[None])[0], is_bspnet=False)
        shape_abl_image = jutils.fresnelvis.renderMeshCloud(
                mesh={"vert": vert_abl / 2, "face": face_abl}, **camera_kwargs
            ) 
        gmm_abl_images.append(gmm_abl_image)
        shape_abl_images.append(Image.fromarray(shape_abl_image))
        
    image_to_draw = make_img_grid([gmm_images, shape_images, gmm_abl_images, shape_abl_images], nrow=2)
    # display(image_to_draw)
    # ================================================================================
    """

    # ================================================================================
    # SDEdit based shape editing
    assert out_dir.exists()
    edit_result_dir = out_dir / "edit_result_dir"
    edit_result_dir.mkdir(parents=True, exist_ok=True)
    assert edit_result_dir.exists(), "Failed to create directory"

    for batch_idx, (start, end) in enumerate(
        track(zip(batch_start_indices, batch_end_indices))
    ):

        gmm_orig_images = []
        orig_shape_images = []
        part_images = []
        gmm_edit_images = []
        resampled_shape_images = []

        edit_shape_images = []

        for var_idx in range(n_variation):
            gmm_var_edit_images = []
            resampled_var_shape_images = []

            edit_var_shape_images = []

            batch_gmm = gmms[start:end]
            batch_zh = zhs[start:end]
            batch_abl_indices = ablated_indices[start:end]
            assert batch_gmm.size(0) == batch_abl_indices.size(
                0
            ), f"Got tensors of shape {batch_gmm.shape}, {batch_abl_indices.shape}"
            assert batch_zh.size(0) == batch_abl_indices.size(
                0
            ), f"Got tensors of shape {batch_gmm.shape}, {batch_abl_indices.shape}"

            for gmm_idx, (gmm, zh) in enumerate(zip(batch_gmm, batch_zh)):
                # Edit Gaussians
                edit_indices = batch_abl_indices[gmm_idx]
                gmm_edit = edit_gmm(
                    p1_model, gmm[None], edit_indices, start_t, verbose=False
                )[0]

                gmm_orig_clipped = clip_gmm_eigvals(gmm[None])
                gmm_edit_clipped = clip_gmm_eigvals(gmm_edit[None])
                
                if var_idx == 0:  # Render original Gaussians and their decoded shapes

                    # Render Gaussians
                    gmm_orig_img = jutils.visutil.render_gaussians(
                        jutils.thutil.th2np(gmm_orig_clipped[0]), is_bspnet=False
                    )
                    assert isinstance(gmm_orig_img, Image.Image)
                    gmm_orig_images.append(gmm_orig_img)

                    # Render edited parts
                    part_img = jutils.visutil.render_gaussians(
                        jutils.thutil.th2np(gmm_orig_clipped[0, edit_indices]),
                        is_bspnet=False,
                    )
                    assert isinstance(part_img, Image.Image)
                    part_images.append(part_img)

                    zh_orig = zh[None]

                    # Render decoded shape
                    vert_orig, face_orig = decode_gmm_and_intrinsic(
                        spaghetti, mesher, gmm[None], zh_orig
                    )
                    shape_orig_image = jutils.fresnelvis.renderMeshCloud(
                        mesh={"vert": vert_orig / 2, "face": face_orig}, **camera_kwargs
                    )
                    orig_shape_images.append(Image.fromarray(shape_orig_image))

                gmm_edit_img = jutils.visutil.render_gaussians(
                    jutils.thutil.th2np(gmm_edit_clipped[0]), is_bspnet=False
                )
                gmm_var_edit_images.append(gmm_edit_img)

                # SDEdit-style editing on phase 2
                zh_resampled = p2_model.sample(gmm_edit[None], return_traj=False)

                zh_edited = edit_zh(
                    p2_model, zh[None], gmm_edit[None], edit_indices, start_t
                )

                # Render shape decoded with resampled intrinsics
                vert_resampled, face_resampled = decode_gmm_and_intrinsic(
                    spaghetti, mesher, gmm_edit[None], zh_resampled
                )
                resampled_image = jutils.fresnelvis.renderMeshCloud(
                    mesh={"vert": vert_resampled / 2, "face": face_resampled}, **camera_kwargs
                )
                resampled_var_shape_images.append(Image.fromarray(resampled_image))

                # Render shape decoded with edited intrinsics
                vert_edited, face_edited = decode_gmm_and_intrinsic(
                    spaghetti, mesher, gmm_edit[None], zh_edited
                )
                shape_edit_image = jutils.fresnelvis.renderMeshCloud(
                    mesh={"vert": vert_edited / 2, "face": face_edited}, **camera_kwargs
                )
                edit_var_shape_images.append(Image.fromarray(shape_edit_image))
            assert len(gmm_var_edit_images) == len(batch_gmm) and len(
                edit_var_shape_images
            ) == len(batch_gmm)

            gmm_edit_images.append(gmm_var_edit_images)
            edit_shape_images.append(edit_var_shape_images)
            resampled_shape_images.append(resampled_var_shape_images)

        # Collect all images and draw
        image_lists = [gmm_orig_images, part_images, orig_shape_images]
        for var_idx in range(n_variation):
            image_lists.append(gmm_edit_images[var_idx])
            image_lists.append(resampled_shape_images[var_idx])
            image_lists.append(edit_shape_images[var_idx])
        image = make_img_grid(image_lists, nrow=1)
        image.save(str(edit_result_dir / f"{batch_idx:05}.png"))
    # ================================================================================

    print("Done")


if __name__ == "__main__":
    main()
