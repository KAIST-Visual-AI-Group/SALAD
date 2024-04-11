"""
sdedit.py

A script for demonstrating part regeneration using the proposed method.
"""

import argparse
from PIL import Image
from pathlib import Path
from time import time
from typing import List

import jutils
import numpy as np

from pvd.utils.spaghetti_utils import clip_eigenvalues, project_eigenvectors
from pvd.utils.spaghetti_utils import batch_gmms_to_gaus

from demo.utils.io_utils import (
    load_phase1_and_phase2,
    load_spaghetti_and_mesher,
)
from pvd.diffusion.single_phase import SingleZbSALDM

import demo.utils.paths as demo_paths
from demo.utils.vis import decode_and_render
from demo.utils import *
from scripts.utils import log_args


# camera parameters for rendering
camera_kwargs = dict(
    camPos=np.array([-2, 2, -2]),
    camLookat=np.array([0, 0, 0]),
    camUp=np.array([0, 1, 0]),
    resolution=(512, 512),
    samples=32,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_type",
        type=str,
        help="Type of the model used. Can be either ('single_phase', 'cascaded', 'zb')",
    )
    parser.add_argument(
        "shape_category",
        type=str,
        help="Type of the shapes. Can be either ('chairs', 'airplanes', 'tables')",
    )
    parser.add_argument(
        "n_shape", type=int, help="Number of shapes used for editing"
    )
    parser.add_argument(
        "n_variation", type=int, help="Number of variations per part"
    )
    parser.add_argument(
        "start_t", type=int, help="Number of SDEdit step"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./demo_out/sdedit",
        help="Root directory of log directories",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device used for the experiment"
    )
    parser.add_argument(
        "--debug", action="store_true", help="A flag for enabling debug mode"
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="Batch size for inference"
    )

    args = parser.parse_args()

    # Validate args
    assert args.model_type in (
        "single_phase",
        "cascaded",
        "zb",
    ), f"Unsupported model type: {str(args.model_type)}"
    assert args.shape_category in (
        "airplanes",
        "chairs",
        "tables",
    ), f"Unsupported model type: {str(args.shape_category)}"
    assert isinstance(args.n_shape, int) and args.n_shape > 0
    assert isinstance(args.batch_size, int) and args.batch_size > 0

    return args


def main():
    """
    Entry point of the script.
    """
    args = parse_args()
    log_dir = create_out_dir(
        Path(args.out_root) / str(args.shape_category) / str(args.model_type)
    )
    log_args(args, str(log_dir / "args.txt"))

    if args.model_type == "cascaded":
        run_sdedit_cascaded(args, log_dir)
    elif args.model_type == "single_phase":
        run_sdedit_single_phase(args, log_dir)
    elif args.model_type == "zb":
        run_sdedit_zb(args, log_dir)
    else:
        raise NotImplementedError(
            f"Unsupported model architecture: {str(args.model_type)}"
        )


def run_sdedit_cascaded(args, log_dir):

    # Load data
    start_t = time()
    gmms, zhs, _, part_labels, shape_ids = select_data_for_demo(
        *load_data_for_demo(
            args.shape_category, args.device
        ),
        args.n_shape,
    )
    args.n_shape = gmms.shape[0]
    elapsed = time() - start_t
    print(f"[Data Loading] Took {elapsed:3f} seconds")
    
    # Log shape IDs used for the experiment
    with open(log_dir / "shape_ids.txt", "w") as f:
        f.writelines([l + "\n" for l in shape_ids])
    
    # Load model
    start_t = time()
    phase1, phase2, spaghetti, mesher = load_models_cascaded(
        args.shape_category, args.device
    )
    elapsed = time() - start_t
    print(f"[Model Loading] Took {elapsed:3f} seconds")
    print(type(phase1), type(phase2), type(spaghetti), type(mesher))

    # SDEdit: Phase 1
    start_t = time()
    gmm_variations, gmm_splits, gmm_masks = loop_sdedit_uncond(phase1, gmms, part_labels, args.n_variation, args.start_t)
    elapsed = time() - start_t
    print(f"[SDEdit Phase 1] Took {elapsed:3f} seconds")
    
    with h5py.File(log_dir / "phase1_output.hdf5", "w") as f:
        f.create_dataset("gmm_variations", data=gmm_variations.cpu().numpy())
        f.create_dataset("gmm_splits", data=gmm_splits.cpu().numpy())
        f.create_dataset("gmm_masks", data=gmm_masks.cpu().numpy())
        f.create_dataset("part_labels", data=part_labels.cpu().numpy())
        f.create_dataset("shape_ids", data=shape_ids)
    
    # Visualize Gaussians
    if args.debug:  
        gmm_img_dir = log_dir / "gmm_img"
        gmm_img_dir.mkdir(parents=True, exist_ok=True)
        loop_visualize_gmms(
            gmms,
            gmm_variations,
            gmm_splits,
            gmm_masks,
            shape_ids,
            gmm_img_dir,
        )

    # SDEdit: Phase 2
    start_t = time()
    zh_variations = loop_sdedit_cond(phase2, zhs, gmm_variations, gmm_splits, gmm_masks, args.start_t)
    elapsed = time() - start_t
    print(f"[SDEdit Phase 2] Took {elapsed:3f} seconds")
    torch.save(
        zh_variations,
        log_dir / "zh_variations.pt",
    )

    with h5py.File(log_dir / "phase2_output.hdf5", "w") as f:
        # Source GMMs & Zhs
        f.create_dataset("gmms", data=gmms.cpu().numpy())
        f.create_dataset("zhs", data=zhs.cpu().numpy())
        
        # Variations
        f.create_dataset("gmm_variations", data=gmm_variations.cpu().numpy())
        f.create_dataset("zh_variations", data=zh_variations.cpu().numpy())
        
        # Auxilary info for editing
        f.create_dataset("gmm_splits", data=gmm_splits.cpu().numpy())
        f.create_dataset("gmm_masks", data=gmm_masks.cpu().numpy())

        # IDs
        f.create_dataset("part_labels", data=part_labels.cpu().numpy())
        f.create_dataset("shape_ids", data=shape_ids)

    # Decode and save meshes
    mesh_dir = log_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    start_t = time()
    loop_decode_and_save(
        gmm_variations, 
        zh_variations, 
        part_labels, 
        shape_ids, 
        gmm_splits, 
        mesh_dir, 
        spaghetti, 
        mesher,
    )
    elapsed = time() - start_t
    print(f"[Decode] Took {elapsed:3f} seconds")

    if args.debug:
        vis_dir = log_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        n_vis = min(5, gmm_variations.size(0))
        print(f"Visualizing {n_vis} shapes")

        common_args = spaghetti, mesher, camera_kwargs
        for vis_idx, (gmm, zh) in enumerate(zip(gmm_variations, zh_variations)):
            img = decode_and_render(
                gmm, zh, *common_args
            )
            img.save(vis_dir / f"{vis_idx:03}.png")            


def run_sdedit_single_phase(args, log_dir):

    # Load data
    start_t = time()
    gmms, zhs, _, part_labels, shape_ids = select_data_for_demo(
        *load_data_for_demo(
            args.shape_category, args.device
        ),
        args.n_shape,
    )
    args.n_shape = gmms.shape[0]
    elapsed = time() - start_t
    print(f"[Data Loading] Took {elapsed:3f} seconds")

    # Log shape IDs used for the experiment
    with open(log_dir / "shape_ids.txt", "w") as f:
        f.writelines([l + "\n" for l in shape_ids])

    # Load model
    start_t = time()
    model, spaghetti, mesher = load_models_single_phase(
        args.shape_category, args.device
    )
    elapsed = time() - start_t
    print(f"[Model Loading] Took {elapsed:3f} seconds")
    print(type(model), type(spaghetti), type(mesher))

    # SDEdit: Single Phase
    start_t = time()
    variations, splits, masks = loop_sdedit_single_phase(
        model, torch.cat([zhs, gmms], -1), part_labels, args.n_variation, args.start_t
    )
    elapsed = time() - start_t
    print(f"[SDEdit Single Phase] Took {elapsed:3f} seconds")
    zh_variations, gmm_variations = variations.split([zhs.size(-1), gmms.size(-1)], dim=-1)

    torch.save(
        zh_variations,
        log_dir / "zh_variations.pt"
    )
    torch.save(
        gmm_variations,
        log_dir / "gmm_variations.pt"
    )

    # Decode and save meshes
    mesh_dir = log_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    start_t = time()
    loop_decode_and_save(
        gmm_variations, 
        zh_variations, 
        part_labels, 
        shape_ids, 
        splits, 
        mesh_dir, 
        spaghetti, 
        mesher,
    )
    elapsed = time() - start_t
    print(f"[Decode] Took {elapsed:3f} seconds")

    if args.debug:
        vis_dir = log_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        n_vis = min(5, gmm_variations.size(0))
        print(f"Visualizing {n_vis} shapes")

        common_args = spaghetti, mesher, camera_kwargs
        for vis_idx, (gmm, zh) in enumerate(zip(gmm_variations, zh_variations)):
            img = decode_and_render(
                gmm, zh, *common_args
            )
            img.save(vis_dir / f"{vis_idx:03}.png")            


def run_sdedit_zb(args, log_dir):

    # Load data
    start_t = time()
    _, _, zbs, part_labels, shape_ids = select_data_for_demo(
        *load_data_for_demo(
            args.shape_category, args.device
        ),
        args.n_shape,
    )
    elapsed = time() - start_t
    print(f"[Data Loading] Took {elapsed:3f} seconds")

    # Log shape IDs used for the experiment
    with open(log_dir / "shape_ids.txt", "w") as f:
        f.writelines([l + "\n" for l in shape_ids])

    # Load model
    start_t = time()
    if args.shape_category == "chairs":
        model = SingleZbSALDM.load_from_checkpoint(
            str(demo_paths.chair_zb_ckpt_dir / "epoch=9199-val_loss=0.0000.ckpt"), strict=False
        )
        spaghetti, mesher = load_spaghetti_and_mesher(args.device, "chairs_large")
    elif args.shape_category == "airplanes":
        model = SingleZbSALDM.load_from_checkpoint(
            str(demo_paths.airplane_zb_ckpt_dir / "epoch=9999-val_loss=0.0000.ckpt"), strict=False
        )
        spaghetti, mesher = load_spaghetti_and_mesher(args.device, "airplanes")
    else:
        raise NotImplementedError()
    model = model.to(args.device)
    
    elapsed = time() - start_t
    print(f"[Model Loading] Took {elapsed:3f} seconds")
    
    # SDEdit: Single Phase
    start_t = time()
    zb_variations, splits, masks = loop_sdedit_zb(
        model, zbs, part_labels, args.n_variation, args.start_t
    )
    elapsed = time() - start_t
    print(f"[SDEdit Single Phase] Took {elapsed:3f} seconds")

    torch.save(
        zb_variations,
        log_dir / "zb_variations.pt"
    )

    # Decode and save meshes
    mesh_dir = log_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    start_t = time()

    loop_decode_zb_save(
        zb_variations,
        part_labels,
        shape_ids,
        splits,
        mesh_dir,
        spaghetti,
        mesher,
    )

    elapsed = time() - start_t
    print(f"[Decode] Took {elapsed:3f} seconds")


def loop_decode_zb_save(
    zb_variations,
    part_labels,
    shape_ids,
    splits,
    mesh_dir,
    spaghetti,
    mesher,        
):
    starts, ends = splits[0:-1], splits[1:]

    for idx, (start, end) in enumerate(zip(starts, ends)):
    
        # retrieve GMMs and intrinsics of the current shape
        shape_zb = zb_variations[start:end]

        # decode: zb -> GMM, Zh
        shape_zh, shape_gmm = spaghetti.decomposition_control.forward_mid(shape_zb)
        shape_gmm = batch_gmms_to_gaus(shape_gmm)

        # retrieve the ShapeNet ID of the current shape
        shape_id = shape_ids[idx]
    
        # look-up the number of parts in the current shape
        # and calculate the number of variations made for each part
        part_ids = part_labels[idx].unique()
        n_variation = (end - start).item() // len(part_ids)
        assert n_variation * len(part_ids) == (end - start), (
            f"{n_variation * len(part_ids)} {(end - start)}"
        )

        # iterate over the parts, and save their variations
        for part_idx in range(len(part_ids)):

            # NOTE: Add 12 to the part ID as our preprocessing script
            # subtracted it from the original ShapeNet part ID.
            # 
            # This hard-coded number should be removed later by modifying
            # the preprocessing script.
            part_id = part_ids[part_idx] + 12

            for var_id in range(n_variation):
                gmm = shape_gmm[n_variation * part_idx + var_id]
                zh = shape_zh[n_variation * part_idx + var_id]

                vert, face = decode_gmm_and_intrinsic(
                    spaghetti, mesher, gmm[None], zh[None]
                )

                jutils.meshutil.write_obj_triangle(
                    str(mesh_dir / f"{shape_id}_{str(int(part_id))}_var{str(int(var_id))}.obj"),
                    vert,
                    face,
                )

@torch.no_grad()
def loop_sdedit_uncond(
        model, 
        gmms: torch.Tensor, 
        part_labels: torch.Tensor, 
        n_variation_per_part: int,
        start_t: int,
    ):
    """
    A helper function for SDEdit that loops over the shapes in the dataset.

    This function is used to edit latents using unconditional diffusion models.
    """
    assert isinstance(gmms, torch.Tensor)
    assert isinstance(part_labels, torch.Tensor)
    assert isinstance(n_variation_per_part, int) and n_variation_per_part > 0
    assert isinstance(start_t, int) and start_t > 0
    assert gmms.size(0) == part_labels.size(0), f"{gmms.size(0)} != {part_labels.size(0)}"

    # iterate over shapes, make variations
    splits = [0]
    gmm_variations = []
    masks = []
    for (gmm, part_label) in zip(gmms, part_labels):
        mask = create_mask(gmm, part_label, n_variation_per_part)
        n_mask = mask.size(0)

        gmm_to_edit = gmm[None].clone().repeat(n_mask, 1, 1)
        gmm_variation = sdedit_gmms(model, gmm_to_edit, mask, start_t, verbose=False)

        splits.append(n_mask)
        gmm_variations.append(gmm_variation)
        masks.append(mask)

    splits = torch.cumsum(torch.tensor(splits), 0)
    gmm_variations = torch.cat(gmm_variations, 0)
    masks = torch.cat(masks, 0)

    return gmm_variations, splits, masks


@torch.no_grad()
def loop_sdedit_zb(
        model, 
        zbs: torch.Tensor, 
        part_labels: torch.Tensor, 
        n_variation_per_part: int,
        start_t: int,
    ):
    """
    A helper function for SDEdit that loops over the shapes in the dataset.

    This function is used to edit latents using unconditional diffusion models.
    """
    assert isinstance(zbs, torch.Tensor)
    assert isinstance(part_labels, torch.Tensor)
    assert isinstance(n_variation_per_part, int) and n_variation_per_part > 0
    assert isinstance(start_t, int) and start_t > 0
    assert zbs.size(0) == part_labels.size(0), f"{zbs.size(0)} != {part_labels.size(0)}"

    # iterate over shapes, make variations
    splits = [0]
    gmm_variations = []
    masks = []
    for (gmm, part_label) in zip(zbs, part_labels):
        mask = create_mask(gmm, part_label, n_variation_per_part)
        n_mask = mask.size(0)

        gmm_to_edit = gmm[None].clone().repeat(n_mask, 1, 1)
        gmm_variation = edit_zb(model, gmm_to_edit, mask, start_t, verbose=False)

        splits.append(n_mask)
        gmm_variations.append(gmm_variation)
        masks.append(mask)

    splits = torch.cumsum(torch.tensor(splits), 0)
    gmm_variations = torch.cat(gmm_variations, 0)
    masks = torch.cat(masks, 0)

    return gmm_variations, splits, masks


@torch.no_grad()
def loop_sdedit_cond(
        model,
        zhs: torch.Tensor, 
        gmm_variations: torch.Tensor, 
        gmm_splits: torch.Tensor,
        gmm_masks: torch.Tensor,
        start_t: int,
    ):
    """
    A helper function for SDEdit that loops over the shapes in the dataset.

    This function is used to edit latents using conditional diffusion models.    
    """
    assert isinstance(zhs, torch.Tensor)
    assert isinstance(gmm_variations, torch.Tensor)
    assert isinstance(gmm_splits, torch.Tensor)
    assert isinstance(gmm_masks, torch.Tensor)
    assert zhs.size(0) == gmm_splits.size(0) - 1, f"{zhs.size(0)} != {gmm_splits.size(0) - 1}"
    assert gmm_variations.size(0) == gmm_masks.size(0), (
        f"{gmm_variations.size(0)} != {gmm_masks.size(0)}"
    )


    starts, ends = gmm_splits[0:-1], gmm_splits[1:]

    zh_variations = []
    for shape_idx, (start, end) in enumerate(zip(starts, ends)):
        current_shape_gmm = gmm_variations[start:end]
        current_shape_mask = gmm_masks[start:end]
        current_shape_zh = zhs[shape_idx].clone()[None].repeat(end - start, 1, 1)

        # Hacky way to turn GMM masks for Zh masks.
        # GMM dim = 16, Zh dim = 512 -> (Zh dim / GMM dim) = 32
        current_shape_mask_zh = current_shape_mask.repeat(1, 1, 32)
        assert current_shape_mask_zh.shape == current_shape_zh.shape, (
            f"{current_shape_mask_zh.shape} != {current_shape_zh.shape}"
        )

        current_shape_zh_variation = edit_cond(
            model, 
            current_shape_zh,
            current_shape_gmm,
            current_shape_mask_zh,
            start_t,
            verbose=False,
        )
        zh_variations.append(current_shape_zh_variation)
    
    zh_variations = torch.cat(zh_variations, 0)

    return zh_variations


@torch.no_grad()
def loop_sdedit_single_phase(
        model, 
        gmms: torch.Tensor, 
        part_labels: torch.Tensor, 
        n_variation_per_part: int,
        start_t: int,
    ):
    """
    A helper function for SDEdit that loops over the shapes in the dataset.

    This function is used to edit latents using unconditional diffusion models.
    """
    assert isinstance(gmms, torch.Tensor)
    assert isinstance(part_labels, torch.Tensor)
    assert isinstance(n_variation_per_part, int) and n_variation_per_part > 0
    assert isinstance(start_t, int) and start_t > 0
    assert gmms.size(0) == part_labels.size(0), f"{gmms.size(0)} != {part_labels.size(0)}"

    # iterate over shapes, make variations
    splits = [0]
    gmm_variations = []
    masks = []
    for (gmm, part_label) in zip(gmms, part_labels):
        mask = create_mask(gmm, part_label, n_variation_per_part)
        n_mask = mask.size(0)

        gmm_to_edit = gmm[None].clone().repeat(n_mask, 1, 1)
        gmm_variation = sdedit_single_phase(model, gmm_to_edit, mask, start_t, verbose=False)

        splits.append(n_mask)
        gmm_variations.append(gmm_variation)
        masks.append(mask)

    splits = torch.cumsum(torch.tensor(splits), 0)
    gmm_variations = torch.cat(gmm_variations, 0)
    masks = torch.cat(masks, 0)

    return gmm_variations, splits, masks


@torch.no_grad()
def sdedit_single_phase(model, gmms, masks, start_t, method="sdedit", verbose=True):
    """
    Performs shape completion given incomplete sets of GMMs.

    Args:
        model:
        part_repr: (B, N, D). A batch of part-wise representations to be modified
        masks: (B, N, D). A batch of binary masks indicating the parts to be modified.
        start_t: The timestep where denoising process starts.
        method: 
        verbose:
    """
    assert gmms.shape[-1] == 528, f"{gmms.shape[-1]}"

    assert start_t >= 0 and start_t <= model.var_sched.num_steps
    assert method in ("mcg", "repaint", "sdedit")
    assert gmms.ndim == 3, f"Got 'gmms' of dimension {gmms.ndim}"
    assert gmms.shape == masks.shape, f"{gmms.shape} != {masks.shape}"

    timesteps = list(range(start_t, 0, -1))
    if verbose:
        # timesteps = tqdm(timesteps)
        # timesteps = track(timesteps)
        pass

    x0 = gmms.clone()

    # Process input if additional steps are required
    use_scaled_eigvec = model.hparams.get("use_scaled_eigenvectors")
    if use_scaled_eigvec:
        raise NotImplementedError("Use of scaled Gaussians is not supported yet")

    normalize_method = model.hparams.get("global_normalization")
    if normalize_method is not None:
        if not hasattr(model, "data_val"):
            model._build_dataset("val")
            if verbose:
                print(
                    "[*] Loaded statistics for Gaussians since the model was trained on normalized data"
                )

        if normalize_method == "partial":
            normalized_gaus = model.data_val.normalize_global_static(x0.detach().cpu().numpy()[..., 512:528], slice(12, None))
            x0[..., 512:528] = torch.from_numpy(normalized_gaus).to(x0.device)
        else:
            raise NotImplementedError(
                f"Encountered unknown normalization method: {str(normalize_method)}"
            )
        assert isinstance(x0, torch.Tensor), f"{type(x0)}"
        if not isinstance(x0, torch.Tensor):
            x0 = torch.from_numpy(x0).to(gmms.device)
        assert gmms.shape == x0.shape, (
            f"Shape of a tensor must not change during preprocessing, got {gmms.shape} and {x0.shape}"
        )

    x = add_noise(model, x0, start_t)

    for t in timesteps:
        if method == "mcg":
            x = x.requires_grad_()  # record gradient
            x_ = denoise_one_step(model, x, t)
        else:
            with torch.no_grad():
                x_ = denoise_one_step(model, x, t)

        if method == "mcg":
            raise NotImplementedError()
        elif method == "repaint":
            raise NotImplementedError()
        elif method == "sdedit":
            x = add_noise(model, x0, t - 1)
            x[masks == 1.0] = x_[masks == 1.0]
        else:
            raise NotImplementedError()
    assert x.shape == gmms.shape, f"'x' shape: {x.shape} / 'gmms' shape: {gmms.shape}"
    assert torch.all(x[masks == 0.0] == x0[masks == 0.0]), (
        "The algorithm must not edit the unmasked parts"
    )

    # if necessary, unnormalized the data
    if normalize_method is not None:
        if normalize_method == "partial":
            # x = model.data_val.unnormalize_global_static(x, slice(12, None))
            unnormalized_gaus = model.data_val.unnormalize_global_static(x[..., 512:528], slice(12, None))
            x[..., 512:528] = torch.from_numpy(unnormalized_gaus).to(x.device)
        else:
            raise NotImplementedError(
                f"Encountered unknown normalization method: {str(normalize_method)}"
            )
        assert isinstance(x, torch.Tensor), f"{type(x)}"

    # apply projection and clipping
    x[..., 512:528] = project_eigenvectors(clip_eigenvalues(x[..., 512:528])).to(x0.device)
    
    return x


@torch.no_grad()
def loop_visualize_gmms(
    gmm_orig: torch.Tensor,
    gmm_variations: torch.Tensor,
    gmm_splits: torch.Tensor,
    gmm_masks: torch.Tensor,
    shape_ids: List[str],
    out_dir: Path,
):
    """
    Args:
        gmm_orig: (N, M, 16), a set of original GMMs.
        gmm_variations: (N * M * n_variations, M, 16), a set of modified GMMs.
        gmm_splits: (N + 1,). A split indicating the number of shapes derived from the same shape.
        gmm_masks: (N * M * n_variations, M, 16), a set of binary masks indicating the parts to be modified.
    """
    starts, ends = gmm_splits[0:-1], gmm_splits[1:]

    for shape_idx, (start, end) in enumerate(zip(starts, ends)):

        # retrieve the ShapeNet ID of the current shape
        shape_id = shape_ids[shape_idx]

        # retrieve GMMs of the current shape
        curr_gmm_orig = gmm_orig[shape_idx]
        curr_gmm_variations = gmm_variations[start:end]
        curr_gmm_masks = gmm_masks[start:end]
        assert curr_gmm_variations.shape == curr_gmm_masks.shape, (
            f"{curr_gmm_variations.shape} != {curr_gmm_masks.shape}"
        )

        # render the source Gaussians
        orig_img = jutils.visutil.render_gaussians(
            curr_gmm_orig, is_bspnet=False
        )

        for var_idx in range(end - start):
            gmm_variation = curr_gmm_variations[var_idx]
            mask = curr_gmm_masks[var_idx]
            assert curr_gmm_orig.shape == mask.shape, (
                f"{curr_gmm_orig.shape} != {mask.shape}"
            )
            assert gmm_variation.shape == mask.shape, (
                f"{gmm_variation.shape} != {mask.shape}"
            )

            # render the final editing result
            variation_img = jutils.visutil.render_gaussians(
                clip_gmm_eigvals(gmm_variation[None])[0], is_bspnet=False
            )

            # render the selected Gaussians only
            curr_gmm_selected = curr_gmm_orig.clone()
            curr_gmm_selected = torch.where(
                mask == 0,
                0.0,
                curr_gmm_selected,
            )
            selection_img = jutils.visutil.render_gaussians(
                curr_gmm_selected, is_bspnet=False
            )

            # merge images and save
            img = jutils.imageutil.merge_images(
                [orig_img, selection_img, variation_img]
            )
            img.save(out_dir / f"{shape_id}_{var_idx}.png")


@torch.no_grad()
def loop_decode_and_save(
    gmms,
    zhs,
    part_labels,
    shape_ids,
    splits,
    mesh_dir,
    spaghetti,
    mesher,
):
    """
    Args:
        gmms: (N * n_variation * n_part_labels, 16, 16). A set of GMM variations created with SDEdit.
        zhs: (N * n_variation * n_part_labels, 16, 512). A set of intrinsic variations created with SDEdit.
        part_labels: (N, 16). A part labels for the original shapes.
        shape_ids: (N,). ShapeNet IDs of the original shapes.
        splits: (N + 1,). A split indicating the number of shapes derived from the same shape.
        mesh_dir: A directory to save output meshes.
        spaghetti: A pretrained SPAGHETTI used to decode the latents.
        mesher: A class implementing Marching Cubes algorithm.
    """
    assert gmms.size(0) == zhs.size(0), f"{gmms.size(0)} != {zhs.size(0)}"
    starts, ends = splits[0:-1], splits[1:]

    for idx, (start, end) in enumerate(zip(starts, ends)):
        
        # retrieve GMMs and intrinsics of the current shape
        shape_gmm = gmms[start:end]
        shape_zh = zhs[start:end]

        # retrieve the ShapeNet ID of the current shape
        shape_id = shape_ids[idx]
        
        # look-up the number of parts in the current shape
        # and calculate the number of variations made for each part
        part_ids = part_labels[idx].unique()
        n_variation = (end - start).item() // len(part_ids)
        assert n_variation * len(part_ids) == (end - start), (
            f"{n_variation * len(part_ids)} {(end - start)}"
        )

        # iterate over the parts, and save their variations
        for part_idx in range(len(part_ids)):
            
            # NOTE: Add 12 to the part ID as our preprocessing script
            # subtracted it from the original ShapeNet part ID.
            # 
            # This hard-coded number should be removed later by modifying
            # the preprocessing script.
            part_id = part_ids[part_idx] + 12

            for var_id in range(n_variation):
                gmm = shape_gmm[n_variation * part_idx + var_id]
                zh = shape_zh[n_variation * part_idx + var_id]
            
                vert, face = decode_gmm_and_intrinsic(
                    spaghetti, mesher, gmm[None], zh[None]
                )

                jutils.meshutil.write_obj_triangle(
                    str(mesh_dir / f"{shape_id}_{str(int(part_id))}_var{str(int(var_id))}.obj"),
                    vert,
                    face,
                )


def load_data_for_demo(shape_category: str, device):

    if shape_category == "chairs":
        gmms, zhs, zbs, part_labels, shape_ids = load_dataset(
            str(demo_paths.chair_data_path),
            device,
        )
    elif shape_category == "airplanes":
        gmms, zhs, zbs, part_labels, shape_ids = load_dataset(
            str(demo_paths.airplane_data_path),
            device,
        )
    elif shape_category == "tables":
        gmms, zhs, zbs, part_labels, shape_ids = load_dataset(
            str(demo_paths.table_data_path),
            device,
        )
    else:
        raise NotImplementedError(f"Unsupported shape type: {shape_category}")

    # Make sure to load data to device
    gmms = gmms.to(device).float()
    zhs = zhs.to(device).float()

    zbs = zbs.to(device).float()

    part_labels = part_labels.to(device)

    assert gmms.size(1) == zhs.size(1), f"{gmms.size(1)} != {zhs.size(1)}"
    assert gmms.size(1) == part_labels.size(1)

    return gmms, zhs, zbs, part_labels, shape_ids


#### 0308 Zb
# def select_data_for_demo(gmms, zhs, part_labels, shape_ids, n_shape):
def select_data_for_demo(gmms, zhs, zbs, part_labels, shape_ids, n_shape):
####

    # Validate arguments
    assert gmms.ndim == 3, f"{gmms.ndim}"
    assert zhs.ndim == 3, f"{zhs.ndim}"

    #### 0308 Zb
    assert zbs.ndim == 3, f"{zbs.ndim}"
    ####

    assert part_labels.ndim == 2, f"{part_labels}"
    
    n_total_shape = gmms.shape[0]
    assert n_total_shape == zhs.shape[0], f"{n_total_shape} vs {zhs.shape[0]}"
    assert n_total_shape == part_labels.shape[0], f"{n_total_shape} vs {part_labels.shape[0]}"
    assert n_total_shape == len(shape_ids), f"{n_total_shape} vs {len(shape_ids)}"

    n_part = gmms.shape[1]
    assert n_part == zhs.shape[1], f"{n_part} vs {zhs.shape[1]}"
    assert n_part == part_labels.shape[1], f"{n_part} vs {part_labels.shape[1]}"

    # Shapes for demo: the first N shapes in the dataset
    n_shape_for_demo = min(n_total_shape, n_shape)
    demo_gmms = gmms[:n_shape_for_demo]
    demo_zhs = zhs[:n_shape_for_demo]

    #### 0308
    demo_zbs = zbs[:n_shape_for_demo]
    ####

    demo_part_labels = part_labels[:n_shape_for_demo]
    demo_shape_ids = shape_ids[:n_shape_for_demo]
    print(f"[*] Using {n_shape_for_demo} shapes for demo")

    return demo_gmms, demo_zhs, demo_zbs, demo_part_labels, demo_shape_ids


def load_models_cascaded(shape_category: str, device):
    """
    Loads the models required for 'cascaded' setup.

    Loads the phase 1 and phase 2 models, as well as SPAGHETTI and mesher for shape decoding and rendering.
    """
    phase1, phase2 = load_phase1_and_phase2(shape_category, use_sym=False, device=device)

    if shape_category == "chairs":
        spaghetti, mesher = load_spaghetti_and_mesher(device, "chairs_large")
    elif shape_category == "airplanes":
        spaghetti, mesher = load_spaghetti_and_mesher(device, "airplanes")
    elif shape_category == "tables":
        spaghetti, mesher = load_spaghetti_and_mesher(device, "tables")
    else:
        raise NotImplementedError(f"Unknown shape category {shape_category}")        

    return phase1, phase2, spaghetti, mesher


def load_models_single_phase(shape_category: str, device):
    model = load_single_phase(shape_category, device=device)

    if shape_category == "chairs":
        spaghetti, mesher = load_spaghetti_and_mesher(device, "chairs_large")
    else:
        spaghetti, mesher = load_spaghetti_and_mesher(device, "airplanes")

    return model, spaghetti, mesher


def create_mask(part_repr: torch.Tensor, part_label: torch.Tensor, n_variation: int):
    """
    Args:
        part_repr: (N, D) where N is the number of parts.
        part_label: (N,) where N is the number of parts.
        n_variation: Number of variation made for each part.

    Returns:
        masks: (N * n_variation, N, D). Binary mask indicating the parts to be modified
            during denoising process.
    """
    assert isinstance(part_repr, torch.Tensor)
    assert isinstance(part_label, torch.Tensor)
    assert part_repr.ndim == 2, f"{part_repr.ndim}"
    assert part_label.ndim == 1, f"{part_label.ndim}"
    assert part_repr.size(0) == part_label.size(0), f"{part_repr.size(0)} != {part_label.size(0)}"
    assert isinstance(n_variation, int) and n_variation > 0

    part_ids = part_label.unique()
    
    masks = []
    for part_id in part_ids:
        # initialize an empty mask
        mask = torch.zeros_like(part_repr)

        # set the values of edited parts to 1
        edit_indices = torch.where(part_label == part_id)[0]
        mask[edit_indices, ...] = 1.0

        # duplicate the mask 
        mask = mask[None]
        mask = mask.repeat(n_variation, 1, 1)

        masks.append(mask)
    masks = torch.cat(masks, 0)
    assert masks.size(0) == len(part_ids) * n_variation, (
        f"{masks.size(0)} != {len(part_ids) * n_variation}"
    )

    return masks
    

@torch.no_grad()
def sdedit_gmms(model, gmms, masks, start_t, method="sdedit", verbose=True):
    """
    Performs shape completion given incomplete sets of GMMs.

    Args:
        model:
        part_repr: (B, N, D). A batch of part-wise representations to be modified
        masks: (B, N, D). A batch of binary masks indicating the parts to be modified.
        start_t: The timestep where denoising process starts.
        method: 
        verbose:
    """
    assert start_t >= 0 and start_t <= model.var_sched.num_steps
    assert method in ("mcg", "repaint", "sdedit")
    assert gmms.ndim == 3, f"Got 'gmms' of dimension {gmms.ndim}"
    assert gmms.shape == masks.shape, f"{gmms.shape} != {masks.shape}"

    timesteps = list(range(start_t, 0, -1))
    if verbose:
        # timesteps = tqdm(timesteps)
        # timesteps = track(timesteps)
        pass

    x0 = gmms.clone()

    # Process input if additional steps are required
    use_scaled_eigvec = model.hparams.get("use_scaled_eigenvectors")
    if use_scaled_eigvec:
        raise NotImplementedError("Use of scaled Gaussians is not supported yet")

    normalize_method = model.hparams.get("global_normalization")
    if normalize_method is not None:
        if not hasattr(model, "data_val"):
            model._build_dataset("val")
            if verbose:
                print(
                    "[*] Loaded statistics for Gaussians since the model was trained on normalized data"
                )

        if normalize_method == "partial":
            x0 = model.data_val.normalize_global_static(x0.detach().cpu().numpy(), slice(12, None))           
        elif normalize_method == "all":
            x0 = model.data_val.normalize_global_static(x0.detach().cpu().numpy(), slice(None))
        else:
            raise NotImplementedError(
                f"Encountered unknown normalization method: {str(normalize_method)}"
            )
        x0 = torch.from_numpy(x0).to(gmms.device)
        assert gmms.shape == x0.shape, (
            f"Shape of a tensor must not change during preprocessing, got {gmms.shape} and {x0.shape}"
        )

    x = add_noise(model, x0, start_t)

    for t in timesteps:
        if method == "mcg":
            x = x.requires_grad_()  # record gradient
            x_ = denoise_one_step(model, x, t)
        else:
            with torch.no_grad():
                x_ = denoise_one_step(model, x, t)

        if method == "mcg":
            raise NotImplementedError()
        elif method == "repaint":
            raise NotImplementedError()
        elif method == "sdedit":
            x = add_noise(model, x0, t - 1)
            x[masks == 1.0] = x_[masks == 1.0]
        else:
            raise NotImplementedError()
    assert x.shape == gmms.shape, f"'x' shape: {x.shape} / 'gmms' shape: {gmms.shape}"
    assert torch.all(x[masks == 0.0] == x0[masks == 0.0]), (
        "The algorithm must not edit the unmasked parts"
    )

    # if necessary, unnormalized the data
    if normalize_method is not None:
        if normalize_method == "partial":
            x = model.data_val.unnormalize_global_static(x, slice(12, None))
        elif normalize_method == "all":
            x = model.data_val.unnormalize_global_static(x, slice(None))
        else:
            raise NotImplementedError(
                f"Encountered unknown normalization method: {str(normalize_method)}"
            )
        x = torch.from_numpy(x).to(gmms.device)
    
    # apply projection and clipping
    x = project_eigenvectors(clip_eigenvalues(x)).to(x0.device)
    
    return x


@torch.no_grad()
def edit_zb(model, zbs, masks, start_t, method="sdedit", verbose=True):
    """
    Performs shape completion given incomplete sets of Zbs.

    Args:
        model:
        part_repr: (B, N, D). A batch of part-wise representations to be modified
        masks: (B, N, D). A batch of binary masks indicating the parts to be modified.
        start_t: The timestep where denoising process starts.
        method: 
        verbose:
    """
    assert start_t >= 0 and start_t <= model.var_sched.num_steps
    assert method in ("mcg", "repaint", "sdedit")
    assert zbs.ndim == 3, f"Got 'gmms' of dimension {zbs.ndim}"
    assert zbs.shape == masks.shape, f"{zbs.shape} != {masks.shape}"

    timesteps = list(range(start_t, 0, -1))
    if verbose:
        # timesteps = tqdm(timesteps)
        # timesteps = track(timesteps)
        pass

    x0 = zbs.clone()

    x = add_noise(model, x0, start_t)

    for t in timesteps:
        if method == "mcg":
            x = x.requires_grad_()  # record gradient
            x_ = denoise_one_step(model, x, t)
        else:
            with torch.no_grad():
                x_ = denoise_one_step(model, x, t)

        if method == "mcg":
            raise NotImplementedError()
        elif method == "repaint":
            raise NotImplementedError()
        elif method == "sdedit":
            x = add_noise(model, x0, t - 1)
            x[masks == 1.0] = x_[masks == 1.0]
        else:
            raise NotImplementedError()
    assert x.shape == zbs.shape, f"'x' shape: {x.shape} / 'gmms' shape: {zbs.shape}"
    assert torch.all(x[masks == 0.0] == x0[masks == 0.0]), (
        "The algorithm must not edit the unmasked parts"
    )
    
    return x


@torch.no_grad()
def edit_cond(model, zhs, gmms, masks, start_t, method="sdedit", verbose=True):
    assert start_t >= 0 and start_t <= model.var_sched.num_steps
    assert method in ("mcg", "repaint", "sdedit")
    assert zhs.ndim == 3, f"Got 'zhs' of shape {zhs.shape}"
    assert gmms.ndim == 3, f"Got 'gmms' of shape {gmms.shape}"

    timesteps = list(range(start_t, 0, -1))
    if verbose:
        # timesteps = tqdm(timesteps)
        # timesteps = track(timesteps)
        pass

    x0 = zhs

    x = add_noise(model, x0, start_t)

    for t in timesteps:
        if method == "mcg":
            x = x.requires_grad_()  # record gradient
            x_ = denoise_one_step_cond(model, x, gmms, t)
        else:
            with torch.no_grad():
                x_ = denoise_one_step_cond(model, x, gmms, t)

        if method == "mcg":
            raise NotImplementedError()
        elif method == "repaint":
            raise NotImplementedError()
        elif method == "sdedit":
            x = add_noise(model, x0, t - 1)
            x[masks == 1.0] = x_[masks == 1.0]
        else:
            raise NotImplementedError()
    assert x.shape == zhs.shape, f"'x' shape: {x.shape} / 'zhs' shape: {zhs.shape}"
    assert torch.all(x[masks == 0.0] == x0[masks == 0.0]), (
        "The algorithm must not edit the unmasked parts"
    )

    return x


if __name__ == "__main__":
    main()