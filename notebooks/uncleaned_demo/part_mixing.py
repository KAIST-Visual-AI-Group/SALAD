"""
part_mixing.py

A script for demonstrating part-mixing capability of the proposed method.
"""

import argparse
from PIL import Image
from pathlib import Path
import random
from time import time
from typing import Tuple

import jutils
import numpy as np

from pvd.utils.spaghetti_utils import reflect_and_concat_gmms

from demo.utils.io_utils import (
    load_dataset,
    load_phase1_and_phase2,
    load_spaghetti_and_mesher,
)

import demo.utils.paths as demo_paths
from demo.utils.mixing import exchange_parts, refine_cascaded, refine_single_phase
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
        help="Type of the model used. Can be either ('single_phase', 'cascaded')",
    )
    parser.add_argument(
        "shape_category",
        type=str,
        help="Type of the shapes. Can be either ('chairs', 'airplanes', 'tables')",
    )
    parser.add_argument(
        "n_shape", type=int, help="Number of shape 'pairs' used for part mixing"
    )
    parser.add_argument(
        "--refine_t", type=int, default=10, help="Number of refinement SDEdit step"
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./demo_out/part_mixing",
        help="Root directory of log directories",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device used for the experiment"
    )
    parser.add_argument(
        "--debug", action="store_true", help="A flag for enabling debug mode"
    )

    args = parser.parse_args()

    # Validate args
    assert args.model_type in (
        "single_phase",
        "cascaded",
    ), f"Unsupported model type: {str(args.model_type)}"
    assert args.shape_category in (
        "airplanes",
        "chairs",
        "tables",
    ), f"Unsupported model type: {str(args.shape_category)}"
    assert isinstance(args.n_shape, int) and args.n_shape > 0

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
        run_part_mixing_cascaded(args, log_dir)
    elif args.model_type == "single_phase":
        run_part_mixing_single_phase(args, log_dir)
    else:
        raise NotImplementedError(
            f"Unsupported model architecture: {str(args.model_type)}"
        )


def run_part_mixing_cascaded(args: argparse.Namespace, log_dir: Path):

    # Load data
    gmms, zhs, part_labels, shape_ids = load_data_for_demo(
        args.shape_category, args.device
    )

    # Split data: set A and set B
    set_a, set_b = create_shape_pairs(args.n_shape, gmms, zhs, part_labels, shape_ids)
    a_gmms, a_zhs, a_part_labels, a_shape_ids = set_a
    b_gmms, b_zhs, b_part_labels, b_shape_ids = set_b
    with open(log_dir / "pair_ids.txt", "w") as f:
        f.writelines(
            [",".join(id_pair) + "\n" for id_pair in zip(a_shape_ids, b_shape_ids)]
        )

    # Iterate over all shapes, and for every semantic part, swap it
    start_t = time()
    (
        (a2b_gmms, a2b_zhs, a2b_splits),
        (b2a_gmms, b2a_zhs, b2a_splits),
        pair_split, swapped_part_ids,
    ) = loop_swap_parts(set_a, set_b)
    elapsed = time() - start_t
    print(f"[Part Swap] Took {elapsed:3f} seconds")
    torch.save(
        {
            "a_gmms": a_gmms.detach().cpu(),
            "a_zhs": a_zhs.detach().cpu(),
            "a_shape_ids": a_shape_ids,

            "b_gmms": b_gmms.detach().cpu(),
            "b_zhs": b_zhs.detach().cpu(),
            "b_shape_ids": b_shape_ids,

            "a2b_gmms": detach_cpu_list(a2b_gmms),
            "a2b_zhs": detach_cpu_list(a2b_zhs),
            "a2b_splits": detach_cpu_list(a2b_splits),

            "b2a_gmms": detach_cpu_list(b2a_gmms),
            "b2a_zhs": detach_cpu_list(b2a_zhs),
            "b2a_splits": detach_cpu_list(b2a_splits),

            "pair_split": detach_cpu_list(pair_split),
        },
        str(log_dir / "swap_results.pt"),
    )

    # Load model
    start_t = time()
    phase1, phase2, spaghetti, mesher = load_models_cascaded(
        args.shape_category, args.device
    )
    elapsed = time() - start_t
    print(f"[Model Loading] Took {elapsed:3f} seconds")

    # Refine using SDEdit
    start_t = time()
    (
        refined_a2b_gmms,
        refined_a2b_zhs,
        refined_b2a_gmms,
        refined_b2a_zhs,
    ) = loop_refine_parts_without_symmetry(
        (a2b_gmms, a2b_zhs, a2b_splits),
        (b2a_gmms, b2a_zhs, b2a_splits),
        phase1,
        phase2,
        args.refine_t,
    )

    elapsed = time() - start_t
    print(f"[Refine] Took {elapsed:3f} seconds")
    torch.save(
        {
            # Source shape A
            "src_a_gmms": a_gmms.detach().cpu(),
            "src_a_zhs": a_zhs.detach().cpu(),
            "src_a_part_labels": a_part_labels.detach().cpu().long(),
            "src_a_shape_ids": a_shape_ids,
            # Source shape B
            "src_b_gmms": b_gmms.detach().cpu(),
            "src_b_zhs": b_zhs.detach().cpu(),
            "src_b_part_labels": b_part_labels.detach().cpu().long(),
            "src_b_shape_ids": b_shape_ids,
            # A -> B, after swapping
            "swap_a2b_gmms": detach_cpu_list(a2b_gmms),
            "swap_a2b_zhs": detach_cpu_list(a2b_zhs),
            "swap_a2b_splits": detach_cpu_list(a2b_splits),
            # A <- B, after swapping
            "swap_b2a_gmms": detach_cpu_list(b2a_gmms),
            "swap_b2a_zhs": detach_cpu_list(b2a_zhs),
            "swap_b2a_splits": detach_cpu_list(b2a_splits),
            # The split that distinguishes different shape pairs
            "pair_split": detach_cpu_list(pair_split),
            # The part IDs of swapped part for each swap example
            "swapped_part_ids": swapped_part_ids.detach().cpu(),
            # A -> B, after refinement
            "refined_a2b_gmms": detach_cpu_list(refined_a2b_gmms),
            "refined_a2b_zhs": detach_cpu_list(refined_a2b_zhs),
            # B <- A, after refinement
            "refined_b2a_gmms": detach_cpu_list(refined_b2a_gmms),
            "refined_b2a_zhs": detach_cpu_list(refined_b2a_zhs),
        },
        # str(log_dir / "refine_results.pt"),
        str(log_dir / "part_mixing_results.pt")
    )

    if args.debug:
        start_t = time()
        loop_visualize_results(
            10, 
            (a_gmms, a_zhs), 
            (b_gmms, b_zhs),
            (a2b_gmms, a2b_zhs),
            (b2a_gmms, b2a_zhs),
            (refined_a2b_gmms, refined_a2b_zhs),
            (refined_b2a_gmms, refined_b2a_zhs),
            pair_split,
            spaghetti,
            mesher,
            log_dir,
        )
        elapsed = time() - start_t
        print(f"[Visualization] Took {elapsed:3f} seconds")
        print("[*] RUNNING IN DEBUG MODE. SKIP SAVING MESHES")
        return
    
    # Save swapped meshes
    start_t = time()
    swapped_mesh_dir = log_dir / "swapped_mesh"
    loop_decode_and_save(
        (a2b_gmms, a2b_zhs),
        (b2a_gmms, b2a_zhs),
        swapped_mesh_dir,
        spaghetti,
        mesher,
    )
    elapsed = time() - start_t
    print(f"[Swapped / Decoding] Took {elapsed:3f} seconds")

    # Save refined meshes
    start_t = time()
    refined_mesh_dir = log_dir / "refined_mesh"
    loop_decode_and_save(
        (refined_a2b_gmms, refined_a2b_zhs),
        (refined_b2a_gmms, refined_b2a_zhs),
        refined_mesh_dir,
        spaghetti,
        mesher,
    )
    elapsed = time() - start_t
    print(f"[Refined / Decoding] Took {elapsed:3f} seconds")


def run_part_mixing_single_phase(args: argparse.Namespace, log_dir: Path):
    
    # Load data
    gmms, zhs, part_labels, shape_ids = load_data_for_demo(
        args.shape_category, args.device
    )

    # Split data: set A and set B
    set_a, set_b = create_shape_pairs(args.n_shape, gmms, zhs, part_labels, shape_ids)
    a_gmms, a_zhs, a_part_labels, a_shape_ids = set_a
    b_gmms, b_zhs, b_part_labels, b_shape_ids = set_b
    with open(log_dir / "pair_ids.txt", "w") as f:
        f.writelines(
            [",".join(id_pair) + "\n" for id_pair in zip(a_shape_ids, b_shape_ids)]
        )

    # Iterate over all shapes, and for every semantic part, swap it
    start_t = time()
    (
        (a2b_gmms, a2b_zhs, a2b_splits),
        (b2a_gmms, b2a_zhs, b2a_splits),
        pair_split, swapped_part_ids,
    ) = loop_swap_parts(set_a, set_b)
    elapsed = time() - start_t
    print(f"[Part Swap] Took {elapsed:3f} seconds")
    torch.save(
        {
            "a2b_gmms": detach_cpu_list(a2b_gmms),
            "a2b_zhs": detach_cpu_list(a2b_zhs),
            "a2b_splits": detach_cpu_list(a2b_splits),
            "b2a_gmms": detach_cpu_list(b2a_gmms),
            "b2a_zhs": detach_cpu_list(b2a_zhs),
            "b2a_splits": detach_cpu_list(b2a_splits),
            "pair_split": detach_cpu_list(pair_split),
        },
        str(log_dir / "swap_results.pt"),
    )

    # Load model
    start_t = time()
    model, spaghetti, mesher = load_models_single_phase(
        args.shape_category, args.device
    )
    elapsed = time() - start_t
    print(f"[Model Loading] Took {elapsed:3f} seconds")

    # Refine using SDEdit
    start_t = time()
    (
        refined_a2b_gmms,
        refined_a2b_zhs,
        refined_b2a_gmms,
        refined_b2a_zhs,
    ) =  loop_refine_parts(
        (a2b_gmms, a2b_zhs, a2b_splits),
        (b2a_gmms, b2a_zhs, b2a_splits),
        model,
        args.refine_t,
    )
    elapsed = time() - start_t
    print(f"[Refine] Took {elapsed:3f} seconds")
    torch.save(
        {
            # Source shape A
            "src_a_gmms": a_gmms.detach().cpu(),
            "src_a_zhs": a_zhs.detach().cpu(),
            "src_a_part_labels": a_part_labels.detach().cpu().long(),
            "src_a_shape_ids": a_shape_ids,
            # Source shape B
            "src_b_gmms": b_gmms.detach().cpu(),
            "src_b_zhs": b_zhs.detach().cpu(),
            "src_b_part_labels": b_part_labels.detach().cpu().long(),
            "src_b_shape_ids": b_shape_ids,
            # A -> B, after swapping
            "swap_a2b_gmms": detach_cpu_list(a2b_gmms),
            "swap_a2b_zhs": detach_cpu_list(a2b_zhs),
            "swap_a2b_splits": detach_cpu_list(a2b_splits),
            # A <- B, after swapping
            "swap_b2a_gmms": detach_cpu_list(b2a_gmms),
            "swap_b2a_zhs": detach_cpu_list(b2a_zhs),
            "swap_b2a_splits": detach_cpu_list(b2a_splits),
            # The split that distinguishes different shape pairs
            "pair_split": detach_cpu_list(pair_split),
            # The part IDs of swapped part for each swap example
            "swapped_part_ids": swapped_part_ids.detach().cpu(),
            # A -> B, after refinement
            "refined_a2b_gmms": detach_cpu_list(refined_a2b_gmms),
            "refined_a2b_zhs": detach_cpu_list(refined_a2b_zhs),
            # B <- A, after refinement
            "refined_b2a_gmms": detach_cpu_list(refined_b2a_gmms),
            "refined_b2a_zhs": detach_cpu_list(refined_b2a_zhs),
        },
        # str(log_dir / "refine_results.pt"),
        str(log_dir / "part_mixing_results.pt")
    )

    if args.debug:
        start_t = time()
        loop_visualize_results(
            10, 
            (a_gmms, a_zhs), 
            (b_gmms, b_zhs),
            (a2b_gmms, a2b_zhs),
            (b2a_gmms, b2a_zhs),
            (refined_a2b_gmms, refined_a2b_zhs),
            (refined_b2a_gmms, refined_b2a_zhs),
            pair_split,
            spaghetti,
            mesher,
            log_dir,
        )
        elapsed = time() - start_t
        print(f"[Visualization] Took {elapsed:3f} seconds")
        print("[*] RUNNING IN DEBUG MODE. SKIP SAVING MESHES")
        return

    ####
    # Skip meshing
    # Save refined meshes
    # start_t = time()
    # refined_mesh_dir = log_dir / "refined_mesh"
    # loop_decode_and_save(
    #     (refined_a2b_gmms, refined_a2b_zhs),
    #     (refined_b2a_gmms, refined_b2a_zhs),
    #     refined_mesh_dir,
    #     spaghetti,
    #     mesher,
    # )
    # elapsed = time() - start_t
    # print(f"[Refined / Decoding] Took {elapsed:3f} seconds")
    ####


def loop_swap_parts(set_a: Tuple[torch.Tensor], set_b: Tuple[torch.Tensor]):
    a_gmms, a_zhs, a_part_labels, _ = set_a
    b_gmms, b_zhs, b_part_labels, _ = set_b

    a2b_gmms = []
    a2b_zhs = []
    a2b_splits = []

    b2a_gmms = []
    b2a_zhs = []
    b2a_splits = []

    n_swap_per_pair = [0]  # Number of part pairs that are swapped
    swapped_part_ids = []
    
    for pair_idx, (a_gmm, b_gmm, a_zh, b_zh, a_pl, b_pl) in enumerate(
        zip(a_gmms, b_gmms, a_zhs, b_zhs, a_part_labels, b_part_labels)
    ):
        parts_in_common = find_parts_in_common(a_pl, b_pl)
        swapped_part_ids = swapped_part_ids + parts_in_common
        
        for part in parts_in_common:
            a2b_gmm, a2b_zh, a2b_split, b2a_gmm, b2a_zh, b2a_split = exchange_parts(
                a_gmm, a_zh, a_pl, b_gmm, b_zh, b_pl, part
            )

            # collect (part A) -> (B)
            a2b_gmms.append(a2b_gmm)
            a2b_zhs.append(a2b_zh)
            a2b_splits.append(a2b_split)

            # collect (part B) -> (A)
            b2a_gmms.append(b2a_gmm)
            b2a_zhs.append(b2a_zh)
            b2a_splits.append(b2a_split)
        n_swap_per_pair.append(len(parts_in_common))

    n_swap_per_pair = torch.tensor(n_swap_per_pair)
    swapped_part_ids = torch.tensor(swapped_part_ids)
    pair_split = torch.cumsum(n_swap_per_pair, dim=0)

    return (a2b_gmms, a2b_zhs, a2b_splits), (b2a_gmms, b2a_zhs, b2a_splits), pair_split, swapped_part_ids


# 2023. 3. 4 This is the latest
@torch.no_grad()
def loop_refine_parts_without_symmetry(
    set_a2b: Tuple[torch.Tensor],
    set_b2a: Tuple[torch.Tensor],
    phase1,
    phase2,
    refine_t: int,
):
    # collection of "refined" A -> B latents
    a2b_gmm_rs = []
    a2b_zh_rs = []

    # collection of "refined" B -> A latents
    b2a_gmm_rs = []
    b2a_zh_rs = []

    for idx, (a2b_gmm, a2b_zh, a2b_split, b2a_gmm, b2a_zh, b2a_split) in enumerate(
        zip(*set_a2b, *set_b2a)
    ):
        a2b_gmm_r, a2b_zh_r = refine_cascaded(
            a2b_gmm,
            a2b_zh,
            phase1,
            phase2,
            list(range(a2b_split[0], a2b_split[1])) + list(range(a2b_split[2], a2b_split[3])),
            list(range(a2b_split[0], a2b_split[1])) + list(range(a2b_split[2], a2b_split[3])),
            refine_t,
            use_half_gmm=False,
        )
        a2b_gmm_rs.append(a2b_gmm_r)
        a2b_zh_rs.append(a2b_zh_r)

        b2a_gmm_r, b2a_zh_r = refine_cascaded(
            b2a_gmm,
            b2a_zh,
            phase1,
            phase2,
            list(range(b2a_split[0], b2a_split[1])) + list(range(b2a_split[2], b2a_split[3])),
            list(range(b2a_split[0], b2a_split[1])) + list(range(b2a_split[2], b2a_split[3])),
            refine_t,
            use_half_gmm=False,
        )
        b2a_gmm_rs.append(b2a_gmm_r)
        b2a_zh_rs.append(b2a_zh_r)

    return a2b_gmm_rs, a2b_zh_rs, b2a_gmm_rs, b2a_zh_rs


@torch.no_grad()
def loop_refine_parts_with_symmetry(
    set_a2b: Tuple[torch.Tensor],
    set_b2a: Tuple[torch.Tensor],
    phase1,
    phase2,
    refine_t: int,
):
    raise NotImplementedError("No longer supported")

    # collection of "refined" A -> B latents
    a2b_gmm_rs = []
    a2b_zh_rs = []

    # collection of "refined" B -> A latents
    b2a_gmm_rs = []
    b2a_zh_rs = []

    for idx, (a2b_gmm, a2b_zh, a2b_split, b2a_gmm, b2a_zh, b2a_split) in enumerate(
        zip(*set_a2b, *set_b2a)
    ):
        # TODO: How about name each element in split for better readability?
        a2b_gmm_r, a2b_zh_r = refine_cascaded(
            a2b_gmm[a2b_split[0] : a2b_split[2]],
            a2b_zh,
            phase1,
            phase2,
            list(range(a2b_split[0], a2b_split[1])),
            list(range(a2b_split[0], a2b_split[1]))
            + list(range(a2b_split[2], a2b_split[3])),
            refine_t,
            use_half_gmm=True,
        )
        a2b_gmm_rs.append(a2b_gmm_r)
        a2b_zh_rs.append(a2b_zh_r)

        b2a_gmm_r, b2a_zh_r = refine_cascaded(
            b2a_gmm[b2a_split[0] : b2a_split[2]],
            b2a_zh,
            phase1,
            phase2,
            list(range(b2a_split[0], b2a_split[1])),
            list(range(b2a_split[0], b2a_split[1]))
            + list(range(b2a_split[2], b2a_split[3])),
            refine_t,
            use_half_gmm=True,
        )
        b2a_gmm_rs.append(b2a_gmm_r)
        b2a_zh_rs.append(b2a_zh_r)

    return a2b_gmm_rs, a2b_zh_rs, b2a_gmm_rs, b2a_zh_rs


@torch.no_grad()
def loop_refine_parts(
    set_a2b: Tuple[torch.Tensor],
    set_b2a: Tuple[torch.Tensor],
    model,
    refine_t: int,
):
    # collection of "refined" A -> B latents
    a2b_gmm_rs = []
    a2b_zh_rs = []

    # collection of "refined" B -> A latents
    b2a_gmm_rs = []
    b2a_zh_rs = []

    for idx, (a2b_gmm, a2b_zh, a2b_split, b2a_gmm, b2a_zh, b2a_split) in enumerate(
        zip(*set_a2b, *set_b2a)
    ):
        a2b_zh_r, a2b_gmm_r = refine_single_phase(
            a2b_zh,
            a2b_gmm,
            model,
            a2b_split,
            refine_t,
        )
        a2b_gmm_rs.append(a2b_gmm_r)
        a2b_zh_rs.append(a2b_zh_r)

        b2a_zh_r, b2a_gmm_r = refine_single_phase(
            b2a_zh,
            b2a_gmm,
            model,
            b2a_split,
            refine_t,
        )
        b2a_gmm_rs.append(b2a_gmm_r)
        b2a_zh_rs.append(b2a_zh_r)

    return a2b_gmm_rs, a2b_zh_rs, b2a_gmm_rs, b2a_zh_rs


@torch.no_grad()
def loop_decode_and_save(
    set_a2b: Tuple[torch.Tensor],
    set_b2a: Tuple[torch.Tensor],
    mesh_dir,
    spaghetti,
    mesher,
):
    a2b_dir = mesh_dir / "a2b"
    b2a_dir = mesh_dir / "b2a"
    a2b_dir.mkdir(parents=True, exist_ok=True)
    b2a_dir.mkdir(parents=True, exist_ok=True)

    for save_idx, (a2b_gmm, a2b_zh, b2a_gmm, b2a_zh) in enumerate(
        zip(*set_a2b, *set_b2a)
    ):
        a2b_vert, a2b_face = decode_gmm_and_intrinsic(
            spaghetti, mesher, a2b_gmm[None], a2b_zh[None]
        )
        b2a_vert, b2a_face = decode_gmm_and_intrinsic(
            spaghetti, mesher, b2a_gmm[None], b2a_zh[None]
        )

        jutils.meshutil.write_obj_triangle(
            str(a2b_dir / f"{save_idx}.obj"),
            a2b_vert,
            a2b_face,
        )

        jutils.meshutil.write_obj_triangle(
            str(b2a_dir / f"{save_idx}.obj"),
            b2a_vert,
            b2a_face,
        )


@torch.no_grad()
def loop_visualize_results(
    n_vis: int,
    gmms_zhs_a: Tuple[torch.Tensor],
    gmms_zhs_b: Tuple[torch.Tensor],
    gmms_zhs_a2b: Tuple[torch.Tensor],
    gmms_zhs_b2a: Tuple[torch.Tensor],
    refined_gmms_zhs_a2b: Tuple[torch.Tensor],
    refined_gmms_zhs_b2a: Tuple[torch.Tensor],
    pair_split: torch.Tensor,
    spaghetti,
    mesher,
    log_dir,
):
    # original shape
    a_gmms, a_zhs = gmms_zhs_a
    b_gmms, b_zhs = gmms_zhs_b

    # part-mixed shapes
    a2b_gmms, a2b_zhs = gmms_zhs_a2b
    b2a_gmms, b2a_zhs = gmms_zhs_b2a

    # after refinement
    refined_a2b_gmms, refined_a2b_zhs = refined_gmms_zhs_a2b
    refined_b2a_gmms, refined_b2a_zhs = refined_gmms_zhs_b2a

    n_vis = min(n_vis, len(a_gmms))
    print(f"Visualizing {n_vis} cases")

    vis_dir = log_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for vis_idx in range(n_vis):
        split_start, split_end = pair_split[vis_idx], pair_split[vis_idx + 1]
        common_args = spaghetti, mesher, camera_kwargs

        rows = []
        a_img = decode_and_render(a_gmms[vis_idx], a_zhs[vis_idx], *common_args)
        b_img = decode_and_render(b_gmms[vis_idx], b_zhs[vis_idx], *common_args)
        for var_idx in range(split_start, split_end):
            a2b_img = decode_and_render(
                a2b_gmms[var_idx], a2b_zhs[var_idx], *common_args
            )
            b2a_img = decode_and_render(
                b2a_gmms[var_idx], b2a_zhs[var_idx], *common_args
            )

            refined_a2b_img = decode_and_render(
                refined_a2b_gmms[var_idx],
                refined_a2b_zhs[var_idx],
                *common_args,
            )
            refined_b2a_img = decode_and_render(
                refined_b2a_gmms[var_idx],
                refined_b2a_zhs[var_idx],
                *common_args,
            )

            row_img = jutils.imageutil.merge_images(
                [
                    a_img,
                    a2b_img,
                    refined_a2b_img,
                    b2a_img,
                    refined_b2a_img,
                    b_img,
                ]
            )
            rows.append([row_img])
        img = jutils.imageutil.merge_images(rows)
        img.save(vis_dir / f"{vis_idx:03}.png")


def find_parts_in_common(a_part_label, b_part_label):
    a_semantic_parts = set(a_part_label.unique().tolist())
    b_semantic_parts = set(b_part_label.unique().tolist())
    intersection = list(a_semantic_parts.intersection(b_semantic_parts))
    return sorted(intersection)


def load_models_cascaded(shape_category: str, device):
    """
    Loads the models required for 'cascaded' setup.

    Loads the phase 1 and phase 2 models, as well as SPAGHETTI and mesher for shape decoding and rendering.
    """
    ####
    # phase1, phase2 = load_phase1_and_phase2(shape_category, use_sym=True, device=device)
    phase1, phase2 = load_phase1_and_phase2(shape_category, use_sym=False, device=device)
    ####

    if shape_category == "chairs":
        spaghetti, mesher = load_spaghetti_and_mesher(device, "chairs_large")
    elif shape_category == "airplanes":
        spaghetti, mesher = load_spaghetti_and_mesher(device, "airplanes")
    elif shape_category == "tables":
        spaghetti, mesher = load_spaghetti_and_mesher(device, "tables")
    else:
        raise NotImplementedError()

    return phase1, phase2, spaghetti, mesher


def load_models_single_phase(shape_category: str, device):
    model = load_single_phase(shape_category, device=device)

    if shape_category == "chairs":
        spaghetti, mesher = load_spaghetti_and_mesher(device, "chairs_large")
    else:
        spaghetti, mesher = load_spaghetti_and_mesher(device, "airplanes")

    return model, spaghetti, mesher


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
        raise NotImplementedError(f"Unsupported shape {shape_category}")

    # Make sure to load data to device
    gmms = gmms.to(device).float()
    zhs = zhs.to(device).float()
    part_labels = part_labels.to(device)

    assert gmms.size(1) == zhs.size(1), f"{gmms.size(1)} {zhs.size(1)}"
    assert gmms.size(1) == part_labels.size(1)

    return gmms, zhs, part_labels, shape_ids


def create_shape_pairs(n_shape_per_set: int, gmms, zhs, part_labels, shape_ids):

    # Set A
    a_gmms = gmms[:n_shape_per_set]
    a_zhs = zhs[:n_shape_per_set]
    a_part_labels = part_labels[:n_shape_per_set]
    a_shape_ids = shape_ids[:n_shape_per_set]

    # Set B
    b_gmms = gmms[n_shape_per_set : 2 * n_shape_per_set]
    b_zhs = zhs[n_shape_per_set : 2 * n_shape_per_set]
    b_part_labels = part_labels[n_shape_per_set : 2 * n_shape_per_set]
    b_shape_ids = shape_ids[n_shape_per_set : 2 * n_shape_per_set]

    return (a_gmms, a_zhs, a_part_labels, a_shape_ids), (
        b_gmms,
        b_zhs,
        b_part_labels,
        b_shape_ids,
    )


if __name__ == "__main__":
    main()
