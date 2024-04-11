
import random
from PIL import Image

import jutils
import torch

####
# NOTE: 20240411
# Seungwoo: Need to redirect paths
from demo.utils.gmm_edit import edit_gmm, edit_zh
from pvd.utils.spaghetti_utils import clip_eigenvalues, project_eigenvectors
####


def exchange_parts(
    gmm_a: torch.Tensor,
    zh_a: torch.Tensor,
    pl_a: torch.Tensor,
    gmm_b: torch.Tensor,
    zh_b: torch.Tensor,
    pl_b: torch.Tensor,
    exchange_idx: int,
    debug=False,
):
    assert gmm_a.ndim == 2 and gmm_b.ndim == 2, (
        f"Got GMMs of dimensions {gmm_a.ndim} and {gmm_b.ndim}"
    )
    assert gmm_a.shape[1] == gmm_b.shape[1], (
        f"Expected input GMMs to have same number of parts, got {gmm_a.shape[1]} and {gmm_b.shape[1]}"
    )
    n_total_part = gmm_a.shape[1]
    total_part_indices = set(range(n_total_part))
    
    # find the semantic parts that exist in both shapes
    part_ids_a = set(pl_a.unique().tolist())
    part_ids_b = set(pl_b.unique().tolist())
    # id_intersec = part_ids_a.intersection(part_ids_b)
    # assert len(id_intersec) != 0, "Two shapes have no part in common"
    assert exchange_idx in part_ids_a and exchange_idx in part_ids_b

    selection = exchange_idx
    
    gmm_a_to_b, zh_a_to_b, a2b_splits = donate_gmm_and_zh(
        gmm_a, zh_a, pl_a, gmm_b, zh_b, pl_b, selection,
    )
    
    gmm_b_to_a, zh_b_to_a, b2a_splits = donate_gmm_and_zh(
        gmm_b, zh_b, pl_b, gmm_a, zh_a, pl_a, selection,
    )
    
    # render Gaussians
    if debug:
        gmm_a_img = jutils.visutil.render_gaussians(gmm_a)
        sel_a_indices = torch.where(pl_a == selection)[0].tolist()    
        sel_a_img = jutils.visutil.render_gaussians(gmm_a[sel_a_indices])
    
        gmm_b_img = jutils.visutil.render_gaussians(gmm_b)
        sel_b_indices = torch.where(pl_b == selection)[0].tolist()
        sel_b_img = jutils.visutil.render_gaussians(gmm_b[sel_b_indices])
    
        gmm_a_to_b_img = jutils.visutil.render_gaussians(gmm_a_to_b)
        gmm_b_to_a_img = jutils.visutil.render_gaussians(gmm_b_to_a)
    
        gmm_imgs = jutils.imageutil.merge_images(
            [gmm_a_img, sel_a_img, gmm_a_to_b_img, gmm_b_img, sel_b_img, gmm_b_to_a_img]
        )
        assert isinstance(gmm_imgs, Image.Image)
     
    ret_vals = [gmm_a_to_b, zh_a_to_b, a2b_splits, gmm_b_to_a, zh_b_to_a, b2a_splits]
    if debug:
        ret_vals.append(gmm_imgs)
    return tuple(ret_vals)


def donate_gmm_and_zh(gmm_donor, zh_donor, pl_donor, gmm_donee, zh_donee, pl_donee, chosen_part_id):
    """
    Donate the part specified by 'chosen_part_id' from the donor shape to the corresponding part
    in the donee shape.
    """
    # check: the number of parts
    assert gmm_donor.shape[1] == gmm_donee.shape[1], (
        f"Expected two shapes to have the same number of parts, got {gmm_donor.shape[1]} and {gmm_donee.shape[1]}"
    )
    n_part = gmm_donor.shape[1]
    n_part_half = int(n_part // 2)
    
    # check: symmetry
    # two GMMs are assumed to be symmetric (i.e., its first 8 parts have the same label as its last 8 parts)
    assert torch.all(pl_donor[:n_part_half] == pl_donor[n_part_half:]), "Donor GMM is not symmetric"
    assert torch.all(pl_donee[:n_part_half] == pl_donee[n_part_half:]), "Donee GMM is not symmetric"
    
    # check: existence of selected semantic part in both shapes
    part_in_donor = pl_donor.unique().tolist()
    part_in_donee = pl_donee.unique().tolist()
    assert int(chosen_part_id) in part_in_donor, (
        f"Selected part must exist in donor, donor has part {part_in_donor} missing {chosen_part_id}"
    )
    assert chosen_part_id in part_in_donee, (
        f"Selected part must exist in donee, donee has part {part_in_donee} missing {chosen_part_id}"
    )
    
    half_part_indices = set(range(n_part_half))
        
    donor_half_chosen_indices = torch.where(pl_donor[:n_part_half] == chosen_part_id)[0]
    donor_other_chosen_indices = donor_half_chosen_indices + 8
                
    donee_half_chosen_indices = torch.where(pl_donee[:n_part_half] == chosen_part_id)[0]
    donee_half_kept_indices = torch.tensor(
        list(half_part_indices - set(donee_half_chosen_indices.tolist()))
    )
    donee_other_kept_indices = donee_half_kept_indices + 8
        
    # the first half of the mixed shape
    gmm_half_from_donor = gmm_donor[donor_half_chosen_indices]
    gmm_half_from_donee = gmm_donee[donee_half_kept_indices]
    zh_half_from_donor = zh_donor[donor_half_chosen_indices]
    zh_half_from_donee = zh_donee[donee_half_kept_indices]
        
    # the second half of the mixed shape
    gmm_other_from_donor = gmm_donor[donor_other_chosen_indices]
    gmm_other_from_donee = gmm_donee[donee_other_kept_indices]
    zh_other_from_donor = zh_donor[donor_other_chosen_indices]
    zh_other_from_donee = zh_donee[donee_other_kept_indices]
        
    assert gmm_half_from_donor.shape[0] == gmm_other_from_donor.shape[0]
    assert gmm_half_from_donee.shape[0] == gmm_other_from_donee.shape[0]
    assert zh_half_from_donor.shape[0] == zh_other_from_donor.shape[0]
    assert zh_half_from_donee.shape[0] == zh_other_from_donee.shape[0]
        
    gmm_combined = torch.cat(
        [
            gmm_half_from_donor,  # need refinement
            gmm_half_from_donee,
            gmm_other_from_donor,  # refined automatically
            gmm_other_from_donee,
        ],
        dim=0,
    )
    zh_combined = torch.cat(
        [
            zh_half_from_donor,  # need refinement
            zh_half_from_donee,
            zh_other_from_donor,  # need to be refined together
            zh_other_from_donee,
        ],
        dim=0,
    )
    splits = torch.tensor(
        [
            0,
            int(gmm_half_from_donor.shape[0]),
            int(gmm_half_from_donee.shape[0]),
            int(gmm_other_from_donor.shape[0]),
            int(gmm_other_from_donee.shape[0]),
        ],
    )
    splits = torch.cumsum(splits, 0)
    
    return gmm_combined, zh_combined, splits


def refine_cascaded(
    gmm,
    zh,
    p1_model,
    p2_model,
    gmm_indices,
    zh_indices,
    timesteps: int,
    use_half_gmm: bool
):
    assert gmm.ndim == 2
    assert zh.ndim == 2
    
    gmm_n_part = gmm.shape[0]
    gmm_device = gmm.device
    
    gmm_refined = edit_gmm(
        p1_model, gmm[None], gmm_indices, timesteps, recon_sym_parts=use_half_gmm,
    )[0].to(gmm_device)
    assert gmm_refined.shape[0] == zh.shape[0], (
        f"Expected same number of parts, got {gmm_refined.shape[0]} {zh.shape[0]}"
    )

    # project and clip
    gmm_refined = project_eigenvectors(clip_eigenvalues(gmm_refined)).to(p1_model.device)
    if gmm_refined.ndim == 3:
        gmm_refined = gmm_refined[0]
    
    zh_refined = edit_zh(
        p2_model, zh[None], gmm_refined[None], zh_indices, timesteps,
    )[0].to(gmm_device)

    assert gmm_refined.ndim == gmm.ndim
    assert zh_refined.ndim == zh.ndim
    
    return gmm_refined, zh_refined


def refine_single_phase(
    zh,
    gmm,
    model,
    split,
    timesteps: int,
):
    assert zh.ndim == 2, f"Got GMM of dimension {zh.ndim}"
    assert gmm.ndim == 2, f"Got GMM of dimension {gmm.ndim}"

    donor_half_begin = split[0]
    donor_half_end = split[1]
    donor_other_begin = split[2]
    donor_other_end = split[3]
    indices_to_edit = list(range(donor_half_begin, donor_half_end)) + list(range(donor_other_begin, donor_other_end))

    # NOTE: The single phase model was trained on 
    # the concatenated latents of form: [Zh | GMM]
    zh_and_gmm = torch.cat([zh, gmm], -1)
    
    zh_and_gmm_refined = edit_gmm(
        model, zh_and_gmm[None], indices_to_edit, timesteps, recon_sym_parts=False
    )[0]

    zh_refined, gmm_refined = zh_and_gmm_refined.split([zh.size(1), gmm.size(1)], dim=1)

    ####
    gmm_refined = project_eigenvectors(clip_eigenvalues(gmm_refined)).to(model.device)[0]
    ####

    assert gmm_refined.shape == gmm.shape, f"{gmm_refined.shape} != {gmm.shape}"
    assert zh_refined.shape == zh.shape, f"{zh_refined.shape} != {zh.shape}"

    return zh_refined, gmm_refined