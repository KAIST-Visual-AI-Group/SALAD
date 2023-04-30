import sys
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import open3d as o3d
import torch
from rich.progress import track

from salad.utils.paths import SPAGHETTI_DIR
from salad.utils import nputil, thutil, sysutil, meshutil


# TODO rewrite SPAGHETTI's relative path dependecies.
# Too lazy to refactorize SPAGHETTI's relative paths..
def add_spaghetti_path(spaghetti_path=SPAGHETTI_DIR):
    spaghetti_path = str(spaghetti_path)
    if spaghetti_path not in sys.path:
        sys.path.append(spaghetti_path)


def delete_spaghetti_path(
    spaghetti_path=SPAGHETTI_DIR,
):
    spaghetti_path = str(spaghetti_path)
    if spaghetti_path in sys.path:
        sys.path.remove(spaghetti_path)


def load_spaghetti(device, tag="chairs_large"):
    assert tag in [
        "chairs_large",
        "airplanes",
        "tables",
    ], f"tag should be 'chairs_large', 'airplanes' or 'tables'."

    add_spaghetti_path()
    from salad.spaghetti.options import Options
    from salad.spaghetti.ui import occ_inference

    opt = Options()
    opt.dataset_size = 1
    opt.device = device
    opt.tag = tag
    infer_module = occ_inference.Inference(opt)
    spaghetti = infer_module.model.to(device)
    spaghetti.eval()
    for p in spaghetti.parameters():
        p.requires_grad_(False)
    delete_spaghetti_path()
    return spaghetti


def load_mesher(
    device,
    min_res=64,
):
    from salad.spaghetti.utils.mcubes_meshing import MarchingCubesMeshing

    mesher = MarchingCubesMeshing(device=device, min_res=min_res)
    delete_spaghetti_path()
    return mesher


def get_mesh_and_pc(spaghetti, mesher, zc):
    vert, face = get_mesh_from_spaghetti(spaghetti, mesher, zc)
    pc = poisson_sampling(vert, face)
    return vert, face, pc


def get_mesh_from_spaghetti(spaghetti, mesher, zc, res=256):
    mesh = mesher.occ_meshing(
        decoder=get_occ_func(spaghetti, zc), res=res, get_time=False, verbose=False
    )
    vert, face = list(map(lambda x: thutil.th2np(x), mesh))
    return vert, face


def poisson_sampling(vert: np.array, face: np.array):
    vert_o3d = o3d.utility.Vector3dVector(vert)
    face_o3d = o3d.utility.Vector3iVector(face)
    mesh_o3d = o3d.geometry.TriangleMesh(vert_o3d, face_o3d)
    pc_o3d = mesh_o3d.sample_points_poisson_disk(2048)
    pc = np.asarray(pc_o3d.points).astype(np.float32)
    return pc


def get_occ_func(spaghetti, zc):
    device = spaghetti.device
    zc = nputil.np2th(zc).to(device)

    def forward(x):
        nonlocal zc
        x = x.unsqueeze(0)
        out = spaghetti.occupancy_network(x, zc)[0, :]
        out = 2 * out.sigmoid_() - 1
        return out

    if zc.dim() == 2:
        zc = zc.unsqueeze(0)
    return forward


def generate_zc_from_sj_gaus(
    spaghetti,
    sj: Union[torch.Tensor, np.ndarray],
    gaus: Union[torch.Tensor, np.ndarray],
):
    """
    Input:
        sj: [B,16,512] or [16,512]
        gaus: [B,16,16] or [16,16]
    Output:
        zc: [B,16,512]
    """
    device = spaghetti.device
    sj = nputil.np2th(sj)
    gaus = nputil.np2th(gaus)
    assert sj.dim() == gaus.dim()

    if sj.dim() == 2:
        sj = sj.unsqueeze(0)
    batch_sj = sj.to(device)
    batch_gmms = batch_gaus_to_gmms(gaus, device)
    zcs, _ = spaghetti.merge_zh(batch_sj, batch_gmms)
    return zcs


def generate_zc_from_za(spaghetti, za: Union[torch.Tensor, np.ndarray]):
    device = spaghetti.device
    za = nputil.np2th(za).to(device)
    sjs, gmms = spaghetti.decomposition_control(za)
    zcs, _ = spaghetti.merge_zh(sjs, gmms)
    return zcs


def generate_gaus_from_za(spaghetti, za):
    # device = spaghetti.device
    # za = nputil.np2th(za).to(device)
    sjs, gmms = spaghetti.decomposition_control(za)
    if isinstance(gmms[0], list):
        gaus = gmms[0]
    else:
        gaus = list(gmms)
    gaus = [flatten_gmms_item(x) for x in gaus]
    gaus = torch.cat(gaus, -1)

    # gaus = batch_gmms_to_gaus(gmms)
    return gaus


def generate_zc_from_single_phase_latent(
    spaghetti, sj_gaus: Union[torch.Tensor, np.ndarray]
):
    device = spaghetti.device
    sj_gaus = nputil.np2th(sj_gaus).to(device)
    sj, gaus = sj_gaus.split(split_size=[512, 16], dim=-1)
    zcs = generate_zc_from_sj_gaus(spaghetti, sj, gaus)
    return zcs


def flatten_gmms_item(x):
    """
    Input: [B,1,G,*shapes]
    Output: [B,G,-1]
    """
    return x.reshape(x.shape[0], x.shape[2], -1)


@torch.no_grad()
def batch_gmms_to_gaus(gmms):
    """
    Input:
        [T(B,1,G,3), T(B,1,G,3,3), T(B,1,G), T(B,1,G,3)]
    Output:
        T(B,G,16)
    """
    if isinstance(gmms[0], list):
        gaus = gmms[0].copy()
    else:
        gaus = list(gmms).copy()

    gaus = [flatten_gmms_item(x) for x in gaus]
    return torch.cat(gaus, -1)


@torch.no_grad()
def batch_gaus_to_gmms(gaus, device="cpu"):
    """
    Input: T(B,G,16)
    Output: [mu: T(B,1,G,3), eivec: T(B,1,G,3,3), pi: T(B,1,G), eival: T(B,1,G,3)]
    """
    gaus = nputil.np2th(gaus).to(device)
    if len(gaus.shape) < 3:
        gaus = gaus.unsqueeze(0)  # expand dim for batch

    B, G, _ = gaus.shape
    mu = gaus[:, :, :3].reshape(B, 1, G, 3)
    eivec = gaus[:, :, 3:12].reshape(B, 1, G, 3, 3)
    pi = gaus[:, :, 12].reshape(B, 1, G)
    eival = gaus[:, :, 13:16].reshape(B, 1, G, 3)

    return [mu, eivec, pi, eival]


def reflect_and_concat_gmms(gmms: torch.Tensor):
    """
    Input:
        gmms: (B, 8, 16). A batch of GMMs
    Output:
        new_gmms: (B, 16, 16)
    """
    gmms = nputil.np2th(gmms)
    gmms = gmms.clone()
    if gmms.dim() == 2:
        gmms = gmms.unsqueeze(0)

    affine = torch.eye(3).to(gmms)
    affine[0, 0] = -1.0

    mu, p, phi, eigen = torch.split(gmms, [3, 9, 1, 3], dim=2)
    if affine.ndim == 2:
        affine = affine.unsqueeze(0).expand(mu.size(0), *affine.shape)

    bs, n_part, _ = mu.shape
    p = p.reshape(bs, n_part, 3, 3)

    mu_r = torch.einsum("bad, bnd -> bna", affine, mu)
    p_r = torch.einsum("bad, bncd -> bnca", affine, p)
    p_r = p_r.reshape(bs, n_part, -1)
    gmms_t = torch.cat([mu_r, p_r, phi, eigen], dim=2)
    assert (
        gmms.shape == gmms_t.shape
    ), "Input and reflected gmms shapes must be the same"

    return torch.cat([gmms, gmms_t], dim=1)


def clip_eigenvalues(gaus: Union[torch.Tensor, np.ndarray], eps=1e-4):
    """
    Input:
        gaus: [B,G,16] or [G,16]
    Output:
        gaus_clipped: [B,G,16] or [G,16] torch.Tensor
    """
    gaus = nputil.np2th(gaus)
    clipped_gaus = gaus.clone()
    clipped_gaus[..., 13:16] = torch.clamp_min(clipped_gaus[..., 13:16], eps)
    return clipped_gaus


def project_eigenvectors(gaus: Union[torch.Tensor, np.ndarray]):
    """
    Input:
        gaus: [B,G,16] or [G,16]
    Output:
        gaus_projected: [B,G,16] or [1,G,16]
    """
    gaus = nputil.np2th(gaus).clone()
    if gaus.ndim == 2:
        gaus = gaus.unsqueeze(0)

    B, G = gaus.shape[:2]
    eigvec = gaus[:, :, 3:12]
    eigvec_projected = get_orthonormal_bases_svd(eigvec)
    gaus[:, :, 3:12] = eigvec_projected
    return gaus


def get_orthonormal_bases_svd(vs: torch.Tensor):
    """
    Implements the solution for the Orthogonal Procrustes problem,
    which projects a matrix to the closest rotation matrix / reflection matrix using SVD.
    Args:
        vs: Tensor of shape (B, M, 9)
    Returns:
        p: Tensor of shape (B, M, 9).
    """
    # Compute SVDs of matrices in batch
    b, m, _ = vs.shape
    vs_ = vs.reshape(b * m, 3, 3)
    U, _, Vh = torch.linalg.svd(vs_)
    # Determine the diagonal matrix to make determinants 1
    sigma = torch.eye(3)[None, ...].repeat(b * m, 1, 1).to(vs_.device)
    det = torch.linalg.det(torch.bmm(U, Vh))  # Compute determinants of UVT
    ####
    # Do not set the sign of determinants to 1.
    # Inputs contain reflection matrices.
    # sigma[:, 2, 2] = det
    ####
    # Construct orthogonal matrices
    p = torch.bmm(torch.bmm(U, sigma), Vh)
    return p.reshape(b, m, 9)


def save_meshes_and_pointclouds(
    spaghetti,
    mesher,
    zcs,
    save_top_dir,
    mesh_save_dir=None,
    pc_save_dir=None,
    num_shapes=2000,
):
    save_top_dir = Path(save_top_dir)
    print(f"Save dir is: {save_top_dir}")
    if mesh_save_dir is None:
        mesh_save_dir = save_top_dir / "meshes"
        mesh_save_dir.mkdir(exist_ok=True)
    if pc_save_dir is None:
        pc_save_dir = save_top_dir / "pointclouds"
        pc_save_dir.mkdir(exist_ok=True)

    mesh_save_dir = Path(mesh_save_dir)
    pc_save_dir = Path(pc_save_dir)

    all_pointclouds = np.zeros((num_shapes, 2048, 3))
    for i in track(range(num_shapes), description="extracting pc and mesh"):
        zc = zcs[i]
        vert_np, face_np, pc_np = get_mesh_and_pc(spaghetti, mesher, zc)
        sysutil.clean_gpu()
        all_pointclouds[i] = pc_np
        meshutil.write_obj_triangle(mesh_save_dir / f"{i}.obj", vert_np, face_np)
        np.save(pc_save_dir / f"{i}.npy", pc_np)

        if i == 1000:
            with h5py.File(save_top_dir / "o3d_all_pointclouds.hdf5", "w") as f:
                f["data"] = all_pointclouds[:1000]

    with h5py.File(save_top_dir / "o3d_all_pointclouds.hdf5", "w") as f:
        f["data"] = all_pointclouds
