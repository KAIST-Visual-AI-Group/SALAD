import sys
from pathlib import Path
import h5py
from rich.progress import track
from typing import Union, List
import numpy as np
import torch
import jutils
import open3d as o3d

def add_spaghetti_path_v2(spaghetti_path="/home/juil/projects/3D_CRISPR/spaghetti_v2"):
    if spaghetti_path not in sys.path:
        sys.path.append(spaghetti_path)

def delete_spaghetti_path_v2(spaghetti_path="/home/juil/projects/3D_CRISPR/spaghetti_v2"):
    if spaghetti_path in sys.path:
        sys.path.remove(spaghetti_path)

def load_spaghetti_v2(device, tag="chairs_0218"):
    add_spaghetti_path_v2()
    from options import Options
    from ui import occ_inference
    
    opt = Options()
    opt.dataset_size = 1
    opt.device = device
    opt.tag = tag
    infer_module = occ_inference.Inference(opt)
    spaghetti = infer_module.model.to(device)
    spaghetti.eval()
    for p in spaghetti.parameters(): p.requires_grad_(False)
    delete_spaghetti_path_v2()
    return spaghetti

def load_mesher_v2(device):
    add_spaghetti_path_v2()
    from utils.mcubes_meshing import MarchingCubesMeshing
    mesher = MarchingCubesMeshing(device=device)
    delete_spaghetti_path_v2()
    return mesher


def get_mesh_from_spaghetti_v2(spaghetti, mesher, zc, res=256):
    mesh = mesher.occ_meshing(
            decoder=get_occ_func_v2(spaghetti, zc),
            res=res,
            get_time=False,
            verbose=False
            )
    vert, face = list(map(lambda x : jutils.thutil.th2np(x), mesh))
    return vert, face



def get_occ_func_v2(spaghetti, zc):
    device = spaghetti.device
    zc = jutils.nputil.np2th(zc).to(device)
    def forward(x):
        nonlocal zc
        x = x.unsqueeze(0)
        # out = spaghetti.occupancy_network(x, zc)[0,:]
        out = spaghetti.occ_head(x, zc)[0,:]
        out = 2 * out.sigmoid_() - 1
        return out
    if zc.dim() == 2:
        zc = zc.unsqueeze(0)
    return forward

def get_mesh_and_pc_v2(spaghetti, mesher, zc):
    vert, face = get_mesh_from_spaghetti_v2(spaghetti, mesher, zc)
    pc = poisson_sampling(vert, face)
    return vert, face, pc

def save_meshes_and_pointclouds_v2(spaghetti, mesher, zcs, save_top_dir, mesh_save_dir=None, pc_save_dir=None, num_shapes=2000):
    save_top_dir = Path(save_top_dir)
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
        vert_np, face_np, pc_np = get_mesh_and_pc_v2(spaghetti, mesher, zc)
        jutils.sysutil.clean_gpu()
        all_pointclouds[i] = pc_np
        jutils.meshutil.write_obj_triangle(mesh_save_dir / f"{i}.obj", vert_np, face_np)
        np.save(pc_save_dir / f"{i}.npy", pc_np)

        if i == 1000:
            with h5py.File(save_top_dir / "o3d_all_pointclouds.hdf5", "w") as f:
                f["data"] = all_pointclouds[:1000]
        
    with h5py.File(save_top_dir / "o3d_all_pointclouds.hdf5", "w") as f:
        f["data"] = all_pointclouds


def poisson_sampling(vert: np.array, face: np.array):
    vert_o3d = o3d.utility.Vector3dVector(vert)
    face_o3d = o3d.utility.Vector3iVector(face)
    mesh_o3d = o3d.geometry.TriangleMesh(vert_o3d, face_o3d)
    pc_o3d = mesh_o3d.sample_points_poisson_disk(2048)
    pc = np.asarray(pc_o3d.points).astype(np.float32)
    return pc

def generate_zc_from_item_v2(spaghetti, item: Union[List, int]):
    device = spaghetti.device
    item = torch.tensor(item)
    if item.ndim == 1:
        item=item.unsqueeze(0)
    item = item.to(device).long()
    sjs, zas, gmms = spaghetti.get_embeddings(item)[:3]
    zcs = spaghetti.merge_zh(sjs, gmms)[0]
    return zcs

def generate_zc_from_sj_gaus_v2(spaghetti, sj: Union[torch.Tensor, np.ndarray], gaus: Union[torch.Tensor, np.ndarray]):
    """
    Input:
        sj: [B,16,512] or [16,512]
        gaus: [B,16,16] or [16,16]
    Output:
        zc: [B,16,512]
    """
    device = spaghetti.device
    sj = jutils.nputil.np2th(sj)
    gaus = jutils.nputil.np2th(gaus)
    assert sj.dim() == gaus.dim()
    
    if sj.dim() == 2:
        sj = sj.unsqueeze(0)
    batch_sj = sj.to(device)
    batch_gmms = batch_gaus_to_gmms(gaus, device)
    zcs, _ = spaghetti.merge_zh(batch_sj, batch_gmms)
    return zcs
    
def generate_zc_from_za_v2(spaghetti, za: Union[torch.Tensor, np.ndarray]):
    device = spaghetti.device
    za = jutils.nputil.np2th(za).to(device)
    sjs, gmms = spaghetti.decomposition_control(za)
    zcs, _ = spaghetti.merge_zh(sjs, gmms)
    return zcs

def generate_zc_from_single_phase_latent_v2(spaghetti, sj_gaus: Union[torch.Tensor, np.ndarray]):
    device = spaghetti.device
    sj_gaus = jutils.nputil.np2th(sj_gaus).to(device)
    sj, gaus = sj_gaus.split(split_size=[512,16], dim=-1)
    zcs = generate_zc_from_sj_gaus_v2(spaghetti, sj, gaus)
    return zcs

@torch.no_grad()
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
    gaus = jutils.nputil.np2th(gaus).to(device)
    if len(gaus.shape) < 3:
        gaus = gaus.unsqueeze(0) # expand dim for batch

    B,G,_ = gaus.shape
    mu = gaus[:,:,:3].reshape(B,1,G,3)
    eivec = gaus[:,:,3:12].reshape(B,1,G,3,3)
    pi = gaus[:,:,12].reshape(B,1,G)
    eival = gaus[:,:,13:16].reshape(B,1,G,3)
    
    return [mu, eivec, pi, eival]
