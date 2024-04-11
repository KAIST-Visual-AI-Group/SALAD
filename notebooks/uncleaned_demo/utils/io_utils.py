"""
io_utils.py

A collection of utility functions for I/O.
"""

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

####
# NOTE: 20240411
# Seungwoo: Align the model loading procedure with other notebooks and scripts.
from demo.utils import paths
from pvd.diffusion.phase1 import GaussianSALDM
from pvd.diffusion.phase1_sym import GaussianSymSALDM
from pvd.diffusion.ldm import SpaghettiConditionSALDM
from pvd.diffusion.single_phase import SingleSALDM
from pvd.utils.spaghetti_utils import load_mesher, load_spaghetti
####


def create_out_dir(log_root: Path):
    assert isinstance(log_root, Path)
    out_dir = log_root / datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    assert out_dir.exists(), f"Failed to create output directory {str(out_dir)}"
    print(f"Created directory: {str(out_dir)}")
    return out_dir

def load_dataset(dataset_path, device):
    assert isinstance(dataset_path, (str, Path))
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)
    assert dataset_path.exists()

    with h5py.File(str(dataset_path), "r") as f:
        # Load GMMs representing half of shapes
        gmms = np.array(f["gmms"][:])
        gmms = torch.from_numpy(gmms)
        
        # Load intrinsics representing entire shapes
        zhs = np.array(f["zhs"][:])
        zhs = torch.from_numpy(zhs)

        #### 0308
        zbs = np.array(f["zbs"][:])
        zbs = torch.from_numpy(zbs)
        ####

        shape_ids = [shape_id for shape_id in f["shape_ids"].asstr()]
        part_labels = np.array(f["part_labels"][:])
        part_labels = torch.from_numpy(part_labels)

    assert gmms.ndim == 3, f"Expected GMMs to be 3D tensor got {gmms.ndim}D"
    assert (
        part_labels.ndim == 2
    ), f"Expected part labels to be 2D tensor got {part_labels.ndim}D"
    assert len(shape_ids) == gmms.shape[0] and len(shape_ids) == part_labels.shape[0]

    print(f"GMM shape: {gmms.shape}")
    print(f"Part label shape: {part_labels.shape}")
    print(f"Num ShapeNet IDs: {len(shape_ids)}")

    return gmms, zhs, zbs, part_labels, shape_ids


def load_phase1_and_phase2(category, use_sym, device):
    
    if category == "chairs":        
        if use_sym:
            raise NotImplementedError("Blocked the use of symmetric model")
            p1_ckpt_dir = Path(
                "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/phase1-sym/augment_partial_norm_final_0214/0215_000453/checkpoints"
            )
        else:
            p1_ckpt_dir = Path(
                "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/phase1/partial_norm_0219/0219_124456/checkpoints"
            )
        p1_model = load_phase1(p1_ckpt_dir, use_sym, device)
        if p1_model.hparams.get("global_normalization"):
            p1_model.hparams.dataset_kwargs.data_path= (
                "/home/juil/docker_home/projects/3D_CRISPR/crispr/data/pre_trained_spaghetti_latent_params/5401_2023-01-19-15-02-36_half_intersec_im_net_train.hdf5"
            )
            print("Updated 'dataset_kwargs' for phase 1 model trained on normalized data")

        # Load phase 2
        p2_ckpt_dir = Path(
            "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/phase2/augment_final_0214/0214_202607/checkpoints"
        )
        p2_model = load_phase2(p2_ckpt_dir, device)
        
    elif category == "airplanes":
        if use_sym:
            raise NotImplementedError()
            p1_ckpt_dir = Path(
                "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/phase1-sym-airplane/augment_partial_norm_final_0214/0216_182155/checkpoints"
            )
        else:
            ####
            p1_ckpt_dir = Path(
                "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/phase1-airplane/partial_norm_0219/0222_214952/checkpoints"
            )
        p1_model = load_phase1(p1_ckpt_dir, use_sym, device)
        if p1_model.hparams.get("global_normalization"):
            p1_model.hparams.dataset_kwargs.data_path=(
                "/home/juil/docker_home/projects/3D_CRISPR/crispr/data/pre_trained_spaghetti_latent_params/3236-02-14-183744-inversion_with_replacement_half.hdf5"
            )
            print("Updated 'dataset_kwargs' for phase 1 model trained on normalized data")

        p2_ckpt_dir = Path(
            "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/phase2-airplane/augment_final_0214/0214_202550/checkpoints"
        )
        p2_model = load_phase2(p2_ckpt_dir, device)

    elif category == "tables":
        p1_ckpt_dir = Path(
            "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/phase1-airplane/partial_norm_0302/0302_003448/checkpoints"
        )
        p1_model = load_phase1(p1_ckpt_dir, False, device)
        
        if p1_model.hparams.get("global_normalization"):
            pass
            ####
            p1_model.hparams.dataset_kwargs.data_path = (
                "/home/juil/docker_home/projects/3D_CRISPR/crispr/data/pre_trained_spaghetti_latent_params/tables_4852_03-02-001544.hdf5"
            )
            print("Updated 'dataset_kwargs' for phase 1 model trained on normalized data")
            ####
        p2_ckpt_dir = Path(
            "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/phase2-airplane/0302/0302_003607/checkpoints"
        )
        p2_model = load_phase2(p2_ckpt_dir, device)
    else:
        raise NotImplementedError(f"Unknown category {category}")        
    
    return p1_model, p2_model


def load_single_phase(category, device):

    if category == "chairs":
        ckpt_dir = Path(
            "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/single_phase/partial_norm_x2_fast/0302_162942/checkpoints"
        )
    elif category == "airplanes":
        ckpt_dir = Path(
            "/home/juil/docker_home/projects/3D_CRISPR/pointnet-vae-diffusion/pvd/results/single_phase-airplane/partial_norm_x2_fast/0302_163844/checkpoints"
        )
    else:
        raise NotImplementedError(f"Unknown category {category}")
    assert ckpt_dir.exists(), f"Directory {str(ckpt_dir)} does not exist"

    model = SingleSALDM.load_from_checkpoint(
        str(ckpt_dir / "epoch=9999-val_loss=0.0000.ckpt"),
        strict=False,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model = model.to(device)

    if category == "chairs":
        if model.hparams.get("global_normalization"):
            model.hparams.dataset_kwargs.data_path = (
            "/home/juil/docker_home/projects/3D_CRISPR/crispr/data/pre_trained_spaghetti_latent_params/5401_2023-01-19-15-02-36_intersec_im_net_train.hdf5"
        )
    elif category == "airplanes":
        if model.hparams.get("global_normalization"):
            model.hparams.dataset_kwargs.data_path = (
                "/home/juil/docker_home/projects/3D_CRISPR/crispr/data/pre_trained_spaghetti_latent_params/3236-02-14-183744-inversion_with_replacement.hdf5"
            )
    else:
        raise NotImplementedError(f"Unknown category {category}")

    print(f"Loaded model from {str(ckpt_dir)}")
    print(f"Model type: {type(model)}")

    return model


def load_phase1(ckpt_dir, use_sym, device):
    assert isinstance(ckpt_dir, (str, Path))
    if isinstance(ckpt_dir, str):
        ckpt_dir = Path(ckpt_dir)
    assert ckpt_dir.exists()

    if use_sym:
        raise NotImplementedError("Blocked using non-symmetric model")
        print("Loading symmetric phase 1 model")
        ckpt_file = ckpt_dir / "epoch=4999-val_loss=0.0000.ckpt"
        model = GaussianSymSALDM.load_from_checkpoint(
            str(ckpt_file),
            strict=False,
        )
    else:
        print("Loading phase 1 model")
        ckpt_file = ckpt_dir / "epoch=4999-val_loss=0.0000.ckpt"

        try:
            model = GaussianSALDM.load_from_checkpoint(
                str(ckpt_file),
                strict=False,
            )
        except:
            print("Loading the latest checkpoint.")
            ckpt_file = ckpt_dir / "last.ckpt"
            model = GaussianSALDM.load_from_checkpoint(
                str(ckpt_file),
                strict=False,
            )

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model = model.to(device)

    print(f"Loaded model from {str(ckpt_file)}")
    print(f"Model type: {type(model)}")

    return model


def load_phase2(ckpt_dir, device):
    assert isinstance(ckpt_dir, (str, Path))
    if isinstance(ckpt_dir, str):
        ckpt_dir = Path(ckpt_dir)
    assert ckpt_dir.exists()

    ckpt_file = ckpt_dir / "epoch=4999-val_loss=0.0000.ckpt"
    model = SpaghettiConditionSALDM.load_from_checkpoint(
        str(ckpt_file),
        strict=False,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model = model.to(device)

    print(f"Loaded model from {str(ckpt_file)}")
    print(f"Model type: {type(model)}")

    return model


def load_spaghetti_and_mesher(device, spaghetti_tag):
    spaghetti = load_spaghetti(device, spaghetti_tag)
    print(f"SPAGHETTI type / tag: {type(spaghetti)} / {spaghetti_tag}")

    mesher = load_mesher(device)
    print(f"Mesher type: {type(mesher)}")

    return spaghetti, mesher


