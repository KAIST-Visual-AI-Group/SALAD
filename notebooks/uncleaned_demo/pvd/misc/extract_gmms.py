import argparse
from pvd.dataset import *
from pvd.diffusion.ldm import *
from pvd.diffusion.phase1 import *
import torch
import numpy as np
import jutils
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
from eval3d import Evaluator
import io
import os
from PIL import Image
from datetime import datetime

def load_ldm(path, strict=True):
    path = Path(path)
    assert path.exists()
    
    ldm = GaussianSALDM.load_from_checkpoint(path, strict=strict).cuda(0)
    ldm.eval()
    for p in ldm.parameters(): p.requires_grad_(False)
    dataset = ldm._build_dataset("val")
    return ldm, dataset

def sample_gmms(ldm, num_shapes):
    ldm_gaus = ldm.sample(num_shapes).to("cpu")
    return ldm_gaus

def main(args):
    assert Path(args.save_dir).exists()
    now = datetime.now().strftime("%m-%d-%H%M%S")
    ldm, dataset = load_ldm(args.ldm_path, strict=False)

    ldm_gaus = sample_gmms(ldm, args.num_shapes)
    ldm_gaus = jutils.thutil.th2np(ldm_gaus)
    
    with h5py.File(f"{args.save_dir}/{len(ldm_gaus)}-{now}.hdf5", "w") as f:
        f["data"] = ldm_gaus.astype(np.float32)

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldm_path", type=str)
    parser.add_argument("--num_shapes", type=int)
    parser.add_argument("--save_dir", type=str)

    args = parser.parse_args()
    main(args)
