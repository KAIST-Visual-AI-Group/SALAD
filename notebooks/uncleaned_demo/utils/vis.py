from PIL import Image

import jutils
import numpy as np
import torch

from pvd.utils.spaghetti_utils import batch_gaus_to_gmms, get_occ_func


def make_img_grid(img_lists, nrow=5):
    """
    Creates an image grid using the provided images.

    Args:
        img_lists: Lists of list of images
        nrow: Number of samples in a row.
    """
    for img_list in img_lists:
        assert isinstance(img_list, list)
        assert len(img_lists[0]) == len(img_list)
        assert len(img_list) > 0
        assert isinstance(img_list[0], Image.Image)

    img_pairs = list(zip(*img_lists))

    indices = list(range(0, len(img_lists[0]), nrow))
    if indices[-1] != len(img_lists[0]):
        indices.append(len(img_lists[0]))
    starts, ends = indices[0:-1], indices[1:]

    image_arr = []
    for start, end in zip(starts, ends):
        row_images = []
        for i in range(start, end):
            for img in img_pairs[i]:
                row_images.append(img)
        image_arr.append(row_images)

    return jutils.imageutil.merge_images(image_arr)


def decode_and_render(gmm, zh, spaghetti, mesher, camera_kwargs):
    if gmm.ndim == 2:
        gmm = gmm[None]
    if zh.ndim == 2:
        zh = zh[None]

    vert, face = decode_gmm_and_intrinsic(
        spaghetti, mesher, gmm, zh
    )
    img = render_mesh(vert, face, camera_kwargs)
    return img


def decode_gmm_and_intrinsic(spaghetti, mesher, gmm, intrinsic, verbose=False):
    assert gmm.ndim == 3 and intrinsic.ndim == 3
    assert gmm.shape[0] == intrinsic.shape[0]
    assert gmm.shape[1] == intrinsic.shape[1]

    zc, _ = spaghetti.merge_zh(intrinsic, batch_gaus_to_gmms(gmm, gmm.device))

    mesh = mesher.occ_meshing(
        decoder=get_occ_func(spaghetti, zc),
        res=256,
        get_time=False,
        verbose=False,
    )
    assert mesh is not None, "Marching cube failed"
    vert, face = list(map(lambda x: jutils.thutil.th2np(x), mesh))
    assert isinstance(vert, np.ndarray) and isinstance(face, np.ndarray)

    if verbose:
        print(f"Vert: {vert.shape} / Face: {face.shape}")

    # Free GPU memory after computing mesh
    _ = zc.cpu()
    jutils.sysutil.clean_gpu()
    if verbose:
        print("Freed GPU memory")

    return vert, face


def render_mesh(vert, face, camera_kwargs):
    img = jutils.fresnelvis.renderMeshCloud(
        mesh={"vert": vert / 2, "face": face}, **camera_kwargs
    )
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    assert isinstance(img, Image.Image)
    return img

