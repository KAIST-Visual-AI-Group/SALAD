import numpy as np
import trimesh
import matplotlib.pyplot as plt
from salad.utils import nputil, thutil, fresnelvis
from PIL import Image


def render_pointcloud(
    pointcloud,
    camPos=np.array([-2, 2, -2]),
    camLookat=np.array([0.0, 0.0, 0.0]),
    camUp=np.array([0, 1, 0]),
    camHeight=2,
    resolution=(512, 512),
    samples=16,
    cloudR=0.006,
):
    pointcloud = thutil.th2np(pointcloud)
    img = fresnelvis.renderMeshCloud(
        cloud=pointcloud,
        camPos=camPos,
        camLookat=camLookat,
        camUp=camUp,
        camHeight=camHeight,
        resolution=resolution,
        samples=samples,
        cloudR=cloudR,
    )
    return Image.fromarray(img)


def render_mesh(
    vert,
    face,
    camPos=np.array([-2, 2, -2]),
    camLookat=np.array([0, 0, 0.0]),
    camUp=np.array([0, 1, 0]),
    camHeight=2,
    resolution=(512, 512),
    samples=16,
):
    vert, face = list(map(lambda x: thutil.th2np(x), [vert, face]))
    mesh = {"vert": vert, "face": face}
    img = fresnelvis.renderMeshCloud(
        mesh=mesh,
        camPos=camPos,
        camLookat=camLookat,
        camUp=camUp,
        camHeight=camHeight,
        resolution=resolution,
        samples=samples, 
    )
    return Image.fromarray(img)


def render_gaussians(
    gaussians,
    is_bspnet=False,
    multiplier=1.0,
    gaussians_colors=None,
    attn_map=None,
    camPos=np.array([-2, 2, -2]),
    camLookat=np.array([0.0, 0, 0]),
    camUp=np.array([0, 1, 0]),
    camHeight=2,
    resolution=(512, 512),
    samples=16,
):
    gaussians = thutil.th2np(gaussians)
    N = gaussians.shape[0]
    cmap = plt.get_cmap("jet")

    if attn_map is not None:
        assert N == attn_map.shape[0]
        vmin, vmax = attn_map.min(), attn_map.max()
        if vmin == vmax:
            normalized_attn_map = np.zeros_like(attn_map)
        else:
            normalized_attn_map = (attn_map - vmin) / (vmax - vmin)

        cmap = plt.get_cmap("viridis")

    lights = "rembrandt"
    camera_kwargs = dict(
        camPos=camPos,
        camLookat=camLookat,
        camUp=camUp,
        camHeight=camHeight,
        resolution=resolution,
        samples=samples,
    )
    renderer = fresnelvis.FresnelRenderer(lights=lights, camera_kwargs=camera_kwargs)
    for i, g in enumerate(gaussians):
        if is_bspnet:
            mu, eival, eivec = g[:3], g[3:6], g[6:15]
        else:
            mu, eivec, eival = g[:3], g[3:12], g[13:]
        R = eivec.reshape(3, 3).T
        scale = multiplier * np.sqrt(eival)
        scale_transform = np.diag((*scale, 1))
        rigid_transform = np.hstack((R, mu.reshape(3, 1)))
        rigid_transform = np.vstack((rigid_transform, [0, 0, 0, 1]))
        sphere = trimesh.creation.icosphere()
        sphere.apply_transform(scale_transform)
        sphere.apply_transform(rigid_transform)
        if attn_map is None and gaussians_colors is None:
            color = np.array(cmap(i / N)[:3])
        elif attn_map is not None:
            color = np.array(cmap(normalized_attn_map[i])[:3])
        else:
            color = gaussians_colors[i]

        renderer.add_mesh(
            sphere.vertices, sphere.faces, color=color, outline_width=None
        )
    image = renderer.render()
    return Image.fromarray(image)
