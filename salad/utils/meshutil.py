from enum import Enum

import numpy as np
import torch
import trimesh

from salad.utils import thutil


def write_obj(name: str, vertices: np.ndarray, faces: np.ndarray):
    """
    name: filename
    vertices: (V,3)
    faces: (F,3) Assume the mesh is a triangle mesh.
    """
    vertices = thutil.th2np(vertices)
    faces = thutil.th2np(faces).astype(np.uint32)
    fout = open(name, "w")
    for ii in range(len(vertices)):
        fout.write(
            "v "
            + str(vertices[ii, 0])
            + " "
            + str(vertices[ii, 1])
            + " "
            + str(vertices[ii, 2])
            + "\n"
        )
    for ii in range(len(faces)):
        fout.write(
            "f "
            + str(faces[ii, 0] + 1)
            + " "
            + str(faces[ii, 1] + 1)
            + " "
            + str(faces[ii, 2] + 1)
            + "\n"
        )
    fout.close()


def write_obj_triangle(name: str, vertices: np.ndarray, triangles: np.ndarray):
    fout = open(name, "w")
    for ii in range(len(vertices)):
        fout.write(
            "v "
            + str(vertices[ii, 0])
            + " "
            + str(vertices[ii, 1])
            + " "
            + str(vertices[ii, 2])
            + "\n"
        )
    for ii in range(len(triangles)):
        fout.write(
            "f "
            + str(triangles[ii, 0] + 1)
            + " "
            + str(triangles[ii, 1] + 1)
            + " "
            + str(triangles[ii, 2] + 1)
            + "\n"
        )
    fout.close()


def write_obj_polygon(name: str, vertices: np.ndarray, polygons: np.ndarray):
    fout = open(name, "w")
    for ii in range(len(vertices)):
        fout.write(
            "v "
            + str(vertices[ii][0])
            + " "
            + str(vertices[ii][1])
            + " "
            + str(vertices[ii][2])
            + "\n"
        )
    for ii in range(len(polygons)):
        fout.write("f")
        for jj in range(len(polygons[ii])):
            fout.write(" " + str(polygons[ii][jj] + 1))
        fout.write("\n")
    fout.close()


def read_obj(name: str):
    verts = []
    faces = []
    with open(name, "r") as f:
        lines = [line.rstrip() for line in f]

        for line in lines:
            if line.startswith("v "):
                verts.append(np.float32(line.split()[1:4]))
            elif line.startswith("f "):
                faces.append(
                    np.int32([item.split("/")[0] for item in line.split()[1:4]])
                )

        v = np.vstack(verts)
        f = np.vstack(faces) - 1
        return v, f


def scene_as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None
        else:
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                    if g.faces.shape[1] == 3
                )
            )
    else:
        mesh = scene_or_mesh

    return mesh


def get_center(verts):
    max_vals = verts.max(0)
    min_vals = verts.min(0)
    center = (max_vals + min_vals) / 2
    return center


def to_center(verts):
    verts -= get_center(verts)[None, :]
    return verts


def get_offset_and_scale(verts, radius=1.0):
    verts = thutil.th2np(verts)
    verts = verts.copy()

    offset = get_center(verts)[None, :]
    verts -= offset
    scale = 1 / np.linalg.norm(verts, axis=1).max() * radius

    return offset, scale


def normalize_mesh(mesh: trimesh.Trimesh):
    # unit cube normalization
    v, f = np.array(mesh.vertices), np.array(mesh.faces)
    maxv, minv = np.max(v, 0), np.min(v, 0)
    offset = minv
    v = v - offset
    scale = np.sqrt(np.sum((maxv - minv) ** 2))
    v = v / scale
    normed_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    return dict(mesh=normed_mesh, offset=offset, scale=scale)


def normalize_scene(scene: trimesh.Scene):
    mesh_merged = scene_as_mesh(scene)

    out = normalize_mesh(mesh_merged)
    offset = out["offset"]
    scale = out["scale"]

    submesh_normalized_list = []
    for i, submesh in enumerate(list(scene.geometry.values())):
        v, f = np.array(submesh.vertices), np.array(submesh.faces)
        v = v - offset
        v = v / scale
        submesh_normalized_list.append(trimesh.Trimesh(v, f))

    return trimesh.Scene(submesh_normalized_list)


class SampleBy(Enum):
    AREAS = 0
    FACES = 1
    HYB = 2


def get_faces_normals(mesh):
    if type(mesh) is not torch.Tensor:
        vs, faces = mesh
        vs_faces = vs[faces]
    else:
        vs_faces = mesh
    if vs_faces.shape[-1] == 2:
        vs_faces = torch.cat(
            (
                vs_faces,
                torch.zeros(
                    *vs_faces.shape[:2], 1, dtype=vs_faces.dtype, device=vs_faces.device
                ),
            ),
            dim=2,
        )
    face_normals = torch.cross(
        vs_faces[:, 1, :] - vs_faces[:, 0, :], vs_faces[:, 2, :] - vs_faces[:, 1, :]
    )
    return face_normals


def compute_face_areas(mesh):
    face_normals = get_faces_normals(mesh)
    face_areas = torch.norm(face_normals, p=2, dim=1)
    face_areas_ = face_areas.clone()
    face_areas_[torch.eq(face_areas_, 0)] = 1
    face_normals = face_normals / face_areas_[:, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def sample_uvw(shape, device):
    u, v = torch.rand(*shape, device=device), torch.rand(*shape, device=device)
    mask = (u + v).gt(1)
    u[mask], v[mask] = -u[mask] + 1, -v[mask] + 1
    w = -u - v + 1
    uvw = torch.stack([u, v, w], dim=len(shape))
    return uvw


def sample_on_mesh(mesh, num_samples: int, face_areas=None, sample_s=SampleBy.HYB):
    vs, faces = mesh
    if faces is None:  # sample from pc
        uvw = None
        if vs.shape[0] < num_samples:
            chosen_faces_inds = torch.arange(vs.shape[0])
        else:
            chosen_faces_inds = torch.argsort(torch.rand(vs.shape[0]))[:num_samples]
        samples = vs[chosen_faces_inds]
    else:
        weighted_p = []
        if sample_s == SampleBy.AREAS or sample_s == SampleBy.HYB:
            if face_areas is None:
                face_areas, _ = compute_face_areas(mesh)
            face_areas[torch.isnan(face_areas)] = 0
            weighted_p.append(face_areas / face_areas.sum())
        if sample_s == SampleBy.FACES or sample_s == SampleBy.HYB:
            weighted_p.append(torch.ones(mesh[1].shape[0], device=mesh[0].device))
        chosen_faces_inds = [
            torch.multinomial(weights, num_samples // len(weighted_p), replacement=True)
            for weights in weighted_p
        ]
        if sample_s == SampleBy.HYB:
            chosen_faces_inds = torch.cat(chosen_faces_inds, dim=0)
        chosen_faces = faces[chosen_faces_inds]
        uvw = sample_uvw([num_samples], vs.device)
        samples = torch.einsum("sf,sfd->sd", uvw, vs[chosen_faces])
    return samples, chosen_faces_inds, uvw


def repair_normals(v, f):
    mesh = trimesh.Trimesh(v, f)
    trimesh.repair.fix_normals(mesh)
    v = mesh.vertices
    f = np.asarray(mesh.faces)
    return v, f
