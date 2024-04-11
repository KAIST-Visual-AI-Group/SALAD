import jutils
import numpy as np
import torch
from rich.progress import track


####
# NOTE: 20240411
# Seungwoo: Redirect this import path to the module 'salad'.
# The functions are already in our repository.
from pvd.utils.spaghetti_utils import batch_gaus_to_gmms, get_occ_func, reflect_and_concat_gmms
####


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


def clip_gmm_eigvals(gmms):
    """
    Clips negative eigenvalues.
    """
    assert gmms.ndim == 3  # (B, N, 16)

    if isinstance(gmms, torch.Tensor):
        gmms_ = gmms.clone()  # Copy to prevent overwrite
        gmms_[..., 13:16] = torch.where(
            gmms_[..., 13:16] < 0.0,
            1e-4,
            gmms_[..., 13:16],
        )
    elif isinstance(gmms, np.ndarray):
        gmms_ = gmms.copy()  # Copy to prevent overwrite
        gmms_[..., 13:16] = np.where(
            gmms_[..., 13:16] < 0.0,
            1e-4,
            gmms_[..., 13:16],
        )
    else:
        raise AssertionError(f"Expected torch Tensor or Numpy Array, got {type(gmms)}")
    return gmms_


def ablate_symmetric_parts(gmms, ablation_idx, verbose=False):
    """
    Ablates parts symmetric to each other from the given GMMs.

    TODO: Extend 'ablation_idx' to list of part indices.

    Args:
        gmms: (B, N1, D), A batch of GMMs each consists of N1 parts.
        ablation_idx: A integer in (0, N) specifying the part to ablate.
            The index indicates the location of the part from the left when parts are
            sorted along x-axis.

    Returns:
        gmms_ablate: (B, N2, D), A batch of GMMs whose parts are ablated.
            Here, N2 = N1 - 2 * len(ablation_idx).
    """
    assert isinstance(gmms, torch.Tensor) or isinstance(
        gmms, np.ndarray
    ), f"Got {type(gmms)}"
    assert gmms.ndim == 3 and gmms.size(2) == 16, f"Input GMM shape: {gmms.shape}"
    assert (
        isinstance(ablation_idx, int)
        and ablation_idx >= 0
        and ablation_idx < gmms.size(1)
    )

    n_shape = gmms.shape[0]
    n_part = gmms.shape[1]
    n_dim = gmms.shape[2]
    gmms = jutils.nputil.np2th(gmms)

    gmm_x = gmms[..., 0]
    assert gmm_x.ndim == 2, f"{gmm_x.ndim} {gmm_x.shape}"
    indices = torch.argsort(gmm_x, dim=1)  # (B, N1)
    assert indices.ndim == 2, f"{indices.ndim} {indices.shape}"

    assert torch.all(indices.min(dim=1)[0] == 0) and torch.all(
        indices.max(dim=1)[0] == (n_part - 1)
    ), f"{indices.min(dim=1)} {indices.max(dim=1)}"

    part_left = indices[..., ablation_idx]  # (B,)
    part_right = indices[..., -(ablation_idx + 1)]  # (B,)
    assert (
        part_left.ndim == 1
        and part_right.ndim == 1
        and part_left.size(0) == part_right.size(0) == n_shape
    )

    # Loop over the shapes and remove specified parts
    gmms_ablate = []
    for gmm, left_indices, right_indices in zip(gmms, part_left, part_right):
        mask = torch.ones(n_part).to(gmms.device)  # (N1,)
        mask[left_indices] = 0.0
        mask[right_indices] = 0.0
        to_keep = mask.nonzero(as_tuple=True)
        gmms_ablate.append(gmm[to_keep][None])
    gmms_ablate = torch.cat(gmms_ablate, dim=0)
    assert (
        gmms_ablate.ndim == 3
        and gmms_ablate.size(0) == n_shape
        and gmms_ablate.size(1) == (n_part - 2)
        and gmms_ablate.size(2) == n_dim
    )

    # Also, return the indices of the parts ablated
    if part_left.ndim == 1:
        part_left = part_left[..., None]  # (B,) -> (B, 1)
    if part_right.ndim == 1:
        part_right = part_right[..., None]  # (B,) -> (B, 1)
    ablated_indices = torch.cat([part_left, part_right], dim=1)
    assert ablated_indices.ndim == 2

    return gmms_ablate, ablated_indices


def ablate_semantic_parts(parts, ablation_idx, verbose=False):
    """
    Ablates parts using the known part indices.

    Args:
        parts: (B, 16, 16) or (B, 16, 528), 
            the former is for GMM editing, the latter is for single-phase model editing.
        ablation_idx: List,
    """
    assert isinstance(parts, torch.Tensor) or isinstance(
        parts, np.ndarray
    ), f"Got {type(parts)}"
    assert parts.ndim == 3 and parts.size(2) in (16, 528), f"'parts' shape: {parts.shape}"

    n_part = parts.shape[1]
    parts = jutils.nputil.np2th(parts)

    # Loop over the shapes and remove specific parts
    gmms_ablate = []
    ablated_indices = []
    for gmm in parts:
        to_remove = ablation_idx
        assert isinstance(to_remove, list) and len(to_remove) > 0

        to_keep = list(set(range(n_part)) - set(to_remove))
        assert len(to_keep) > 0
        to_remove = torch.tensor(to_remove).to(gmm.device)

        gmms_ablate.append(gmm[to_keep][None])
        ablated_indices.append(to_remove[None])
    gmms_ablate = torch.cat(gmms_ablate, dim=0)
    ablated_indices = torch.cat(ablated_indices, dim=0)

    return gmms_ablate, ablated_indices


def edit_gmm(model, gmms, part_indices, start_t, recon_sym_parts, method="sdedit", verbose=True):
    """
    Performs shape completion given incomplete sets of GMMs.

    Args:
        model:
        gmms:
        part_indices:
        start_t:
        recon_sym_parts: A flag for reconstructing the other half of the shape.
            Used to output full GMMs when the symmetric phase 1 model is used.
        method:
        verbose:
    """
    assert start_t >= 0 and start_t <= model.var_sched.num_steps
    assert method in ("mcg", "repaint", "sdedit")
    assert gmms.ndim == 3, f"Got 'gmms' of dimension {gmms.ndim}"

    timesteps = list(range(start_t, 0, -1))
    if verbose:
        # timesteps = tqdm(timesteps)
        # timesteps = track(timesteps)
        pass

    n_part = gmms.size(1)
    mask = torch.zeros(n_part).to(gmms.device)
    mask[part_indices] = 1.0

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
            x[:, mask == 1.0] = x_[:, mask == 1.0]
        else:
            raise NotImplementedError()
    assert x.shape == gmms.shape, f"'x' shape: {x.shape} / 'gmms' shape: {gmms.shape}"
    assert torch.all(x[:, mask == 0.0] == x0[:, mask == 0.0]), (
        "The algorithm must not edit the unmasked parts"
    )

    ####
    # if n_part == 8:  # using symmetry
    if recon_sym_parts:  # using symmetry
    ####
        x = reflect_and_concat_gmms(x)
        assert x.size(1) == n_part * 2

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
    return x


def edit_zh(model, zhs, gmms, part_indices, start_t, method="sdedit", verbose=True):
    assert start_t >= 0 and start_t <= model.var_sched.num_steps
    assert method in ("mcg", "repaint", "sdedit")
    assert zhs.ndim == 3, f"Got 'zhs' of shape {zhs.shape}"
    assert gmms.ndim == 3, f"Got 'gmms' of shape {gmms.shape}"

    timesteps = list(range(start_t, 0, -1))
    if verbose:
        # timesteps = tqdm(timesteps)
        # timesteps = track(timesteps)
        pass

    mask = torch.zeros(zhs.size(1)).to(zhs.device)
    mask[part_indices] = 1.0

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
            x[:, mask == 1.0] = x_[:, mask == 1.0]
        else:
            raise NotImplementedError()
    assert x.shape == zhs.shape, f"'x' shape: {x.shape} / 'zhs' shape: {zhs.shape}"
    assert torch.all(x[:, mask == 0.0] == x0[:, mask == 0.0]), (
        "The algorithm must not edit the unmasked parts"
    )

    return x


def edit_single_phase(model, concats, part_indices, start_t, method="sdedit", verbose=True):
    assert start_t >= 1 and start_t <= model.var_sched.num_steps
    assert method in ("mcg", "repaint", "sdedit")
    assert concats.ndim == 3, f"Got 'concats' of dimension {concats.ndim}"

    timesteps = list(range(start_t, 0, -1))
    if verbose:
        # timesteps = tqdm(timesteps)
        # timesteps = track(timesteps)
        pass

    mask = torch.zeros(concats.size(1)).to(concats.device)
    mask[part_indices] = 1.0

    x0 = concats

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
            x[:, mask == 1.0] = x_[:, mask == 1.0]
        else:
            raise NotImplementedError()
    assert x.shape == concats.shape, f"'x' shape: {x.shape} / 'concats' shape: {concats.shape}"
    assert torch.all(x[:, mask == 0.0] == x0[:, mask == 0.0]), (
        "The algorithm must not edit the unmasked parts"
    )

    return x


def run_sdedit(model, gmm, part_indices, start_t: int):
    assert start_t >= 1 and start_t <= model.diffusion.var_sched.num_steps

    mask = torch.zeros(gmm.size(1)).to(gmm.device)
    mask[part_indices] = 1.0

    x0 = gmm
    x = model.diffusion.add_noise(x0, start_t)[0]
    for t in range(start_t, 0, -1):
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        alpha = model.diffusion.var_sched.alphas[t]
        alpha_bar = model.diffusion.var_sched.alpha_bars[t]
        sigma = model.diffusion.var_sched.get_sigmas(t, flexibility=0.0)

        c0 = 1.0 / torch.sqrt(alpha)
        c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

        beta = model.diffusion.var_sched.betas[[t] * x.size(0)]
        e_theta = model.diffusion.net(x, beta=beta, context=None)
        x_ = c0 * (x - c1 * e_theta) + sigma * z

        x = model.diffusion.add_noise(x0, t - 1)[0]
        x[:, mask == 1.0] = x_[:, mask == 1.0]

    edited_part = x.clone()
    edited_part[:, mask != 1.0] = 0.0

    return x


def add_noise(model, x, t):
    alpha_bar = model.var_sched.alpha_bars[t]

    # Drift and diffusion coefficient (DDPM)
    c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # (B, 1, 1)
    c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (B, 1, 1)

    # Sample Brownian motion
    e_rand = torch.randn_like(x)  # (B, N, d)

    return c0 * x + c1 * e_rand


def denoise_one_step(model, x_prev, t):
    z = torch.randn_like(x_prev) if t > 1 else torch.zeros_like(x_prev)

    alpha = model.var_sched.alphas[t]
    alpha_bar = model.var_sched.alpha_bars[t]
    sigma = model.var_sched.get_sigmas(t, flexibility=0.0)

    c0 = 1.0 / torch.sqrt(alpha)
    c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

    beta = model.var_sched.betas[[t] * x_prev.size(0)]
    e_theta = model.net(x_prev, beta=beta)

    x_next = c0 * (x_prev - c1 * e_theta) + sigma * z

    return x_next


def denoise_one_step_cond(model, x_prev, cond, t):
    z = torch.randn_like(x_prev) if t > 1 else torch.zeros_like(x_prev)
    
    alpha = model.var_sched.alphas[t]
    alpha_bar = model.var_sched.alpha_bars[t]
    sigma = model.var_sched.get_sigmas(t, flexibility=0.0)

    c0 = 1.0 / torch.sqrt(alpha)
    c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

    beta = model.var_sched.betas[[t] * x_prev.size(0)]
    e_theta = model.net(x_prev, beta=beta, context=cond)

    x_next = c0 * (x_prev - c1 * e_theta) + sigma * z

    return x_next

def denoise_tweedie(model, x_prev, t, cond=None):
    """
    Denoises the data in a single step using Tweedie's formula.
    """
    device = model.device
    x_prev = jutils.nputil.np2th(x_prev).to(device)
    if x_prev.ndim == 2:
        x_prev = x_prev.unsqueeze(0)
    
    alpha_bar = model.var_sched.alpha_bars[t].view(-1,1,1)
    beta = model.var_sched.betas[t].view(1)
    if cond is None:
        e_theta = model.net(x_prev, beta=beta)
    else:
        device = model.device
        cond = jutils.nputil.np2th(cond).to(device)
        beta = beta.expand(cond.shape[0])
        e_theta = model.net(x_prev, beta=beta, context=cond)

    score = -1 / torch.sqrt(1-alpha_bar) * e_theta
    
    x_denoised = (x_prev + (1 - alpha_bar) * score) / torch.sqrt(alpha_bar)
    return x_denoised
