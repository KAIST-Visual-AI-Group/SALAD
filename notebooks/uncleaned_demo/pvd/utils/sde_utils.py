import torch
from pvd.utils.spaghetti_utils import *
import jutils
import numpy as np
from tqdm import tqdm


def add_noise(model, x, t, gaus_indices=None):
    device = model.device
    x = jutils.nputil.np2th(x).to(device)
    alpha_bar = model.var_sched.alpha_bars[t]
    # Drift and diffusion coefficient (DDPM)
    c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # (B, 1, 1)
    c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (B, 1, 1)
    # Sample Brownian motion
    e_rand = torch.randn_like(x)  # (B, N, d)
    if gaus_indices is not None:
        e_mask = torch.zeros_like(x)
        e_mask[:, gaus_indices] = 1
        e_rand *= e_mask # add noise on mask=1 gaussians only.

    return c0 * x + c1 * e_rand


def denoise_one_step(model, x_prev, t, cond=None):
    z = torch.randn_like(x_prev) if t > 1 else torch.zeros_like(x_prev)
    alpha = model.var_sched.alphas[t]
    alpha_bar = model.var_sched.alpha_bars[t]
    sigma = model.var_sched.get_sigmas(t, flexibility=0.0)
    c0 = 1.0 / torch.sqrt(alpha)
    c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
    beta = model.var_sched.betas[[t] * x_prev.size(0)]
    if cond is None:
        e_theta = model.net(x_prev, beta=beta)
    else:
        device = model.device
        cond = jutils.nputil.np2th(cond).to(device)
        e_theta = model.net(x_prev, beta=beta, context=cond)
    x_next = c0 * (x_prev - c1 * e_theta) + sigma * z
    return x_next

def tweedie_approach(model, x_noisy, t, cond=None):
    """
    Input:
        x_noisy: np.ndarray or torch.Tensor of shape [G,D] or [B,G,D]
    Output:
        x_denoised: torch.Tensor of shape [1,G,D] or [B,G,D]
    """
    device = model.device
    x_noisy = jutils.nputil.np2th(x_noisy).to(device)
    if x_noisy.ndim == 2:
        x_noisy = x_noisy.unsqueeze(0)
    
    alpha_bar = model.var_sched.alpha_bars[t].view(-1,1,1)
    beta = model.var_sched.betas[t].view(-1)
    if cond is None:
        e_theta = model.net(x_noisy, beta=beta)
    else:
        device = model.device
        cond = jutils.nputil.np2th(cond).to(device)
        if beta.shape[0] == 1:
            beta = beta.expand(cond.shape[0])
        e_theta = model.net(x_noisy, beta=beta, context=cond)

    score = -1 / torch.sqrt(1-alpha_bar) * e_theta
    
    x_denoised = (x_noisy + (1 - alpha_bar) * score) / torch.sqrt(alpha_bar)
    return x_denoised


def complete_shape(model, gmms, part_indices, start_t, method="sdedit", verbose=True):
    """
    Performs shape completion given incomplete sets of GMMs.
    Args:
        model:
        gmms:
        part_indices:
        start_t:
        method:
        verbose:
    """
    assert start_t >= 1 and start_t <= model.var_sched.num_steps
    assert method in ("mcg", "repaint", "sdedit")
    assert gmms.ndim == 3, f"Got ‘gmms’ of dimension {gmms.ndim}"
    timesteps = list(range(start_t, 0, -1))
    if verbose:
        timesteps = tqdm(timesteps)
    mask = torch.zeros(gmms.size(1)).to(gmms.device)
    mask[part_indices] = 1.0
    x0 = gmms
    ####
    # x = model.diffusion.add_noise(x0, start_t)[0]
    x = add_noise(model, x0, start_t)
    ####
    # attns = {}
    for t in timesteps:
        if method == "mcg":
            x = x.requires_grad_()  # record gradient
            # x_, attn = denoise_one_step(model, x, t)
            x_ = denoise_one_step(model, x, t)
        else:
            with torch.no_grad():
                # x_, attn = denoise_one_step(model, x, t)
                x_ = denoise_one_step(model, x, t)
        # attns[t] = attn
        if method == "mcg":
            raise NotImplementedError()
        elif method == "repaint":
            raise NotImplementedError()
        elif method == "sdedit":
            ####
            # x = model.diffusion.add_noise(x0, t - 1)[0]
            x = add_noise(model, x0, t - 1)
            ####
            x[:, mask == 1.0] = x_[:, mask == 1.0]
        else:
            raise NotImplementedError()
    ####
    # edited_part = x.clone()
    # edited_part[:, mask != 1.0] = 0.0
    # return x, edited_part, # attns
    # assert x.shape == gmms.shape
    return x
    ####


def run_sdedit(model, gmm, part_indices, start_t: int):
    """
    part_indices: unmasked part indices.
    """
    assert start_t >= 1 and start_t <= model.var_sched.num_steps
    mask = torch.zeros(gmm.size(1)).to(gmm.device)
    mask[part_indices] = 1.0
    x0 = gmm.to(model.device)
    x = model.add_noise(x0, start_t)[0]
    for t in range(start_t, 0, -1):
        z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
        alpha = model.var_sched.alphas[t]
        alpha_bar = model.var_sched.alpha_bars[t]
        sigma = model.var_sched.get_sigmas(t, flexibility=0.0)
        c0 = 1.0 / torch.sqrt(alpha)
        c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
        beta = model.var_sched.betas[[t] * x.size(0)]
        # e_theta = model.net(x, beta=beta, context=None)
        e_theta = model.net(x, beta=beta)
        x_ = c0 * (x - c1 * e_theta) + sigma * z
        x = model.add_noise(x0, t - 1)[0]
        x[:, mask == 1.0] = x_[:, mask == 1.0]
    edited_part = x.clone()
    edited_part[:, mask != 1.0] = 0.0
    return x

def gaus_denoising_tweedie(model, gaus, t, mean=None, std=None):
    """
    gaus: np.ndarray or torch.Tensor of shape [B,G,D]
    mean, std: np.ndarray or torch.Tensor
    t: int or list of size B
    """
    device = model.device
    gaus = jutils.nputil.np2th(gaus).to(device)
    if gaus.ndim == 2:
        gaus = gaus.unsqueeze(0)
    from pvd.diffusion.phase1_sym import GaussianSymSALDM
    if gaus.shape[-2] == 16 and isinstance(model, GaussianSymSALDM):
        gaus_half = gaus[...,:8, :]
        if model.hparams.get("global_normalization") == "partial":
            gaus_half = normalize_global_static_with_mean_std(gaus_half, mean, std, slice(12,None))

        affine = torch.eye(3).to(gaus)
        affine[0,0] = -1

        mu, p, phi, eigen = torch.split(gaus_half, [3,9,1,3], dim=2)
        affine = affine.unsqueeze(0).expand(mu.size(0), *affine.shape)

        B, G, _ = mu.shape
        p = p.reshape(B,G,3,3)

        mu_r = torch.einsum("bad, bnd->bna", affine, mu)
        p_r = torch.einsum("bad, bncd->bnca", affine, p)
        p_r = p_r.reshape(B,G,-1)
        
        gaus_half_t = torch.cat([mu_r, p_r, phi, eigen], dim=2)
        # print(gaus_half.shape, gaus_half_t.shape, gaus_half[:,:,0][None].shape)
        gaus_select_condition = (gaus_half[:,:,0].unsqueeze(-1) > 0)
        gaus_half = torch.where(gaus_select_condition, gaus_half, gaus_half_t)
        
        # gaus_noisy_half = add_noise(model, gaus_half, t)
        gaus_noisy_half = model.add_noise(gaus_half, t)[0]
        gaus_tweedie_half = tweedie_approach(model, gaus_noisy_half, t)
        if model.hparams.get("global_normalization") == "partial":
            gaus_tweedie_half = unnormalize_global_static_with_mean_std(gaus_tweedie_half, mean, std, slice(12,None))

        gaus_tweedie_half = project_eigenvectors(clip_eigenvalues(gaus_tweedie_half))
        gaus_tweedie = reflect_and_concat_gmms(gaus_tweedie_half).to(device) # 0,1,2,7,
        gaus_tweedie_t = gaus_tweedie[:,[8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7],:]
        gaus_select_condition2 = gaus_select_condition.repeat(1,2,1)
        gaus_tweedie = torch.where(gaus_select_condition2, gaus_tweedie, gaus_tweedie_t).to(device)

        return gaus_tweedie
    else:
        if model.hparams.get("global_normalization") == "partial":
            gaus = normalize_global_static_with_mean_std(gaus, mean, std, slice(12,None))
        elif model.hparams.get("global_normalization") == "all":
            gaus = normalize_global_static_with_mean_std(gaus, mean, std, slice(None))

        gaus_noisy = model.add_noise(gaus, t)[0]
        gaus_tweedie = tweedie_approach(model, gaus_noisy, t)
        if model.hparams.get("global_normalization") == "partial":
            gaus_tweedie = unnormalize_global_static_with_mean_std(gaus_tweedie, mean, std, slice(12,None))
        elif model.hparams.get("global_normalization") == "all":
            gaus_tweedie = unnormalize_global_static_with_mean_std(gaus_tweedie, mean, std, slice(None))
        
        gaus_tweedie = project_eigenvectors(clip_eigenvalues(gaus_tweedie))
        return jutils.nputil.np2th(gaus_tweedie).to(device)
        # raise NotImplementedError()
    
    x_noisy = add_noise(model, x, t)
    x_denoised = tweedie_approach(model, x_noisy, t)

    return x_denoised

def normalize_global_static_with_mean_std(data, mean, std, normalize_indices=slice(None)):
    """
    Input: 
        torch.Tensor. [16,16] or [B,16,16]
        mean, std: np.ndarray or torch.Tensor
        slice(None) -> full
        slice(12, None) -> partial
    Output: 
        [16,16] or [B,16,16]
    """
    assert normalize_indices == slice(None) or normalize_indices == slice(12,None), print(f"{normalize_indices} is wrong.")
    # data = jutils.thutil.th2np(data)
    mean, std = list(map(lambda x: jutils.nputil.np2th(x).to(data), [mean, std]))
    data[...,normalize_indices] = (data[...,normalize_indices] - mean[normalize_indices]) / std[normalize_indices]
    return data

def unnormalize_global_static_with_mean_std(data, mean, std,  unnormalize_indices=slice(None)):
    """
    Input: 
        torch.Tensor. [16,16] or [B,16,16]
        mean, std: np.ndarray or torch.Tensor
        slice(None) -> full
        slice(12, None) -> partial
    Output: 
        [16,16] or [B,16,16]
    """
    assert unnormalize_indices == slice(None) or unnormalize_indices == slice(12,None), print(f"{unnormalize_indices} is wrong.")
    # data = jutils.thutil.th2np(data)
    mean, std = list(map(lambda x: jutils.nputil.np2th(x).to(data), [mean, std]))
    data[...,unnormalize_indices] = (data[...,unnormalize_indices]) * std[unnormalize_indices] + mean[unnormalize_indices]
    return data

