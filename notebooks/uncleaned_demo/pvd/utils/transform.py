import torch


def apply_gmm_affine(gmms, affine):
    """
    Applies the given Affine transform to Gaussians in the GMM.

    Args:
        gmms: (B, M, 16). A batch of GMMs
        affine: (3, 3) or (B, 3, 3). A batch of Affine matrices
    """
    mu, p, phi, eigen = torch.split(gmms, [3, 9, 1, 3], dim=2)
    if affine.ndim == 2:
        affine = affine.unsqueeze(0).expand(mu.size(0), *affine.shape)

    bs, n_part, _ = mu.shape
    p = p.reshape(bs, n_part, 3, 3)

    mu_r = torch.einsum("bad, bnd -> bna", affine, mu)
    p_r = torch.einsum("bad, bncd -> bnca", affine, p)
    p_r = p_r.reshape(bs, n_part, -1)
    gmms_t = torch.cat([mu_r, p_r, phi, eigen], dim=2)
    assert gmms.shape == gmms_t.shape, "Input and output shapes must be the same"
    
    return gmms_t