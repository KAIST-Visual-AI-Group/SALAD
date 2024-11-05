import torch
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

class VarianceSchedule(Module):
    def __init__(self, num_steps, beta_1, beta_T, mode="linear"):
        super().__init__()
        # assert mode in ("linear",)
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == "quad":
            betas = torch.linspace(beta_1 ** 0.5, beta_T ** 0.5, num_steps) ** 2
        elif mode == "cosine":
            cosine_s = 8e-3
            timesteps = torch.arange(num_steps + 1) / num_steps + cosine_s
            alphas = timesteps / (1 + cosine_s) * np.pi / 2
            alphas = torch.cos(alphas).pow(2)
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)  # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[
                i
            ]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sigmas_flex", sigmas_flex)
        self.register_buffer("sigmas_inflex", sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (
            1 - flexibility
        )
        return sigmas


def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1.0 + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        assert ctx.dim() == x.dim()
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


# class ModulatedCrossTransformerLayer(Module):
    # def __init__(
        # self,
        # dim_x: int,
        # dim_y: int,
        # dim_out: int,
        # dim_ctx: int,
        # n_head: int,
        # dropout: float,
    # ) -> None:
        # super().__init__()
        # self._cross_attn = CrossTransformerLayer(
            # dim_x, dim_y, dim_out, num_heads=n_head, dropout=dropout
        # )
        # self._bias = Linear(dim_ctx, dim_out, bias=False)
        # self._gate = Linear(dim_ctx, dim_out)

        # self.dim_x = dim_x
        # self.dim_y = dim_y
        # self.dim_out = dim_out
        # self.dim_ctx = dim_ctx
        # self.n_head = n_head
        # self.dropout = dropout

    # def forward(self, ctx: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # gate = torch.sigmoid(self._gate(ctx))
        # bias = self._bias(ctx)
        # ret = self._cross_attn(x, y) * gate + bias
        # return ret


# class ModulatedTransformerLayer(Module):
    # def __init__(
        # self,
        # dim_in: int,
        # dim_out: int,
        # dim_ctx: int,
        # n_head: int,
        # dropout: float,
    # ) -> None:
        # super().__init__()
        # self._attn = TransformerLayer(
            # dim_in, dim_ref=dim_in, dim_out=dim_out, num_heads=n_head, dropout=dropout
        # )
        # self._bias = Linear(dim_ctx, dim_out, bias=False)
        # self._gate = Linear(dim_ctx, dim_out)

        # self.dim_in = dim_in
        # self.dim_out = dim_out
        # self.dim_ctx = dim_ctx

    # def forward(self, ctx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # gate = torch.sigmoid(self._gate(ctx))
        # bias = self._bias(ctx)
        # ret = self._attn(x) * gate + bias
        # return ret


# def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    # def lr_func(epoch):
        # if epoch <= start_epoch:
            # return 1.0
        # elif epoch <= end_epoch:
            # total = end_epoch - start_epoch
            # delta = epoch - start_epoch
            # frac = delta / total
            # return (1 - frac) * 1.0 + frac * (end_lr / start_lr)
        # else:
            # return end_lr / start_lr

    # return LambdaLR(optimizer, lr_lambda=lr_func)


# def lr_func(epoch):
    # if epoch <= start_epoch:
        # return 1.0
    # elif epoch <= end_epoch:
        # total = end_epoch - start_epoch
        # delta = epoch - start_epoch
        # frac = delta / total
        # return (1 - frac) * 1.0 + frac * (end_lr / start_lr)
    # else:
        # return end_lr / start_lr


# def dot(x, y, dim=2):
    # return torch.sum(x * y, dim=dim)


# def remove_projection(v_1, v_2):
    # """
    # outputs the vector orthogonal to v_2
    # """
    # proj = dot(v_1, v_2) / dot(v_2, v_2)
    # return v_1 - proj[:, :, None] * v_2


# def get_orthonormal_bases_svd(vs):
    # """
    # Implements the solution for the Orthogonal Procrustes problem,
    # which projects a matrix to the closest rotation matrix using SVD.

    # Args:
        # vs: Tensor of shape (B, M, 9)
    # Returns:
        # p: Tensor of shape (B, M, 9).
    # """
    # # Compute SVDs of matrices in batch
    # b, m, _ = vs.shape
    # vs_ = vs.reshape(b * m, 3, 3)
    # U, _, Vh = torch.linalg.svd(vs_)

    # # Determine the diagonal matrix to make determinants 1
    # sigma = torch.eye(3)[None, ...].repeat(b * m, 1, 1).to(vs_.device)
    # det = torch.linalg.det(torch.bmm(U, Vh))  # Compute determinants of UVT
    # sigma[:, 2, 2] = det

    # # Construct orthogonal matrices
    # p = torch.bmm(torch.bmm(U, sigma), Vh)
    # return p.reshape(b, m, 9)


# def get_orthonormal_bases(vs):
    # """
    # Implements Gram-Schmidt algorithm to find the orthonormal bases
    # given a set of possibly linearly dependent vectors.

    # Args:
        # vs: Tensor of shape (B, M, 9)
    # Returns:
        # p: Tensor of shape (B, M, 9)
    # """
    # b, m, _ = vs.shape
    # vs_ = vs.reshape(b, m, 3, 3)

    # # Iterate over each column of 'U'.
    # raw_base = []
    # for i in range(vs_.shape[3]):
        # u = vs_[..., i]  # (B, M, 3)
        # for j in range(i):
            # u = remove_projection(u, raw_base[j])
        # raw_base.append(u)
    # p = torch.stack(raw_base, dim=3)  # (B, M, 3, 3)

    # # Normalize each column to be a unit vector.
    # p = p / torch.norm(p, p=2, dim=2)[:, :, None, :]  # (B, M, 3, 3)
    # p = p.reshape(b, m, 9)
    # return p


# if __name__ == "__main__":
    # vs = torch.randn(3, 1, 9)
    # p = get_orthonormal_bases(vs).reshape(3, 1, 3, 3)
    # p_ = get_orthonormal_bases_svd(vs).reshape(3, 1, 3, 3)

    # print(dot(p[..., 0], p[..., 1]))
    # print(dot(p[..., 0], p[..., 2]))
    # print(dot(p[..., 1], p[..., 2]))

    # print(torch.linalg.det(p_))

    # print(dot(p_[..., 0], p_[..., 1]))
    # print(dot(p_[..., 0], p_[..., 2]))
    # print(dot(p_[..., 1], p_[..., 2]))
