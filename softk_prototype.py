"""Prototype of a softmax restricted to a soft top-k by a sigmoid mask that sums to k.

The mask has the form used by the Limited Multi-Label Projection Layer (Amos, Koltun, Kolter 2019).

p = normalize(softmax(z) * m),  m_i = sigmoid((z_i - t)/tau),  sum_i m_i = k
Computed in log space: p = softmax(z + logsigmoid((z - t)/tau)).
"""
import math
import torch


def _solve_threshold(z, k, tau, iters):
    # sum_i sigmoid((z_i - t)/tau) is strictly decreasing in t.
    # Bracket: t_lo gives every m_i >= k/n (sum >= k); t_hi gives every m_i <= k/n (sum <= k).
    n = z.shape[-1]
    c = tau * math.log((k / n) / (1 - k / n))  # tau * logit(k/n)
    lo = z.amin(-1, keepdim=True) - c
    hi = z.amax(-1, keepdim=True) - c
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = torch.sigmoid((z - mid) / tau).sum(-1, keepdim=True)
        too_big = s > k  # threshold too low
        lo = torch.where(too_big, mid, lo)
        hi = torch.where(too_big, hi, mid)
    return 0.5 * (lo + hi)


class _LogSoftTopKMask(torch.autograd.Function):
    """Returns log m, with m_i = sigmoid((z_i - t(z))/tau), sum m = k. Implicit-function backward."""

    @staticmethod
    def forward(ctx, z, k, tau, iters):
        with torch.no_grad():
            t = _solve_threshold(z, k, tau, iters)
            u = (z - t) / tau
            # one Newton step to refine t
            sig = torch.sigmoid(u)
            v = sig * (1 - sig)
            t = t + tau * (sig.sum(-1, keepdim=True) - k) / v.sum(-1, keepdim=True).clamp_min(1e-30)
            u = (z - t) / tau
            sig = torch.sigmoid(u)
        ctx.save_for_backward(sig)
        ctx.tau = tau
        return torch.nn.functional.logsigmoid(u)

    @staticmethod
    def backward(ctx, g):
        (sig,) = ctx.saved_tensors
        tau = ctx.tau
        v = sig * (1 - sig)
        a = (1 - sig) * g  # d log m_i / d u_i = 1 - sigma_i
        # dt/dz_j = v_j / sum v ;  d log m_i/dz_j = (1-sig_i)/tau * (delta_ij - v_j/sum v)
        corr = v * a.sum(-1, keepdim=True) / v.sum(-1, keepdim=True).clamp_min(1e-30)
        return (a - corr) / tau, None, None, None


def soft_topk_softmax(z, k, tau, dim=-1, iters=None):
    z = z.transpose(dim, -1)
    n = z.shape[-1]
    if k >= n:
        out = torch.softmax(z, -1)
    else:
        if iters is None:
            iters = 60 if z.dtype == torch.float64 else 32
        logm = _LogSoftTopKMask.apply(z, k, tau, iters)
        out = torch.softmax(z + logm, -1)
    return out.transpose(dim, -1)


def soft_topk_mask(z, k, tau, iters=60):
    return _LogSoftTopKMask.apply(z, k, tau, iters).exp()


if __name__ == "__main__":
    torch.manual_seed(0)
    # 1. gradcheck of the custom backward
    for tau in [5.0, 1.0, 0.1]:
        z = torch.randn(3, 7, dtype=torch.float64, requires_grad=True)
        ok = torch.autograd.gradcheck(lambda x: soft_topk_softmax(x, 3, tau), (z,), eps=1e-6, atol=1e-5)
        print(f"gradcheck tau={tau}: {ok}")
    # 2. mask sums to k
    z = torch.randn(4, 10, dtype=torch.float64)
    for tau in [10.0, 1.0, 0.01]:
        print(f"tau={tau} mask sum:", soft_topk_mask(z, 3, tau).sum(-1).tolist())
    # 3. limits
    z = torch.randn(10, dtype=torch.float64)
    sm = torch.softmax(z, -1)
    p_hi = soft_topk_softmax(z, 3, 1e4)
    print("max|p(tau=1e4) - softmax| =", (p_hi - sm).abs().max().item())
    idx = z.topk(3).indices
    ref = torch.zeros_like(z); ref[idx] = torch.softmax(z[idx], -1)
    p_lo = soft_topk_softmax(z, 3, 1e-3)
    print("max|p(tau=1e-3) - top3 softmax| =", (p_lo - ref).abs().max().item())
    # 4. fp32 at small tau, gradient finite
    z = torch.randn(2, 1000, requires_grad=True)
    p = soft_topk_softmax(z, 8, 1e-3)
    (p * torch.randn_like(p)).sum().backward()
    print("fp32 tau=1e-3 grad finite:", torch.isfinite(z.grad).all().item(),
          "mass outside top-8:", (1 - p.topk(8, -1).values.sum(-1)).tolist())
