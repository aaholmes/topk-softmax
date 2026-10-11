import torch

from ._solve import _solve_threshold

_NEWTON_STEPS = 2


class _LogSoftTopKMask(torch.autograd.Function):
    """Returns log m along the last dim, with m_i = sigmoid((z_i - t(z))/tau) and sum m = k.

    tau is a tensor that broadcasts over rows. The backward pass uses the implicit function
    theorem on sum m = k and does not differentiate through the solver or through tau.
    """

    @staticmethod
    def forward(ctx, z, k, tau, iters):
        with torch.no_grad():
            # m does not change when a constant is added to z; centering keeps t small, so
            # (z - t)/tau loses less to rounding
            zc = z - z.mean(-1, keepdim=True)
            u = (zc - _solve_threshold(zc, k, tau, iters)) / tau
            # Newton steps on a shift s of u, with sum sigmoid(u - s) = k. Shifting u directly
            # keeps corrections that are smaller than the floating-point spacing of t.
            sig = torch.sigmoid(u)
            resid = sig.sum(-1, keepdim=True) - k
            for _ in range(_NEWTON_STEPS):
                v = sig * (1 - sig)
                u_new = u - resid / v.sum(-1, keepdim=True).clamp_min(1e-30)
                sig_new = torch.sigmoid(u_new)
                resid_new = sig_new.sum(-1, keepdim=True) - k
                # where sum m is nearly flat in t, sum v is tiny and a step can overshoot
                better = resid_new.abs() < resid.abs()
                u = torch.where(better, u_new, u)
                sig = torch.where(better, sig_new, sig)
                resid = torch.where(better, resid_new, resid)
        ctx.save_for_backward(sig, tau)
        return torch.nn.functional.logsigmoid(u)

    @staticmethod
    def backward(ctx, g):
        sig, tau = ctx.saved_tensors
        v = sig * (1 - sig)
        a = (1 - sig) * g  # d log m_i / d u_i = 1 - sigma_i
        # dt/dz_j = v_j / sum v ;  d log m_i/dz_j = (1-sig_i)/tau * (delta_ij - v_j/sum v)
        corr = v * a.sum(-1, keepdim=True) / v.sum(-1, keepdim=True).clamp_min(1e-30)
        return (a - corr) / tau, None, None, None
