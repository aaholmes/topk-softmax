import torch

from . import _triton
from ._autograd import _LogSoftTopKMask

_MODES = ("soft", "hard", "st")
_BACKENDS = ("auto", "eager", "triton")
_HALF_DTYPES = (torch.float16, torch.bfloat16)
_TAU_FLOOR = 1e-4  # in units of the per-row standard deviation of z


def _validate(k, tau):
    if not k > 0:
        raise ValueError(f"k must be positive, got {k}")
    if not tau > 0:
        raise ValueError(f"tau must be positive, got {tau}")


def _default_iters(z):
    # Bisection steps before the two Newton steps. Measured in benchmarks/solvers.py: these counts
    # reach the accuracy of 32 and 60 steps on every input tried.
    return 24 if z.dtype == torch.float64 else 20


def _use_triton(z, backend):
    """Whether to run the fused GPU kernels on z, which is already in 32- or 64-bit floats."""
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    if backend == "eager":
        return False
    if backend == "triton":
        if not z.is_cuda:
            raise ValueError('backend="triton" needs a CUDA tensor')
        if not _triton.available():
            raise ValueError('backend="triton" needs the triton package and torch 2.6 or newer')
    elif not (z.is_cuda and _triton.available() and _triton.faster_for(z.shape[-1])):
        return False
    # the kernels are 32-bit; 64-bit inputs keep their accuracy on the eager path
    return z.dtype == torch.float32 and z.numel() > 0


def _log_mask(z, k, tau, iters):
    """log m along the last dim, for z in 32-bit or 64-bit floating point."""
    if iters is None:
        iters = _default_iters(z)
    # Below about 1e-4 row standard deviations, (z - t)/tau can no longer be resolved, so tau is
    # floored there. The floor is a constant in the backward pass.
    floor = _TAU_FLOOR * z.detach().std(-1, keepdim=True, correction=0)
    return _LogSoftTopKMask.apply(z, k, floor.clamp_min(tau), iters)


def _hard(z, k):
    idx = z.topk(int(k), -1).indices
    return torch.zeros_like(z).scatter(-1, idx, torch.softmax(z.gather(-1, idx), -1))


def soft_topk_softmax(z, k, tau, dim=-1, mode="soft", iters=None, backend="auto"):
    """Softmax of z along `dim`, restricted to a soft top-k with temperature tau.

    With m_i = sigmoid((z_i - t)/tau) and t chosen so that sum m = k, the result is
    softmax(z + log m). Large tau gives softmax(z); tau -> 0 gives a softmax over the k largest
    entries. k may be any positive real number, and k >= n returns softmax(z).

    tau is in the units of z and is floored at 1e-4 times the standard deviation of z along `dim`.

    mode:
      "soft"  the formula above.
      "hard"  softmax over the k largest entries, with exact zeros elsewhere. tau is ignored.
      "st"    straight-through: the "hard" output, with the gradient of "soft".
    "hard" and "st" need an integer k.

    16-bit inputs are computed in 32-bit floating point and cast back.

    backend:
      "eager"   plain PyTorch operations, on any device.
      "triton"  fused GPU kernels for CUDA tensors, which stop iterating on each row once it has
                converged. Needs the triton package. 64-bit inputs and mode="hard" still use the
                PyTorch operations.
      "auto"    "triton" where it is available and measured to be faster, otherwise "eager".
    """
    _validate(k, tau)
    return _soft_topk_softmax(z, k, tau, dim, mode, iters, backend)


def _soft_topk_softmax(z, k, tau, dim, mode, iters=None, backend="auto"):
    """soft_topk_softmax without the check on tau, so tau may be a tensor under torch.compile."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if mode != "soft" and int(k) != k:
        raise ValueError(f"mode={mode!r} needs an integer k, got {k}")
    x = z.float() if z.dtype in _HALF_DTYPES else z
    use_triton = _use_triton(x, backend)
    if k >= z.shape[dim]:
        return torch.softmax(z, dim)

    x = x.transpose(dim, -1)
    if mode == "hard":
        out = _hard(x, k)
    else:
        if use_triton:
            out = _triton.run(x, k, tau, _default_iters(x) if iters is None else iters, want_mask=False)
        else:
            out = torch.softmax(x + _log_mask(x, k, tau, iters), -1)
        if mode == "st":
            # the parenthesized term is exactly zero, so the forward value is the hard output
            out = _hard(x, k).detach() + (out - out.detach())
    return out.transpose(dim, -1).to(z.dtype)


def soft_topk_mask(z, k, tau, dim=-1, iters=None, backend="auto"):
    """The mask m alone: entries in [0, 1] that sum to k along `dim`. Needs k < n.

    `backend` is as in soft_topk_softmax.
    """
    _validate(k, tau)
    if k >= z.shape[dim]:
        raise ValueError(f"k must be less than the size of dim {dim} ({z.shape[dim]}), got {k}")
    x = z.float() if z.dtype in _HALF_DTYPES else z
    use_triton = _use_triton(x, backend)
    x = x.transpose(dim, -1)
    if use_triton:
        m = _triton.run(x, k, tau, _default_iters(x) if iters is None else iters, want_mask=True)
    else:
        m = _log_mask(x, k, tau, iters).exp()
    return m.transpose(dim, -1).to(z.dtype)
