import math

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


def _mask_scores(z, k, flat_top):
    """The scores the mask is computed from: z, or with flat_top, z clamped from above at its
    floor(k)-th largest entry, so that the floor(k) largest entries get one mask value."""
    kf = math.floor(k)
    if not flat_top or kf < 1:
        return z
    return torch.minimum(z, z.topk(kf, -1).values[..., -1:])


def _hard(z, k):
    idx = z.topk(int(k), -1).indices
    return torch.zeros_like(z).scatter(-1, idx, torch.softmax(z.gather(-1, idx), -1))


def soft_topk_softmax(z, k, tau, dim=-1, mode="soft", iters=None, backend="auto", flat_top=True):
    """Softmax of z along `dim`, restricted to a soft top-k with temperature tau.

    With m_i = sigmoid((min(z_i, c) - t)/tau), c the floor(k)-th largest entry and t chosen so
    that sum m = k, the result is softmax(z + log m). Large tau gives softmax(z); tau -> 0 gives a
    softmax over the k largest entries. k may be any positive real number, and k >= n returns
    softmax(z).

    tau is in the units of z and is floored at 1e-4 times the standard deviation of min(z, c)
    along `dim`.

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

    flat_top: the clamp at c gives the floor(k) largest entries one mask value, so they keep
    exactly the ratios a softmax gives them, and each of their probabilities rises monotonically
    as tau decreases. flat_top=False computes the mask from z itself, m_i = sigmoid((z_i - t)/tau).
    That skips finding c and is 1.5 to 2 times faster, but at intermediate tau it raises the
    largest of the top k beyond their softmax ratios, so their probabilities are not monotonic in
    tau. The Triton kernels find c themselves; the PyTorch code calls torch.topk.
    """
    _validate(k, tau)
    return _soft_topk_softmax(z, k, tau, dim, mode, iters, backend, flat_top)


def _soft_topk_softmax(z, k, tau, dim, mode, iters=None, backend="auto", flat_top=True):
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
        if use_triton:  # the kernel clamps the scores itself for flat_top
            out = _triton.run(x, k, tau, _default_iters(x) if iters is None else iters, False, flat_top)
        else:
            out = torch.softmax(x + _log_mask(_mask_scores(x, k, flat_top), k, tau, iters), -1)
        if mode == "st":
            # the parenthesized term is exactly zero, so the forward value is the hard output
            out = _hard(x, k).detach() + (out - out.detach())
    return out.transpose(dim, -1).to(z.dtype)


def soft_topk_mask(z, k, tau, dim=-1, iters=None, backend="auto", flat_top=True):
    """The mask m alone: entries in [0, 1] that sum to k along `dim`. Needs k < n.

    `backend` and `flat_top` are as in soft_topk_softmax.
    """
    _validate(k, tau)
    if k >= z.shape[dim]:
        raise ValueError(f"k must be less than the size of dim {dim} ({z.shape[dim]}), got {k}")
    x = z.float() if z.dtype in _HALF_DTYPES else z
    use_triton = _use_triton(x, backend)
    x = x.transpose(dim, -1)
    if use_triton:
        m = _triton.run(x, k, tau, _default_iters(x) if iters is None else iters, True, flat_top)
    else:
        m = _log_mask(_mask_scores(x, k, flat_top), k, tau, iters).exp()
    return m.transpose(dim, -1).to(z.dtype)
