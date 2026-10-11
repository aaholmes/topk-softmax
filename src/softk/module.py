import torch

from . import stats as _stats
from .functional import _BACKENDS, _HALF_DTYPES, _MODES, _soft_topk_softmax, _validate, soft_topk_mask


class SoftTopKSoftmax(torch.nn.Module):
    """soft_topk_softmax as a module, with tau held in a buffer.

    tau is saved in the state dict, and a TauScheduler writes into it during training. `mode` is a
    plain attribute ("soft", "hard" or "st"); eval() does not change it, so set mode="hard" for
    inference if exact zeros are wanted.

    With track_stats=True, each forward pass stores four scalars in `self.stats`, computed without
    gradient: the mean and 99th percentile over rows of the mass outside the top k, the mean
    effective support exp(entropy), and the mean of sum_i m_i (1 - m_i), which goes to zero as the
    gradient through the mask vanishes. This costs a second solve for the mask.
    """

    def __init__(self, k, dim=-1, tau=1.0, mode="soft", track_stats=False, backend="auto"):
        super().__init__()
        _validate(k, tau)
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if mode != "soft" and int(k) != k:
            raise ValueError(f"mode={mode!r} needs an integer k, got {k}")
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
        self.k = k
        self.backend = backend
        self.dim = dim
        self.mode = mode
        self.track_stats = track_stats
        self.stats = {}
        self.register_buffer("tau", torch.tensor(float(tau)))

    def forward(self, z):
        p = _soft_topk_softmax(z, self.k, self.tau, self.dim, self.mode, backend=self.backend)
        if self.track_stats:
            self._record(z, p)
        return p

    @torch.no_grad()
    def _record(self, z, p):
        if z.dtype in _HALF_DTYPES:  # torch.quantile needs 32-bit or 64-bit floats
            z, p = z.float(), p.float()
        mass = _stats.mass_outside_topk(p, self.k, self.dim)
        if self.k < z.shape[self.dim]:
            m = soft_topk_mask(z, self.k, self.tau, self.dim, backend=self.backend)
            sum_v = (m * (1 - m)).sum(self.dim).mean()
        else:
            sum_v = z.new_zeros(())
        self.stats = {
            "mass_outside_mean": mass.mean(),
            "mass_outside_p99": torch.quantile(mass.flatten(), 0.99),
            "effective_support": _stats.effective_support(p, self.dim).mean(),
            "sum_v": sum_v,
        }

    def extra_repr(self):
        return f"k={self.k}, dim={self.dim}, mode={self.mode!r}, track_stats={self.track_stats}"
