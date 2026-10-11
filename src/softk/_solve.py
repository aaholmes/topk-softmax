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
