"""Slow double-precision (fp64) references and input builders for the tests.

These are written independently of the library: one row at a time, a wide bisection bracket, and
the product softmax(z) * m normalized in probability space instead of log space.
"""
import torch


def randn(*shape, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float64).to(dtype)


def separated(rows, n, gap=0.4, dtype=torch.float64, seed=0):
    """Rows that are random permutations of an evenly spaced grid, so no two entries tie."""
    g = torch.Generator().manual_seed(seed)
    grid = gap * (torch.arange(n, dtype=torch.float64) - (n - 1) / 2)
    return torch.stack([grid[torch.randperm(n, generator=g)] for _ in range(rows)]).to(dtype)


def assert_close(actual, expected, atol):
    assert actual.shape == expected.shape, f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    if actual.numel() == 0:
        return
    err = (actual.double() - expected.double()).abs().max().item()
    assert err <= atol, f"max abs error {err:.3e} > {atol:.1e}"


def _threshold_row(row, k, tau):
    lo = row.min().item() - 800.0 * tau  # every sigmoid is exactly 1 here, so the sum is n > k
    hi = row.max().item() + 800.0 * tau  # every sigmoid is exactly 0 here
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        if torch.sigmoid((row - mid) / tau).sum().item() > k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _mask_row(row, k, tau):
    return torch.sigmoid((row - _threshold_row(row, k, tau)) / tau)


def _soft_row(row, k, tau):
    if k >= row.numel():
        return torch.softmax(row, -1)
    w = torch.softmax(row, -1) * _mask_row(row, k, tau)
    return w / w.sum()


def _hard_row(row, k):
    if k >= row.numel():
        return torch.softmax(row, -1)
    idx = row.topk(int(k)).indices
    out = torch.zeros_like(row)
    out[idx] = torch.softmax(row[idx], -1)
    return out


def _per_row(fn, z, dim):
    moved = z.detach().double().movedim(dim, -1)
    rows = moved.reshape(-1, moved.shape[-1])
    if rows.shape[0] == 0:
        return torch.zeros_like(moved).movedim(-1, dim)
    return torch.stack([fn(r) for r in rows]).reshape(moved.shape).movedim(-1, dim)


def ref_threshold(z, k, tau):
    """Thresholds for the last dim, with that dim kept as size 1."""
    rows = z.detach().double().reshape(-1, z.shape[-1])
    t = torch.tensor([_threshold_row(r, k, tau) for r in rows], dtype=torch.float64)
    return t.reshape(*z.shape[:-1], 1)


def ref_mask(z, k, tau, dim=-1):
    return _per_row(lambda r: _mask_row(r, k, tau), z, dim)


def ref_soft(z, k, tau, dim=-1):
    return _per_row(lambda r: _soft_row(r, k, tau), z, dim)


def ref_hard(z, k, dim=-1):
    return _per_row(lambda r: _hard_row(r, k), z, dim)


def hard_cases(dtype=torch.float64):
    """(name, z, k, tau) inputs on which a root-finder for the threshold is most likely to fail."""
    tie = torch.tensor([[3.0, 2.0, 1.0, 1.0, 1.0, 0.0, -1.0, -2.0]])
    spike = torch.tensor([[40.0, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]])
    clusters = torch.tensor([[50.0, 50.1, 49.9, -50.0, -50.1, -49.9, -50.2]])
    cases = []
    cases += [(f"tie tau={tau:g}", tie, 3, tau) for tau in (1.0, 1e-3, 1e-6)]
    cases += [(f"constant row tau={tau:g}", torch.ones(2, 16), 5, tau) for tau in (1.0, 1e-3)]
    cases += [(f"spike k={k} tau={tau:g}", spike, k, tau) for k in (1, 3) for tau in (1.0, 0.01)]
    cases += [(f"two clusters tau={tau:g}", clusters, 3, tau) for tau in (1.0, 1e-3)]
    cases += [(f"n=1000 k={k} tau={tau:g}", randn(4, 1000), k, tau)
              for k in (1, 100, 999) for tau in (1e-4, 1e-2, 1.0, 1e2, 1e4)]
    cases += [("large scores", 1e4 * randn(4, 64), 5, 1.0)]
    cases += [(f"real k tau={tau:g}", randn(4, 7), 2.5, tau) for tau in (0.1, 1.0)]
    wide = randn(2, 32768)
    outlier = wide.clone()
    outlier[:, 0] = 1e3
    cases += [(f"n=32768 k={k} tau={tau:g}", wide, k, tau) for k in (8, 16384) for tau in (1e-6, 1e-2)]
    cases += [(f"n=32768 with an outlier tau={tau:g}", outlier, 8, tau) for tau in (1e-6, 1e-2)]
    return [(name, z.to(dtype), k, tau) for name, z, k, tau in cases]


def mask_sum_tol(k, dtype):
    """Allowed |sum(m) - k|. A 32-bit sum of terms that add up to k is accurate to about k * 1e-7."""
    return 1e-10 if dtype == torch.float64 else max(1e-5, 2e-7 * k)
