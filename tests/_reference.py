"""Slow fp64 references and input builders for the tests.

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
