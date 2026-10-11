"""tau is floored at 1e-4 * std(z), with std taken per row along `dim`."""
import pytest
import torch

import softk
from _reference import assert_close, ref_hard, ref_mask, ref_soft

K = 3
FLOOR = 1e-4


def _row(scale):
    """16 scores whose 3rd and 4th largest differ by about FLOOR * std, so the floored tau
    leaves the boundary pair visibly soft while an unfloored tiny tau would not."""
    row = 0.5 * (torch.arange(16, dtype=torch.float64) - 7.5)
    row[-4] = row[-3] - FLOOR * row.std()
    perm = torch.randperm(16, generator=torch.Generator().manual_seed(0))
    return scale * row[perm]


def _rows():
    return torch.stack([_row(1.0), _row(1e3)])


def _expected(fn, z):
    return torch.stack([fn(r, K, FLOOR * r.std().item()) for r in z])


def test_the_construction_makes_the_floor_observable():
    z = _rows()
    assert (_expected(ref_soft, z) - ref_hard(z, K)).abs().max().item() > 1e-2


# 2e-3 leaves room for either std convention (n or n - 1 in the denominator), which moves p by
# about 2e-4 here; ignoring the floor moves p by more than 1e-2.
@pytest.mark.parametrize("tau", [1e-9, 1e-6])
def test_softmax_uses_the_per_row_floor(tau):
    z = _rows()
    assert_close(softk.soft_topk_softmax(z, K, tau), _expected(ref_soft, z), 2e-3)


@pytest.mark.parametrize("tau", [1e-9, 1e-6])
def test_mask_uses_the_per_row_floor(tau):
    z = _rows()
    assert_close(softk.soft_topk_mask(z, K, tau), _expected(ref_mask, z), 5e-3)


def test_floor_follows_dim():
    z = _rows().T.contiguous()  # rows now run along dim 0
    assert_close(softk.soft_topk_softmax(z, K, 1e-9, dim=0), _expected(ref_soft, _rows()).T, 2e-3)


def test_tau_above_the_floor_is_used_unchanged():
    z = _rows()[:1]
    assert_close(softk.soft_topk_softmax(z, K, 0.05), ref_soft(z, K, 0.05), 1e-9)
