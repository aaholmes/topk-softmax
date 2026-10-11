import pytest
import torch

import softk
from _reference import assert_close, randn, ref_mask

TAUS = [10.0, 1.0, 0.01]
KS = [1, 3, 2.5, 9]
SUM_TOL = {torch.float32: 1e-5, torch.float64: 1e-10}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("k", KS)
def test_mask_sums_to_k(dtype, tau, k):
    m = softk.soft_topk_mask(randn(4, 10, dtype=dtype), k, tau)
    assert_close(m.sum(-1), torch.full((4,), float(k), dtype=torch.float64), SUM_TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mask_sums_to_k_n1000(dtype):
    m = softk.soft_topk_mask(randn(2, 1000, dtype=dtype), 50, 0.1)
    assert_close(m.sum(-1) / 50, torch.ones(2), SUM_TOL[dtype])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tau", TAUS)
def test_mask_entries_lie_in_unit_interval(dtype, tau):
    m = softk.soft_topk_mask(randn(4, 10, dtype=dtype), 3, tau)
    assert torch.isfinite(m).all()
    assert (m >= 0).all() and (m <= 1).all()


@pytest.mark.parametrize("tau", TAUS)
def test_mask_preserves_the_order_of_z(tau):
    z = randn(4, 10)
    m = softk.soft_topk_mask(z, 3, tau)
    m_sorted = m.gather(-1, z.argsort(-1))
    assert (m_sorted.diff(dim=-1) >= -1e-12).all()


@pytest.mark.parametrize("tau", TAUS + [0.1])
@pytest.mark.parametrize("k", KS)
def test_mask_matches_reference(tau, k):
    z = randn(4, 10)
    assert_close(softk.soft_topk_mask(z, k, tau), ref_mask(z, k, tau), 1e-9)


def test_mask_does_not_require_grad_for_plain_input():
    assert not softk.soft_topk_mask(randn(4, 10), 3, 1.0).requires_grad
