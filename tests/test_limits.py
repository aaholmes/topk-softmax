import pytest
import torch

import softk
from _reference import assert_close, randn, ref_hard, ref_soft, separated


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("k", [1, 3, 9])
def test_large_tau_is_softmax(dtype, k):
    z = randn(4, 10, dtype=dtype)
    assert_close(softk.soft_topk_softmax(z, k, 1e4), torch.softmax(z, -1), 1e-4)


@pytest.mark.parametrize("k", [1, 3, 9])
def test_small_tau_is_topk_softmax(k):
    z = separated(4, 10)  # adjacent scores differ by 0.4, far more than tau
    assert_close(softk.soft_topk_softmax(z, k, 1e-3), ref_hard(z, k), 1e-6)


@pytest.mark.parametrize("tau", [5.0, 1.0, 0.1, 0.01])
@pytest.mark.parametrize("k", [1, 3, 2.5, 9])
def test_intermediate_tau_matches_reference(tau, k):
    z = randn(4, 10)
    assert_close(softk.soft_topk_softmax(z, k, tau), ref_soft(z, k, tau), 1e-9)


@pytest.mark.parametrize("k", [10, 13, 10.0])
@pytest.mark.parametrize("dim", [-1, 0])
def test_k_at_least_n_returns_softmax_exactly(k, dim):
    z = randn(10, 10)
    assert torch.equal(softk.soft_topk_softmax(z, k, 0.5, dim=dim), torch.softmax(z, dim))


@pytest.mark.parametrize("tau", [10.0, 1.0, 0.01])
def test_constant_row_gives_uniform_output(tau):
    z = torch.full((2, 8), 1.7, dtype=torch.float64)
    assert_close(softk.soft_topk_mask(z, 3, tau), torch.full((2, 8), 3 / 8), 1e-12)
    assert_close(softk.soft_topk_softmax(z, 3, tau), torch.full((2, 8), 1 / 8), 1e-12)


def test_k_just_below_n_is_close_to_softmax():
    # the mask must sum to n - 1e-6, so every entry is within 1e-6 of 1
    z = randn(4, 10)
    assert_close(softk.soft_topk_softmax(z, 10 - 1e-6, 1.0), torch.softmax(z, -1), 1e-5)
