import pytest
import torch

import topk_softmax
from _reference import assert_close, randn, ref_hard, separated


def _forward_backward(z, k, tau):
    z = z.clone().requires_grad_()
    p = topk_softmax.soft_topk_softmax(z, k, tau)
    (p * randn(*z.shape, dtype=z.dtype, seed=1)).sum().backward()
    return p.detach(), z.grad


def _assert_finite(z, k, tau):
    p, grad = _forward_backward(z, k, tau)
    assert torch.isfinite(p).all()
    assert torch.isfinite(grad).all()
    assert_close(p.sum(-1), torch.ones(p.shape[:-1]), 1e-4)
    return p, grad


def test_fp32_small_tau_wide_row():
    _assert_finite(randn(2, 32768, dtype=torch.float32), 8, 1e-4)


def test_fp32_small_tau_n1000():
    _assert_finite(randn(2, 1000, dtype=torch.float32), 8, 1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tau", [1e-6, 1e-9])
def test_tau_far_below_the_score_gaps(dtype, tau):
    _assert_finite(randn(4, 100, dtype=dtype), 5, tau)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_small_tau_still_selects_the_top_k(dtype):
    z = separated(4, 64, gap=0.1, dtype=dtype)
    p, _ = _forward_backward(z, 8, 1e-4)
    assert_close(p, ref_hard(z, 8), 1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tau", [10.0, 1.0, 0.01])
def test_large_magnitude_logits(dtype, tau):
    _assert_finite(1e4 * randn(4, 50, dtype=dtype), 5, tau)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tau", [1.0, 1e-3, 1e-6])
def test_tie_at_the_kth_position(dtype, tau):
    # k = 3 cuts through the three tied entries at 1.0
    z = torch.tensor([[3.0, 2.0, 1.0, 1.0, 1.0, 0.0, -1.0, -2.0]], dtype=dtype)
    p, grad = _assert_finite(z, 3, tau)
    assert_close(p[:, 2:5], p[:, 2:3].expand(-1, 3), 1e-6)
    m = topk_softmax.soft_topk_mask(z, 3, tau)
    assert torch.isfinite(m).all()
    assert_close(m.sum(-1), torch.tensor([3.0]), 1e-5)
    assert_close(m[:, 2:5], m[:, 2:3].expand(-1, 3), 1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tau", [1.0, 1e-3])
def test_two_distant_clusters(dtype, tau):
    # k equals the size of the upper cluster, so sum(m) is nearly flat in t between the clusters
    z = torch.tensor([[50.0, 50.1, 49.9, -50.0, -50.1, -49.9, -50.2]], dtype=dtype)
    p, _ = _assert_finite(z, 3, tau)
    assert_close(p, ref_hard(z, 3), 1e-5)
