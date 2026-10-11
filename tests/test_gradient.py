import pytest
import torch
import torch.nn.functional as F

import topk_softmax
from _reference import assert_close, randn, ref_threshold

TAUS = [5.0, 1.0, 0.1]


def _ks(n):
    return sorted({1, max(1, n // 2), n - 1})


SMALL = [(tau, n, k) for tau in TAUS for n in (2, 7) for k in _ks(n)]
LARGE = [(tau, 1000, k) for tau in TAUS for k in _ks(1000)]


def _gradcheck(fn, z):
    return torch.autograd.gradcheck(fn, (z,), eps=1e-6, atol=1e-5)


@pytest.mark.parametrize("tau,n,k", SMALL)
def test_gradcheck_softmax(tau, n, k):
    z = randn(3, n).requires_grad_()
    assert _gradcheck(lambda x: topk_softmax.soft_topk_softmax(x, k, tau), z)


@pytest.mark.slow
@pytest.mark.parametrize("tau,n,k", LARGE)
def test_gradcheck_softmax_n1000(tau, n, k):
    z = randn(n).requires_grad_()
    assert _gradcheck(lambda x: topk_softmax.soft_topk_softmax(x, k, tau), z)


@pytest.mark.parametrize("tau,n,k", SMALL)
def test_gradcheck_mask(tau, n, k):
    z = randn(3, n).requires_grad_()
    assert _gradcheck(lambda x: topk_softmax.soft_topk_mask(x, k, tau), z)


@pytest.mark.parametrize("tau", TAUS)
def test_gradcheck_real_k(tau):
    z = randn(3, 7).requires_grad_()
    assert _gradcheck(lambda x: topk_softmax.soft_topk_softmax(x, 2.5, tau), z)


@pytest.mark.parametrize("dim", [0, 1, -2])
def test_gradcheck_other_dims(dim):
    z = randn(4, 5, 3).requires_grad_()
    assert _gradcheck(lambda x: topk_softmax.soft_topk_softmax(x, 2, 1.0, dim=dim), z)


def _newton_log_mask(z, k, tau):
    """log m with t written as one Newton step from the converged threshold t0.

    t0 and the denominator are constants, so autograd gives dt/dz_j = v_j / sum(v), which is the
    implicit-function-theorem derivative the custom backward is supposed to implement.
    """
    z = torch.minimum(z, z.topk(int(k), -1).values[..., -1:])  # the default flat_top clamp
    t0 = ref_threshold(z, k, tau)
    s0 = torch.sigmoid((z.detach() - t0) / tau)
    v0 = (s0 * (1 - s0)).sum(-1, keepdim=True)
    t = t0 + tau * (torch.sigmoid((z - t0) / tau).sum(-1, keepdim=True) - k) / v0
    return F.logsigmoid((z - t) / tau)


@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("k", [1, 3, 2.5, 6])
def test_backward_matches_autograd_through_newton_step(tau, k):
    z = randn(5, 7)
    w = randn(5, 7, seed=1)

    za = z.clone().requires_grad_()
    (topk_softmax.soft_topk_softmax(za, k, tau) * w).sum().backward()
    zb = z.clone().requires_grad_()
    (torch.softmax(zb + _newton_log_mask(zb, k, tau), -1) * w).sum().backward()
    assert_close(za.grad, zb.grad, 1e-8)

    za = z.clone().requires_grad_()
    (topk_softmax.soft_topk_mask(za, k, tau) * w).sum().backward()
    zb = z.clone().requires_grad_()
    (_newton_log_mask(zb, k, tau).exp() * w).sum().backward()
    assert_close(za.grad, zb.grad, 1e-8)


@pytest.mark.parametrize("tau", TAUS + [0.01])
def test_gradient_of_mask_sum_is_zero(tau):
    # sum(m) = k for every z, so its gradient vanishes
    z = randn(4, 10).requires_grad_()
    topk_softmax.soft_topk_mask(z, 3, tau).sum().backward()
    assert_close(z.grad, torch.zeros_like(z), 1e-8)


@pytest.mark.parametrize("tau", TAUS + [0.01])
def test_gradient_of_probability_sum_is_zero(tau):
    z = randn(4, 10).requires_grad_()
    topk_softmax.soft_topk_softmax(z, 3, tau).sum().backward()
    assert_close(z.grad, torch.zeros_like(z), 1e-10)


def test_gradient_is_nonzero_for_a_generic_loss():
    z = randn(4, 10).requires_grad_()
    (topk_softmax.soft_topk_softmax(z, 3, 1.0) * randn(4, 10, seed=1)).sum().backward()
    assert z.grad.abs().max().item() > 1e-3
