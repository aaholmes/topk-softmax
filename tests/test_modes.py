import inspect

import pytest
import torch

import softk
from _reference import assert_close, randn, ref_hard

DTYPES = [torch.float32, torch.float64]


@pytest.mark.parametrize("k", [1, 3, 11])
@pytest.mark.parametrize("dim", [-1, 0])
def test_hard_is_softmax_over_the_top_k(k, dim):
    z = randn(12, 12)
    p = softk.soft_topk_softmax(z, k, 1.0, dim=dim, mode="hard")
    assert_close(p, ref_hard(z, k, dim), 1e-12)
    assert ((p != 0).sum(dim) == k).all()
    assert_close(p.sum(dim), torch.ones(12), 1e-12)


@pytest.mark.parametrize("dtype", DTYPES)
def test_hard_has_at_most_k_nonzeros_with_ties(dtype):
    z = torch.tensor([[3.0, 2.0, 1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=dtype)
    p = softk.soft_topk_softmax(z, 3, 1.0, mode="hard")
    assert ((p != 0).sum(-1) <= 3).all()
    assert_close(p.sum(-1), torch.ones(2), 1e-6)


def test_hard_does_not_depend_on_tau():
    z = randn(5, 12)
    a = softk.soft_topk_softmax(z, 3, 0.01, mode="hard")
    b = softk.soft_topk_softmax(z, 3, 100.0, mode="hard")
    assert torch.equal(a, b)


@pytest.mark.parametrize("mode", ["hard", "st"])
def test_k_at_least_n_returns_softmax(mode):
    z = randn(5, 12)
    assert_close(softk.soft_topk_softmax(z, 12, 1.0, mode=mode), torch.softmax(z, -1), 1e-12)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tau", [5.0, 0.5, 0.01])
@pytest.mark.parametrize("dim", [-1, 0])
def test_st_forward_equals_hard_bitwise(dtype, tau, dim):
    z = randn(12, 12, dtype=dtype)
    st = softk.soft_topk_softmax(z, 3, tau, dim=dim, mode="st")
    hard = softk.soft_topk_softmax(z, 3, tau, dim=dim, mode="hard")
    assert torch.equal(st, hard)


@pytest.mark.parametrize("tau", [5.0, 0.5, 0.05])
@pytest.mark.parametrize("dim", [-1, 0])
def test_st_gradient_equals_soft_gradient(tau, dim):
    z = randn(12, 12)
    w = randn(12, 12, seed=1)
    grads = {}
    for mode in ("st", "soft"):
        x = z.clone().requires_grad_()
        (softk.soft_topk_softmax(x, 3, tau, dim=dim, mode=mode) * w).sum().backward()
        grads[mode] = x.grad
    assert grads["soft"].abs().max().item() > 1e-6
    assert_close(grads["st"], grads["soft"], 1e-12)


def test_mode_defaults_to_soft():
    assert inspect.signature(softk.soft_topk_softmax).parameters["mode"].default == "soft"
    z = randn(5, 12)
    assert torch.equal(softk.soft_topk_softmax(z, 3, 0.5), softk.soft_topk_softmax(z, 3, 0.5, mode="soft"))


@pytest.mark.parametrize("mode", ["Soft", "straight-through", "", None])
def test_unknown_mode_raises(mode):
    with pytest.raises(ValueError):
        softk.soft_topk_softmax(randn(5, 12), 3, 0.5, mode=mode)
