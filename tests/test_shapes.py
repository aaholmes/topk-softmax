import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

import softk
from _reference import assert_close, randn, ref_mask, ref_soft


@st.composite
def shaped_problems(draw):
    """z of rank 1-4, any valid dim (positive or negative), k from 1 to n + 1."""
    rank = draw(st.integers(1, 4))
    shape = [draw(st.integers(1, 3)) for _ in range(rank)]
    dim = draw(st.integers(-rank, rank - 1))
    shape[dim] = draw(st.integers(2, 5))
    z = randn(*shape, seed=draw(st.integers(0, 2**16)))
    k = draw(st.integers(1, shape[dim] + 1))
    tau = draw(st.sampled_from([0.1, 1.0, 5.0]))
    return z, k, tau, dim


@given(shaped_problems())
def test_softmax_matches_reference_along_any_dim(problem):
    z, k, tau, dim = problem
    p = softk.soft_topk_softmax(z, k, tau, dim=dim)
    assert p.shape == z.shape
    assert_close(p, ref_soft(z, k, tau, dim), 1e-9)


@given(shaped_problems())
def test_mask_matches_reference_along_any_dim(problem):
    z, k, tau, dim = problem
    k = min(k, z.shape[dim] - 1)
    m = softk.soft_topk_mask(z, k, tau, dim=dim)
    assert m.shape == z.shape
    assert_close(m, ref_mask(z, k, tau, dim), 1e-9)
    assert_close(m.sum(dim), torch.full_like(z.sum(dim), float(k)), 1e-10)


def test_dim_defaults_to_last():
    z = randn(3, 4, 5)
    assert_close(softk.soft_topk_softmax(z, 2, 1.0), softk.soft_topk_softmax(z, 2, 1.0, dim=-1), 0.0)
    assert_close(softk.soft_topk_softmax(z, 2, 1.0), ref_soft(z, 2, 1.0, -1), 1e-9)
    assert_close(softk.soft_topk_mask(z, 2, 1.0), ref_mask(z, 2, 1.0, -1), 1e-9)


def test_one_dimensional_input():
    z = randn(9)
    p = softk.soft_topk_softmax(z, 3, 0.5)
    assert p.shape == (9,)
    assert_close(p, ref_soft(z, 3, 0.5), 1e-9)


NON_CONTIGUOUS = {
    "transposed": lambda base: base.transpose(0, 2),
    "strided": lambda base: base[:, :, ::2],
    "reversed-slice": lambda base: base[1:, :, 1:],
    "expanded": lambda base: base[:1].expand(3, -1, -1),
}


@pytest.mark.parametrize("view", NON_CONTIGUOUS)
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_non_contiguous_input(view, dim):
    z = NON_CONTIGUOUS[view](randn(4, 5, 6))
    if view != "reversed-slice":
        assert not z.is_contiguous()
    p = softk.soft_topk_softmax(z, 2, 0.7, dim=dim)
    assert_close(p, softk.soft_topk_softmax(z.contiguous(), 2, 0.7, dim=dim), 1e-12)
    assert_close(p, ref_soft(z, 2, 0.7, dim), 1e-9)


@pytest.mark.parametrize("view", NON_CONTIGUOUS)
def test_gradient_through_non_contiguous_input(view):
    base = randn(4, 5, 6).requires_grad_()
    w = randn(*NON_CONTIGUOUS[view](base).shape, seed=1)
    (softk.soft_topk_softmax(NON_CONTIGUOUS[view](base), 2, 0.7, dim=1) * w).sum().backward()
    got = base.grad.clone()

    base2 = randn(4, 5, 6).requires_grad_()
    (softk.soft_topk_softmax(NON_CONTIGUOUS[view](base2).contiguous(), 2, 0.7, dim=1) * w).sum().backward()
    assert_close(got, base2.grad, 1e-12)


def test_empty_batch():
    z = torch.zeros(0, 5, dtype=torch.float64)
    assert softk.soft_topk_softmax(z, 2, 1.0).shape == (0, 5)
    assert softk.soft_topk_mask(z, 2, 1.0).shape == (0, 5)


@pytest.mark.parametrize("k", [2, 7])
def test_input_is_not_modified(k):
    z = randn(3, 5)
    before = z.clone()
    softk.soft_topk_softmax(z, k, 0.5)
    softk.soft_topk_softmax(z, k, 0.5, dim=0)
    softk.soft_topk_mask(z, 2, 0.5)
    assert torch.equal(z, before)
