import torch
from hypothesis import given
from hypothesis import strategies as st

import softk
from _reference import assert_close

finite = dict(allow_nan=False, allow_infinity=False)


@st.composite
def problems(draw):
    """z of shape (rows, n) in fp64, k integer or real in [0.5, n - 0.5], tau in [0.05, 20]."""
    n = draw(st.integers(2, 10))
    rows = draw(st.integers(1, 3))
    vals = draw(st.lists(st.floats(-5, 5, **finite), min_size=rows * n, max_size=rows * n))
    z = torch.tensor(vals, dtype=torch.float64).reshape(rows, n)
    k = draw(st.one_of(st.integers(1, n - 1), st.floats(0.5, n - 0.5, **finite)))
    tau = draw(st.floats(0.05, 20.0, **finite))
    return z, k, tau


@given(problems())
def test_output_is_a_probability_vector(problem):
    z, k, tau = problem
    p = softk.soft_topk_softmax(z, k, tau)
    assert torch.isfinite(p).all()
    assert (p >= 0).all()
    assert_close(p.sum(-1), torch.ones(z.shape[0]), 1e-10)


@given(problems())
def test_mask_is_feasible(problem):
    z, k, tau = problem
    m = softk.soft_topk_mask(z, k, tau)
    assert (m >= 0).all() and (m <= 1).all()
    assert_close(m.sum(-1), torch.full((z.shape[0],), float(k), dtype=torch.float64), 1e-9)


@given(problems(), st.floats(-10, 10, **finite))
def test_adding_a_constant_changes_nothing(problem, c):
    z, k, tau = problem
    assert_close(softk.soft_topk_softmax(z + c, k, tau), softk.soft_topk_softmax(z, k, tau), 1e-8)
    assert_close(softk.soft_topk_mask(z + c, k, tau), softk.soft_topk_mask(z, k, tau), 1e-8)


@given(problems(), st.floats(0.1, 10, **finite))
def test_mask_depends_only_on_z_over_tau(problem, a):
    z, k, tau = problem
    assert_close(softk.soft_topk_mask(a * z, k, a * tau), softk.soft_topk_mask(z, k, tau), 1e-8)


@given(problems(), st.randoms(use_true_random=False))
def test_permutation_equivariance(problem, rnd):
    z, k, tau = problem
    perm = list(range(z.shape[-1]))
    rnd.shuffle(perm)
    assert_close(softk.soft_topk_softmax(z[:, perm], k, tau), softk.soft_topk_softmax(z, k, tau)[:, perm], 1e-10)
    assert_close(softk.soft_topk_mask(z[:, perm], k, tau), softk.soft_topk_mask(z, k, tau)[:, perm], 1e-10)


@given(problems())
def test_larger_scores_get_no_less_probability(problem):
    z, k, tau = problem
    p = softk.soft_topk_softmax(z, k, tau)
    assert (p.gather(-1, z.argsort(-1)).diff(dim=-1) >= -1e-12).all()


@given(problems())
def test_mass_moves_toward_the_largest_scores(problem):
    # p_i / softmax(z)_i is nondecreasing in z_i, so the j largest scores hold at least as much
    # mass under p as under softmax(z), for every j
    z, k, tau = problem
    order = z.argsort(-1, descending=True)
    p = softk.soft_topk_softmax(z, k, tau).gather(-1, order).cumsum(-1)
    s = torch.softmax(z, -1).gather(-1, order).cumsum(-1)
    assert (p >= s - 1e-9).all()
