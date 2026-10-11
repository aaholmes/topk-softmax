import pytest
import torch

import topk_softmax
from _reference import assert_close, randn

STAT_KEYS = {"mass_outside_mean", "mass_outside_p99", "effective_support", "sum_v"}


# ---- mass_outside_topk ----

def test_mass_outside_matches_a_direct_sum():
    p = torch.softmax(randn(6, 10), -1)
    expected = 1 - p.sort(-1, descending=True).values[:, :3].sum(-1)
    assert_close(topk_softmax.stats.mass_outside_topk(p, 3), expected, 1e-12)


def test_mass_outside_is_zero_for_a_hard_output():
    p = topk_softmax.soft_topk_softmax(randn(6, 10), 3, 1.0, mode="hard")
    assert_close(topk_softmax.stats.mass_outside_topk(p, 3), torch.zeros(6), 1e-12)


def test_mass_outside_of_a_uniform_row():
    p = torch.full((2, 8), 1 / 8, dtype=torch.float64)
    assert_close(topk_softmax.stats.mass_outside_topk(p, 3), torch.full((2,), 5 / 8), 1e-12)


def test_mass_outside_lies_in_unit_interval_and_decreases_with_k():
    p = torch.softmax(randn(6, 10), -1)
    masses = [topk_softmax.stats.mass_outside_topk(p, k) for k in (1, 3, 9)]
    for m in masses:
        assert (m >= 0).all() and (m <= 1).all()
    assert (masses[0] >= masses[1]).all() and (masses[1] >= masses[2]).all()


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_mass_outside_along_any_dim(dim):
    p = torch.softmax(randn(4, 5, 6), dim)
    out = topk_softmax.stats.mass_outside_topk(p, 2, dim=dim)
    assert_close(out, 1 - p.topk(2, dim).values.sum(dim), 1e-12)


# ---- effective_support ----

def test_effective_support_of_a_uniform_row_is_n():
    p = torch.full((2, 8), 1 / 8, dtype=torch.float64)
    assert_close(topk_softmax.stats.effective_support(p), torch.full((2,), 8.0), 1e-10)


def test_effective_support_of_a_one_hot_row_is_one():
    p = torch.zeros(2, 8, dtype=torch.float64)
    p[:, 3] = 1.0
    out = topk_softmax.stats.effective_support(p)
    assert torch.isfinite(out).all()
    assert_close(out, torch.ones(2), 1e-10)


def test_effective_support_of_uniform_over_k_entries_is_k():
    p = torch.zeros(2, 8, dtype=torch.float64)
    p[:, :3] = 1 / 3
    assert_close(topk_softmax.stats.effective_support(p), torch.full((2,), 3.0), 1e-10)


def test_effective_support_matches_exp_entropy():
    p = torch.softmax(randn(6, 10), -1)
    assert_close(topk_softmax.stats.effective_support(p), torch.exp(-(p * p.log()).sum(-1)), 1e-10)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_effective_support_along_any_dim(dim):
    p = torch.softmax(randn(4, 5, 6), dim)
    assert_close(topk_softmax.stats.effective_support(p, dim=dim), torch.exp(-(p * p.log()).sum(dim)), 1e-10)


# ---- topk_churn ----

def _scores(top, n=8):
    """One row whose largest entries are at the indices in `top`, in that order."""
    z = torch.zeros(n, dtype=torch.float64)
    for rank, i in enumerate(top):
        z[i] = 10.0 - rank
    return z


def test_churn_is_zero_for_identical_scores():
    z = randn(6, 10)
    assert_close(topk_softmax.stats.topk_churn(z, z.clone(), 3), torch.zeros(6), 0.0)


def test_churn_ignores_reordering_within_the_top_k():
    old = _scores([0, 1, 2, 3])[None]
    new = _scores([3, 2, 1, 0])[None]
    assert_close(topk_softmax.stats.topk_churn(old, new, 4), torch.zeros(1), 0.0)


def test_churn_is_one_for_disjoint_top_k_sets():
    old = _scores([0, 1, 2, 3])[None]
    new = _scores([4, 5, 6, 7])[None]
    assert_close(topk_softmax.stats.topk_churn(old, new, 4), torch.ones(1), 0.0)


@pytest.mark.parametrize("kept,expected", [(3, 0.25), (2, 0.5), (1, 0.75)])
def test_churn_is_the_fraction_of_indices_replaced(kept, expected):
    old = _scores([0, 1, 2, 3])[None]
    new = _scores(list(range(kept)) + [4, 5, 6][: 4 - kept])[None]
    assert_close(topk_softmax.stats.topk_churn(old, new, 4), torch.tensor([expected]), 1e-12)


def test_churn_is_per_row_and_follows_dim():
    old = torch.stack([_scores([0, 1]), _scores([0, 1])])
    new = torch.stack([_scores([0, 1]), _scores([0, 5])])
    assert_close(topk_softmax.stats.topk_churn(old, new, 2), torch.tensor([0.0, 0.5]), 1e-12)
    assert_close(topk_softmax.stats.topk_churn(old.T, new.T, 2, dim=0), torch.tensor([0.0, 0.5]), 1e-12)


# ---- stats recorded by the module ----

def test_module_stats_are_empty_unless_tracking():
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.5)
    assert layer.stats == {}
    layer(randn(6, 10))
    assert layer.stats == {}


def test_module_stats_before_the_first_forward_are_empty():
    assert topk_softmax.SoftTopKSoftmax(3, tau=0.5, track_stats=True).stats == {}


@pytest.mark.parametrize("dim", [-1, 0])
def test_module_stats_match_the_stat_functions(dim):
    z = randn(40, 40)
    layer = topk_softmax.SoftTopKSoftmax(3, dim=dim, tau=0.5, track_stats=True)
    p = layer(z)
    assert set(layer.stats) == STAT_KEYS

    mass = topk_softmax.stats.mass_outside_topk(p, 3, dim=dim)
    m = topk_softmax.soft_topk_mask(z, 3, 0.5, dim=dim)
    expected = {
        "mass_outside_mean": mass.mean(),
        "mass_outside_p99": torch.quantile(mass, 0.99),
        "effective_support": topk_softmax.stats.effective_support(p, dim=dim).mean(),
        "sum_v": (m * (1 - m)).sum(dim).mean(),
    }
    for key, value in expected.items():
        assert float(layer.stats[key]) == pytest.approx(float(value), rel=1e-6, abs=1e-9), key


def test_module_stats_carry_no_gradient():
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.5, track_stats=True)
    layer(randn(6, 10).requires_grad_())
    for key, value in layer.stats.items():
        assert not getattr(value, "requires_grad", False), key


def test_module_stats_are_refreshed_on_each_forward():
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.5, track_stats=True)
    layer(randn(6, 10))
    first = float(layer.stats["mass_outside_mean"])
    layer(randn(6, 10, seed=1))
    assert float(layer.stats["mass_outside_mean"]) != first


def test_module_stats_in_hard_mode_report_no_mass_outside():
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.5, mode="hard", track_stats=True)
    layer(randn(6, 10))
    assert float(layer.stats["mass_outside_mean"]) == pytest.approx(0.0, abs=1e-12)
