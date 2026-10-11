import pytest

import topk_softmax
from _reference import randn

FUNCTIONS = [topk_softmax.soft_topk_softmax, topk_softmax.soft_topk_mask]


@pytest.mark.parametrize("fn", FUNCTIONS)
@pytest.mark.parametrize("k", [0, -1, -0.5])
def test_nonpositive_k_raises(fn, k):
    with pytest.raises(ValueError):
        fn(randn(4, 10), k, 1.0)


@pytest.mark.parametrize("fn", FUNCTIONS)
@pytest.mark.parametrize("tau", [0, 0.0, -1.0])
def test_nonpositive_tau_raises(fn, tau):
    with pytest.raises(ValueError):
        fn(randn(4, 10), 3, tau)


@pytest.mark.parametrize("mode", ["hard", "st"])
def test_non_integer_k_raises_in_hard_modes(mode):
    with pytest.raises(ValueError):
        topk_softmax.soft_topk_softmax(randn(4, 10), 2.5, 1.0, mode=mode)


def test_non_integer_k_is_allowed_in_soft_mode():
    topk_softmax.soft_topk_softmax(randn(4, 10), 2.5, 1.0, mode="soft")
