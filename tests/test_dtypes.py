import pytest
import torch

import topk_softmax
from _reference import assert_close, randn

HALF = [torch.bfloat16, torch.float16]
FULL = [torch.float32, torch.float64]
MODES = ["soft", "hard", "st"]


@pytest.mark.parametrize("dtype", HALF)
@pytest.mark.parametrize("tau", [5.0, 1.0, 0.05])
def test_half_precision_output(dtype, tau):
    z = randn(4, 64, dtype=dtype)
    p = topk_softmax.soft_topk_softmax(z, 8, tau)
    assert p.dtype == dtype
    assert torch.isfinite(p).all()
    assert_close(p.float().sum(-1), torch.ones(4), 1e-2)


@pytest.mark.parametrize("dtype", HALF)
@pytest.mark.parametrize("tau", [5.0, 1.0, 0.05])
def test_half_precision_is_computed_in_fp32(dtype, tau):
    # The result should be the fp32 result rounded to the input dtype. 4e-3 is one bfloat16
    # spacing just below 1; bisection carried out in bfloat16 misses by far more than that.
    z = randn(4, 64, dtype=dtype)
    expected = topk_softmax.soft_topk_softmax(z.float(), 8, tau)
    assert_close(topk_softmax.soft_topk_softmax(z, 8, tau).float(), expected, 4e-3)
    assert_close(topk_softmax.soft_topk_mask(z, 8, tau).float(), topk_softmax.soft_topk_mask(z.float(), 8, tau), 4e-3)


@pytest.mark.parametrize("dtype", HALF)
def test_half_precision_mask(dtype):
    m = topk_softmax.soft_topk_mask(randn(4, 64, dtype=dtype), 8, 1.0)
    assert m.dtype == dtype
    assert torch.isfinite(m).all()


@pytest.mark.parametrize("dtype", HALF + FULL)
@pytest.mark.parametrize("mode", MODES)
def test_output_dtype_matches_input(dtype, mode):
    z = randn(4, 16, dtype=dtype)
    assert topk_softmax.soft_topk_softmax(z, 3, 1.0, mode=mode).dtype == dtype
    assert topk_softmax.soft_topk_softmax(z, 99, 1.0, mode=mode).dtype == dtype  # the k >= n branch


@pytest.mark.parametrize("dtype", FULL)
def test_mask_dtype_matches_input(dtype):
    assert topk_softmax.soft_topk_mask(randn(4, 16, dtype=dtype), 3, 1.0).dtype == dtype


@pytest.mark.parametrize("dtype", HALF + FULL)
@pytest.mark.parametrize("mode", ["soft", "st"])
def test_gradient_dtype_matches_input(dtype, mode):
    z = randn(4, 64, dtype=dtype).requires_grad_()
    p = topk_softmax.soft_topk_softmax(z, 8, 0.5, mode=mode)
    (p.float() * randn(4, 64, dtype=torch.float32, seed=1)).sum().backward()
    assert z.grad.dtype == dtype
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().max().item() > 0


def test_fp32_agrees_with_fp64():
    z = randn(4, 64)
    for tau in (5.0, 1.0, 0.1):
        p32 = topk_softmax.soft_topk_softmax(z.float(), 8, tau)
        assert_close(p32, topk_softmax.soft_topk_softmax(z.float().double(), 8, tau), 1e-5)
