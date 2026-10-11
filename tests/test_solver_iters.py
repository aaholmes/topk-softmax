"""How many solver iterations the threshold needs.

Measured on these inputs (benchmarks/bench.py --iters), 20 iterations in 32-bit and 24 in 64-bit
floating point give the accuracy of the original 32 and 60. The first two tests guard that
accuracy; the last requires those counts to be the defaults.
"""
import pytest
import torch

import softk
from _reference import hard_cases, mask_sum_tol, ref_mask

ITERS = {torch.float32: 20, torch.float64: 24}
CASES = {name: (z, k, tau) for name, z, k, tau in hard_cases()}


@pytest.mark.parametrize("dtype", list(ITERS))
@pytest.mark.parametrize("name", CASES)
def test_mask_sums_to_k_with_the_reduced_iteration_count(dtype, name):
    z, k, tau = CASES[name]
    m = softk.soft_topk_mask(z.to(dtype), k, tau, iters=ITERS[dtype])
    assert torch.isfinite(m).all()
    assert (m.double().sum(-1) - k).abs().max().item() <= mask_sum_tol(k, dtype)


@pytest.mark.parametrize("name", [n for n, (z, _, tau) in CASES.items() if z.shape[-1] <= 1000 and 1e-2 <= tau <= 1e2])
def test_mask_matches_the_reference_with_the_reduced_iteration_count(name):
    # restricted to tau above the floor, where the reference and the library solve one equation
    z, k, tau = CASES[name]
    m = softk.soft_topk_mask(z, k, tau, iters=ITERS[torch.float64])
    assert (m - ref_mask(z, k, tau)).abs().max().item() <= 1e-9


@pytest.mark.parametrize("dtype", list(ITERS))
def test_the_reduced_counts_are_the_defaults(dtype):
    for name, (z, k, tau) in CASES.items():
        z = z.to(dtype)
        assert torch.equal(softk.soft_topk_mask(z, k, tau), softk.soft_topk_mask(z, k, tau, iters=ITERS[dtype])), name
