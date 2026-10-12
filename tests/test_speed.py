"""The fused kernels have to be faster than what torch already provides.

Each test times three arms on one input: eager, torch.compile of the eager code, and the Triton
backend. Times are medians of 20 calls after 5 warm-up calls. The assertion is a ratio between arms
on one machine, so it does not depend on the GPU model, but it is still a timing test: run it on an
idle GPU.
"""
import statistics
import time

import pytest
import torch

import topk_softmax
from _reference import randn

pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

BATCH = 4096
# Required speedup over the faster of eager and compiled. The forward-and-backward figure is lower
# because the compiled code already runs within 1.7x of a plain softmax there (see
# benchmarks/results/baseline.json), which leaves little room.
FORWARD_SPEEDUP = 1.5
FORWARD_BACKWARD_SPEEDUP = 1.1


def _median_ms(call):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        times.append(1e3 * (time.perf_counter() - start))
    return statistics.median(times)


def _arms(k, tau, flat_top=False):
    torch._dynamo.reset()
    compiled = torch.compile(topk_softmax.soft_topk_softmax)
    return {
        "triton": lambda z: topk_softmax.soft_topk_softmax(z, k, tau, backend="triton", flat_top=flat_top),
        "eager": lambda z: topk_softmax.soft_topk_softmax(z, k, tau, backend="eager", flat_top=flat_top),
        "compiled": lambda z: compiled(z, k, tau, backend="eager", flat_top=flat_top),
    }


CASES = [(1024, False), (1024, True), (16384, False), (16384, True), (262144, True)]


def _batch(n):
    return min(BATCH, 2**27 // n)  # at most 512 MB of scores


def _assert_faster(times, speedup):
    best_existing = min(times["eager"], times["compiled"])
    assert times["triton"] * speedup <= best_existing, {name: f"{t:.3f} ms" for name, t in times.items()}


@pytest.mark.parametrize("n,flat_top", CASES)
def test_triton_forward_is_faster(n, flat_top):
    z = randn(_batch(n), n, dtype=torch.float32).cuda()

    def timed(fn):
        def call():
            with torch.no_grad():
                fn(z)
        return _median_ms(call)

    _assert_faster({name: timed(fn) for name, fn in _arms(n // 8, 1.0, flat_top).items()}, FORWARD_SPEEDUP)


@pytest.mark.parametrize("n,flat_top", CASES)
def test_triton_forward_and_backward_are_faster(n, flat_top):
    z = randn(_batch(n), n, dtype=torch.float32).cuda()
    w = randn(_batch(n), n, dtype=torch.float32, seed=1).cuda()

    def timed(fn):
        def call():
            x = z.detach().requires_grad_()
            (fn(x) * w).sum().backward()
        return _median_ms(call)

    _assert_faster({name: timed(fn) for name, fn in _arms(n // 8, 1.0, flat_top).items()}, FORWARD_BACKWARD_SPEEDUP)
