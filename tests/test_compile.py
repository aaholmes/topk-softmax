import pytest
import torch

import topk_softmax
from _reference import assert_close, randn

pytestmark = pytest.mark.slow

MODES = ["soft", "hard", "st"]


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("mode", MODES)
def test_compiled_matches_eager(mode):
    z = randn(4, 64, dtype=torch.float32)
    compiled = torch.compile(topk_softmax.soft_topk_softmax)
    assert_close(compiled(z, 8, 0.5, mode=mode), topk_softmax.soft_topk_softmax(z, 8, 0.5, mode=mode), 1e-6)


def test_compiled_gradient_matches_eager():
    z = randn(4, 64, dtype=torch.float32)
    w = randn(4, 64, dtype=torch.float32, seed=1)
    compiled = torch.compile(topk_softmax.soft_topk_softmax)
    grads = []
    for fn in (compiled, topk_softmax.soft_topk_softmax):
        x = z.clone().requires_grad_()
        (fn(x, 8, 0.5) * w).sum().backward()
        grads.append(x.grad)
    assert_close(grads[0], grads[1], 1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_no_graph_breaks(mode):
    z = randn(4, 64, dtype=torch.float32)
    explanation = torch._dynamo.explain(topk_softmax.soft_topk_softmax)(z, 8, 0.5, mode=mode)
    assert explanation.graph_break_count == 0, explanation.break_reasons
