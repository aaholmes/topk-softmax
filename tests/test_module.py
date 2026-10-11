import inspect

import pytest
import torch

import topk_softmax
from _reference import assert_close, randn

MODES = ["soft", "hard", "st"]


def test_constructor_signature():
    params = inspect.signature(topk_softmax.SoftTopKSoftmax).parameters
    assert list(params)[:5] == ["k", "dim", "tau", "mode", "track_stats"]
    assert params["dim"].default == -1
    assert params["tau"].default == 1.0
    assert params["mode"].default == "soft"
    assert params["track_stats"].default is False


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_forward_equals_functional(mode, dim):
    z = randn(6, 7, 8)
    layer = topk_softmax.SoftTopKSoftmax(3, dim=dim, tau=0.5, mode=mode)
    assert_close(layer(z), topk_softmax.soft_topk_softmax(z, 3, 0.5, dim=dim, mode=mode), 1e-12)


@pytest.mark.parametrize("mode", ["soft", "st"])
def test_gradient_equals_functional(mode):
    z = randn(5, 12)
    w = randn(5, 12, seed=1)
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.5, mode=mode)
    a = z.clone().requires_grad_()
    (layer(a) * w).sum().backward()
    b = z.clone().requires_grad_()
    (topk_softmax.soft_topk_softmax(b, 3, 0.5, mode=mode) * w).sum().backward()
    assert b.grad.abs().max().item() > 1e-6
    assert_close(a.grad, b.grad, 1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_output_dtype_follows_the_input(dtype):
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.5)
    assert layer(randn(4, 16, dtype=dtype)).dtype == dtype


def test_tau_is_a_scalar_buffer_and_not_a_parameter():
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.25)
    assert isinstance(layer.tau, torch.Tensor)
    assert layer.tau.ndim == 0
    assert float(layer.tau) == 0.25
    assert "tau" in dict(layer.named_buffers())
    assert list(layer.parameters()) == []


def test_tau_is_saved_and_restored_by_state_dict():
    saved = topk_softmax.SoftTopKSoftmax(3, tau=0.25).state_dict()
    assert "tau" in saved
    layer = topk_softmax.SoftTopKSoftmax(3, tau=7.0)
    layer.load_state_dict(saved)
    assert float(layer.tau) == 0.25


def test_tau_is_saved_inside_a_larger_model():
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), topk_softmax.SoftTopKSoftmax(3, tau=0.25))
    assert float(model.state_dict()["1.tau"]) == 0.25


def test_changing_the_buffer_changes_the_output():
    z = randn(5, 12)
    layer = topk_softmax.SoftTopKSoftmax(3, tau=5.0)
    with torch.no_grad():
        layer.tau.fill_(0.125)
    assert_close(layer(z), topk_softmax.soft_topk_softmax(z, 3, 0.125), 1e-12)


def test_changing_mode_changes_the_output():
    z = randn(5, 12)
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.5)
    assert layer.mode == "soft"
    layer.mode = "hard"
    assert_close(layer(z), topk_softmax.soft_topk_softmax(z, 3, 0.5, mode="hard"), 1e-12)


def test_eval_does_not_change_mode():
    z = randn(5, 12)
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.5)
    before = layer(z)
    layer.eval()
    assert layer.mode == "soft"
    assert torch.equal(layer(z), before)


def test_k_and_dim_are_readable():
    layer = topk_softmax.SoftTopKSoftmax(3, dim=1)
    assert layer.k == 3
    assert layer.dim == 1


def test_k_at_least_n_gives_softmax():
    z = randn(5, 4)
    assert_close(topk_softmax.SoftTopKSoftmax(9)(z), torch.softmax(z, -1), 1e-12)


@pytest.mark.parametrize("kwargs", [
    dict(k=0), dict(k=-1), dict(k=3, tau=0.0), dict(k=3, tau=-1.0),
    dict(k=3, mode="straight-through"), dict(k=2.5, mode="hard"),
])
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        topk_softmax.SoftTopKSoftmax(**kwargs)


@pytest.mark.parametrize("mode", MODES)
def test_tracking_stats_does_not_change_the_output(mode):
    z = randn(5, 12)
    plain = topk_softmax.SoftTopKSoftmax(3, tau=0.5, mode=mode)
    tracked = topk_softmax.SoftTopKSoftmax(3, tau=0.5, mode=mode, track_stats=True)
    assert torch.equal(tracked(z), plain(z))


def test_tracking_stats_does_not_change_the_gradient():
    z = randn(5, 12)
    w = randn(5, 12, seed=1)
    grads = []
    for track in (False, True):
        x = z.clone().requires_grad_()
        (topk_softmax.SoftTopKSoftmax(3, tau=0.5, track_stats=track)(x) * w).sum().backward()
        grads.append(x.grad)
    assert torch.equal(grads[0], grads[1])


@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES)
def test_module_compiles_without_graph_breaks(mode):
    torch._dynamo.reset()
    z = randn(4, 64, dtype=torch.float32)
    layer = topk_softmax.SoftTopKSoftmax(8, tau=0.5, mode=mode)
    assert_close(torch.compile(layer)(z), layer(z), 1e-6)
    explanation = torch._dynamo.explain(layer)(z)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    torch._dynamo.reset()
