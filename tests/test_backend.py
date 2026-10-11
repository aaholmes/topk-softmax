"""The `backend` argument: "eager" is the reference implementation, "triton" the fused GPU kernels.

The Triton results are checked by comparing them with the eager results on identical inputs.
"""
import inspect

import pytest
import torch

import topk_softmax
from _reference import assert_close, hard_cases, mask_sum_tol, randn, ref_soft



def on_cuda(test):
    return pytest.mark.gpu(pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")(test))

MODES = ["soft", "hard", "st"]


def _both(z, k, tau, **kwargs):
    return (topk_softmax.soft_topk_softmax(z, k, tau, backend="triton", **kwargs),
            topk_softmax.soft_topk_softmax(z, k, tau, backend="eager", **kwargs))


def _grad(z, k, tau, backend, mode="soft", dim=-1):
    w = randn(*z.shape, dtype=z.dtype, seed=1).to(z.device)
    x = z.clone().requires_grad_()
    (topk_softmax.soft_topk_softmax(x, k, tau, dim=dim, mode=mode, backend=backend) * w).sum().backward()
    return x.grad


# ---- on any machine ----

@pytest.mark.parametrize("fn", ["soft_topk_softmax", "soft_topk_mask", "SoftTopKSoftmax"])
def test_backend_argument_defaults_to_auto(fn):
    assert inspect.signature(getattr(topk_softmax, fn)).parameters["backend"].default == "auto"


@pytest.mark.parametrize("mode", MODES)
def test_auto_equals_eager_on_cpu(mode):
    z = randn(5, 12)
    assert torch.equal(topk_softmax.soft_topk_softmax(z, 3, 0.5, mode=mode, backend="auto"),
                       topk_softmax.soft_topk_softmax(z, 3, 0.5, mode=mode, backend="eager"))
    assert torch.equal(topk_softmax.soft_topk_mask(z, 3, 0.5, backend="auto"),
                       topk_softmax.soft_topk_mask(z, 3, 0.5, backend="eager"))


def test_eager_backend_matches_the_reference():
    z = randn(5, 12)
    assert_close(topk_softmax.soft_topk_softmax(z, 3, 0.5, backend="eager"), ref_soft(z, 3, 0.5), 1e-10)


def test_triton_on_a_cpu_tensor_raises():
    with pytest.raises(ValueError):
        topk_softmax.soft_topk_softmax(randn(5, 12), 3, 0.5, backend="triton")
    with pytest.raises(ValueError):
        topk_softmax.soft_topk_mask(randn(5, 12), 3, 0.5, backend="triton")


@pytest.mark.parametrize("backend", ["cuda", "Triton", "", None])
def test_unknown_backend_raises(backend):
    with pytest.raises(ValueError):
        topk_softmax.soft_topk_softmax(randn(5, 12), 3, 0.5, backend=backend)
    with pytest.raises(ValueError):
        topk_softmax.SoftTopKSoftmax(3, backend=backend)


def test_module_passes_its_backend_on():
    z = randn(5, 12)
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.5, backend="eager")
    assert layer.backend == "eager"
    assert torch.equal(layer(z), topk_softmax.soft_topk_softmax(z, 3, 0.5, backend="eager"))
    with pytest.raises(ValueError):
        topk_softmax.SoftTopKSoftmax(3, tau=0.5, backend="triton")(z)


# ---- on a CUDA GPU ----

@pytest.mark.parametrize("n,k", [(2, 1), (7, 3), (64, 8), (1000, 1), (1000, 100), (1000, 999), (1000, 2.5),
                                 (16384, 2048), (131072, 64)])
@pytest.mark.parametrize("tau", [10.0, 1.0, 0.05])
@on_cuda
def test_triton_output_matches_eager(n, k, tau):
    z = randn(3, n, dtype=torch.float32).cuda()
    triton, eager = _both(z, k, tau)
    assert triton.dtype == z.dtype and triton.device == z.device
    assert_close(triton.cpu(), eager.cpu(), 1e-6)
    assert_close(triton.sum(-1).cpu(), torch.ones(3), 1e-5)


@pytest.mark.parametrize("n,k", [(2, 1), (7, 3), (64, 8), (1000, 100), (1000, 2.5), (16384, 2048), (131072, 64)])
@pytest.mark.parametrize("tau", [10.0, 1.0, 0.05])
@on_cuda
def test_triton_mask_matches_eager_and_sums_to_k(n, k, tau):
    z = randn(3, n, dtype=torch.float32).cuda()
    m = topk_softmax.soft_topk_mask(z, k, tau, backend="triton")
    assert_close(m.cpu(), topk_softmax.soft_topk_mask(z, k, tau, backend="eager").cpu(), 1e-5)
    assert_close(m.double().sum(-1).cpu(), torch.full((3,), float(k), dtype=torch.float64),
                 mask_sum_tol(k, torch.float32))


@pytest.mark.parametrize("n,k", [(2, 1), (7, 3), (64, 8), (1000, 100), (1000, 2.5), (16384, 2048), (131072, 64)])
@pytest.mark.parametrize("tau", [10.0, 1.0, 0.05])
@on_cuda
def test_triton_gradient_matches_eager(n, k, tau):
    z = randn(3, n, dtype=torch.float32).cuda()
    eager = _grad(z, k, tau, "eager")
    assert_close(_grad(z, k, tau, "triton").cpu(), eager.cpu(), 1e-5)


@pytest.mark.parametrize("mode", MODES)
@on_cuda
def test_triton_modes(mode):
    z = randn(4, 64, dtype=torch.float32).cuda()
    triton, eager = _both(z, 8, 0.5, mode=mode)
    assert_close(triton.cpu(), eager.cpu(), 1e-6)
    if mode != "soft":
        assert ((triton != 0).sum(-1) == 8).all()
    if mode != "hard":
        assert_close(_grad(z, 8, 0.5, "triton", mode).cpu(), _grad(z, 8, 0.5, "eager", mode).cpu(), 1e-5)


@pytest.mark.parametrize("dim", [0, 1, -1])
@on_cuda
def test_triton_any_dim(dim):
    z = randn(5, 6, 7, dtype=torch.float32).cuda()
    triton, eager = _both(z, 2, 0.5, dim=dim)
    assert_close(triton.cpu(), eager.cpu(), 1e-6)
    assert_close(_grad(z, 2, 0.5, "triton", dim=dim).cpu(), _grad(z, 2, 0.5, "eager", dim=dim).cpu(), 1e-5)


@on_cuda
def test_triton_non_contiguous_input_and_batch_shapes():
    zt = randn(64, 6, dtype=torch.float32).cuda().t()
    assert not zt.is_contiguous()
    assert_close(topk_softmax.soft_topk_softmax(zt, 8, 0.5, backend="triton").cpu(),
                 topk_softmax.soft_topk_softmax(zt.contiguous(), 8, 0.5, backend="eager").cpu(), 1e-6)
    for shape in [(64,), (1, 64), (2, 3, 4, 64), (0, 64)]:
        z = randn(*shape, dtype=torch.float32).cuda()
        triton, eager = _both(z, 8, 0.5)
        assert_close(triton.cpu(), eager.cpu(), 1e-6)


@on_cuda
def test_triton_k_at_least_n_gives_softmax():
    z = randn(5, 4, dtype=torch.float32).cuda()
    assert_close(topk_softmax.soft_topk_softmax(z, 9, 0.5, backend="triton").cpu(), torch.softmax(z, -1).cpu(), 1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
@on_cuda
def test_triton_other_dtypes(dtype):
    # 64-bit inputs may be computed by either implementation, but must keep 64-bit accuracy
    z = randn(4, 64, dtype=dtype).cuda()
    triton, eager = _both(z, 8, 0.5)
    assert triton.dtype == dtype
    assert torch.isfinite(triton).all()
    assert_close(triton.cpu(), eager.cpu(), {torch.float64: 1e-12}.get(dtype, 1e-2))


@on_cuda
def test_triton_hard_cases():
    for name, z, k, tau in hard_cases(torch.float32):
        z = z.cuda()
        triton, eager = _both(z, k, tau)
        assert torch.isfinite(triton).all(), name
        assert (triton.cpu() - eager.cpu()).abs().max().item() <= 1e-5, name
        m = topk_softmax.soft_topk_mask(z, k, tau, backend="triton")
        assert (m.double().sum(-1) - k).abs().max().item() <= mask_sum_tol(k, torch.float32), name
        assert torch.isfinite(_grad(z, k, tau, "triton")).all(), name


@on_cuda
def test_triton_small_tau_wide_row_is_finite():
    z = randn(2, 32768, dtype=torch.float32).cuda()
    p = topk_softmax.soft_topk_softmax(z, 8, 1e-4, backend="triton")
    assert torch.isfinite(p).all()
    assert torch.isfinite(_grad(z, 8, 1e-4, "triton")).all()


@on_cuda
def test_auto_uses_triton_on_cuda():
    z = randn(4, 1000, dtype=torch.float32).cuda()
    assert torch.equal(topk_softmax.soft_topk_softmax(z, 100, 0.5, backend="auto"),
                       topk_softmax.soft_topk_softmax(z, 100, 0.5, backend="triton"))


@on_cuda
def test_triton_module_follows_the_tau_buffer():
    z = randn(5, 64, dtype=torch.float32).cuda()
    layer = topk_softmax.SoftTopKSoftmax(8, tau=5.0, backend="triton").cuda()
    with torch.no_grad():
        layer.tau.fill_(0.125)
    assert_close(layer(z).cpu(), topk_softmax.soft_topk_softmax(z, 8, 0.125, backend="eager").cpu(), 1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES)
@on_cuda
def test_triton_under_torch_compile(mode):
    torch._dynamo.reset()
    z = randn(4, 64, dtype=torch.float32).cuda()
    fn = torch.compile(topk_softmax.soft_topk_softmax)
    assert_close(fn(z, 8, 0.5, mode=mode, backend="triton").cpu(),
                 topk_softmax.soft_topk_softmax(z, 8, 0.5, mode=mode, backend="eager").cpu(), 1e-6)
    explanation = torch._dynamo.explain(topk_softmax.soft_topk_softmax)(z, 8, 0.5, mode=mode, backend="triton")
    assert explanation.graph_break_count == 0, explanation.break_reasons
    torch._dynamo.reset()
