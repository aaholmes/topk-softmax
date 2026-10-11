"""The operator on a CUDA GPU, checked by comparing it with the CPU result. Skipped without CUDA."""
import pytest
import torch

import softk
from _reference import assert_close, mask_sum_tol, randn, ref_hard, ref_soft, separated

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

MODES = ["soft", "hard", "st"]
# output tolerance when comparing devices: (32-bit, 64-bit)
TOL = {torch.float32: 1e-6, torch.float64: 1e-12}


def _grad(z, k, tau, mode="soft", dim=-1):
    w = randn(*z.shape, dtype=z.dtype, seed=1).to(z.device)
    x = z.clone().requires_grad_()
    (softk.soft_topk_softmax(x, k, tau, dim=dim, mode=mode) * w).sum().backward()
    return x.grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n,k,tau", [(7, 3, 1.0), (64, 8, 0.3), (1000, 100, 0.05), (1000, 2.5, 1.0)])
def test_output_matches_cpu(dtype, mode, n, k, tau):
    if mode != "soft" and int(k) != k:
        pytest.skip("hard and st need an integer k")
    z = randn(6, n, dtype=dtype)
    out = softk.soft_topk_softmax(z.cuda(), k, tau, mode=mode)
    assert out.device.type == "cuda" and out.dtype == dtype
    assert_close(out.cpu(), softk.soft_topk_softmax(z, k, tau, mode=mode), TOL[dtype])


@pytest.mark.parametrize("n,k,tau", [(7, 3, 1.0), (64, 8, 0.3), (1000, 100, 0.05)])
def test_output_matches_the_independent_reference(n, k, tau):
    z = randn(6, n)
    assert_close(softk.soft_topk_softmax(z.cuda(), k, tau).cpu(), ref_soft(z, k, tau), 1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n,k,tau", [(7, 3, 1.0), (64, 8, 0.3), (1000, 100, 0.05)])
def test_mask_matches_cpu_and_sums_to_k(dtype, n, k, tau):
    z = randn(6, n, dtype=dtype)
    m = softk.soft_topk_mask(z.cuda(), k, tau)
    assert m.device.type == "cuda"
    assert_close(m.cpu(), softk.soft_topk_mask(z, k, tau), 10 * TOL[dtype])
    assert_close(m.sum(-1).cpu(), torch.full((6,), float(k), dtype=torch.float64), mask_sum_tol(k, dtype))


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.float64, 1e-11)])
@pytest.mark.parametrize("mode", ["soft", "st"])
@pytest.mark.parametrize("n,k,tau", [(7, 3, 1.0), (64, 8, 0.3), (1000, 100, 0.05)])
def test_gradient_matches_cpu(dtype, atol, mode, n, k, tau):
    z = randn(6, n, dtype=dtype)
    cpu = _grad(z, k, tau, mode)
    assert cpu.abs().max().item() > 1e-6
    assert_close(_grad(z.cuda(), k, tau, mode).cpu(), cpu, atol)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_any_dim_and_non_contiguous_input(dim):
    z = randn(5, 6, 7)
    assert_close(softk.soft_topk_softmax(z.cuda(), 2, 0.5, dim=dim).cpu(),
                 softk.soft_topk_softmax(z, 2, 0.5, dim=dim), 1e-12)
    zt = randn(7, 6).cuda().t()
    assert not zt.is_contiguous()
    assert_close(softk.soft_topk_softmax(zt, 2, 0.5).cpu(), ref_soft(zt.cpu(), 2, 0.5), 1e-10)


def test_k_at_least_n_gives_softmax():
    z = randn(5, 4).cuda()
    assert_close(softk.soft_topk_softmax(z, 9, 0.5).cpu(), torch.softmax(z, -1).cpu(), 1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tau", [1.0, 1e-3, 1e-6])
def test_tie_at_the_kth_position(dtype, tau):
    z = torch.tensor([[3.0, 2.0, 1.0, 1.0, 1.0, 0.0, -1.0, -2.0]], dtype=dtype, device="cuda")
    m = softk.soft_topk_mask(z, 3, tau)
    assert torch.isfinite(m).all()
    assert_close(m.sum(-1).cpu(), torch.tensor([3.0]), 1e-5)
    assert_close(m[:, 2:5].cpu(), m[:, 2:3].expand(-1, 3).cpu(), 1e-6)
    assert torch.isfinite(_grad(z, 3, tau)).all()


def test_small_tau_wide_row_is_finite():
    z = randn(2, 32768, dtype=torch.float32).cuda()
    p = softk.soft_topk_softmax(z, 8, 1e-4)
    assert torch.isfinite(p).all()
    assert torch.isfinite(_grad(z, 8, 1e-4)).all()
    assert_close(p.sum(-1).cpu(), torch.ones(2), 1e-4)


def test_small_tau_selects_the_top_k():
    z = separated(4, 64, gap=0.1, dtype=torch.float32)
    assert_close(softk.soft_topk_softmax(z.cuda(), 8, 1e-4).cpu(), ref_hard(z, 8), 1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", MODES)
def test_16_bit_inputs(dtype, mode):
    z = randn(4, 64, dtype=dtype).cuda()
    p = softk.soft_topk_softmax(z, 8, 0.5, mode=mode)
    assert p.dtype == dtype
    assert torch.isfinite(p).all()
    assert_close(p.float().sum(-1).cpu(), torch.ones(4), 1e-2)


def test_autocast():
    z = randn(4, 64, dtype=torch.float32).cuda().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = softk.soft_topk_softmax(z, 8, 0.5)
    assert torch.isfinite(p).all()
    assert_close(p.float().cpu(), softk.soft_topk_softmax(z.detach().cpu(), 8, 0.5), 1e-2)
    p.float().square().sum().backward()
    assert torch.isfinite(z.grad).all()


def test_module_moves_to_cuda_and_follows_the_scheduler():
    z = randn(5, 12, dtype=torch.float32)
    layer = softk.SoftTopKSoftmax(3, tau=5.0, track_stats=True).cuda()
    assert layer.tau.device.type == "cuda"
    sched = softk.TauScheduler(layer, 5.0, 0.5, 10, hold_frac=0.0)
    for _ in range(10):
        sched.step()
    assert layer.tau.device.type == "cuda"
    assert_close(layer(z.cuda()).cpu(), softk.soft_topk_softmax(z, 3, 0.5), 1e-6)
    assert set(layer.stats) == {"mass_outside_mean", "mass_outside_p99", "effective_support", "sum_v"}


def test_stats_functions_run_on_cuda():
    z = randn(6, 10).cuda()
    p = torch.softmax(z, -1)
    assert_close(softk.stats.mass_outside_topk(p, 3).cpu(), softk.stats.mass_outside_topk(p.cpu(), 3), 1e-12)
    assert_close(softk.stats.effective_support(p).cpu(), softk.stats.effective_support(p.cpu()), 1e-10)
    assert_close(softk.stats.topk_churn(z, z.flip(-1), 3).cpu(), softk.stats.topk_churn(z.cpu(), z.cpu().flip(-1), 3), 0.0)


@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES)
def test_compiles_on_cuda_without_graph_breaks(mode):
    torch._dynamo.reset()
    z = randn(4, 64, dtype=torch.float32).cuda()
    eager = softk.soft_topk_softmax(z, 8, 0.5, mode=mode)
    assert_close(torch.compile(softk.soft_topk_softmax)(z, 8, 0.5, mode=mode).cpu(), eager.cpu(), 1e-6)
    explanation = torch._dynamo.explain(softk.soft_topk_softmax)(z, 8, 0.5, mode=mode)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    torch._dynamo.reset()
