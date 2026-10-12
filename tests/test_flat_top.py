"""flat_top=True (the default): the mask is computed from scores clamped at the floor(k)-th largest, so the top
floor(k) entries share one mask value and keep their softmax ratios at every tau."""
import inspect

import pytest
import torch

import topk_softmax
from _reference import assert_close, hard_cases, mask_sum_tol, randn, ref_hard, ref_mask, ref_soft, separated

MODES = ["soft", "hard", "st"]
CASES = [(7, 3, 1.0), (7, 1, 0.3), (64, 8, 0.3), (64, 63, 1.0), (1000, 100, 0.05), (7, 2.5, 0.5), (7, 0.5, 0.5)]


def soft(z, k, tau, **kwargs):
    return topk_softmax.soft_topk_softmax(z, k, tau, flat_top=True, **kwargs)


def mask(z, k, tau, **kwargs):
    return topk_softmax.soft_topk_mask(z, k, tau, flat_top=True, **kwargs)


def on_cuda(test):
    return pytest.mark.gpu(pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")(test))


# ---- the definition ----

@pytest.mark.parametrize("fn", ["soft_topk_softmax", "soft_topk_mask", "SoftTopKSoftmax"])
def test_flat_top_defaults_to_true(fn):
    assert inspect.signature(getattr(topk_softmax, fn)).parameters["flat_top"].default is True


@pytest.mark.parametrize("mode", MODES)
def test_true_is_the_default(mode):
    z = randn(5, 12)
    assert torch.equal(soft(z, 3, 0.5, mode=mode), topk_softmax.soft_topk_softmax(z, 3, 0.5, mode=mode))
    assert torch.equal(mask(z, 3, 0.5), topk_softmax.soft_topk_mask(z, 3, 0.5))


@pytest.mark.parametrize("n,k,tau", CASES)
def test_matches_the_reference(n, k, tau):
    z = randn(6, n)
    assert_close(mask(z, k, tau), ref_mask(z, k, tau, flat_top=True), 1e-10)
    assert_close(soft(z, k, tau), ref_soft(z, k, tau, flat_top=True), 1e-10)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_any_dim(dim):
    z = randn(5, 6, 7)
    assert_close(soft(z, 2, 0.5, dim=dim), ref_soft(z, 2, 0.5, dim=dim, flat_top=True), 1e-10)
    assert_close(mask(z, 2, 0.5, dim=dim), ref_mask(z, 2, 0.5, dim=dim, flat_top=True), 1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n,k,tau", CASES)
def test_mask_sums_to_k(dtype, n, k, tau):
    m = mask(randn(6, n, dtype=dtype), k, tau)
    assert_close(m.double().sum(-1), torch.full((6,), float(k), dtype=torch.float64), mask_sum_tol(k, dtype))


@pytest.mark.parametrize("tau", [10.0, 1.0, 0.3, 0.05])
def test_top_k_share_one_mask_value(tau):
    z = randn(6, 20)
    m = mask(z, 5, tau)
    top = m.gather(-1, z.topk(5, -1).indices)
    assert torch.equal(top, top[:, :1].expand(-1, 5))
    assert (m <= top[:, :1]).all()


@pytest.mark.parametrize("tau", [100.0, 10.0, 1.0, 0.3, 0.05, 1e-3])
def test_top_k_keep_their_softmax_ratios(tau):
    z = randn(6, 20)
    idx = z.topk(5, -1).indices
    p = soft(z, 5, tau).gather(-1, idx)
    expected = torch.softmax(z.gather(-1, idx), -1)
    assert_close(p / p.sum(-1, keepdim=True), expected, 1e-12)


def test_the_original_operator_distorts_those_ratios():
    # the reason the option exists: without it the largest score is favored at intermediate tau
    z = torch.tensor([[2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5]], dtype=torch.float64)
    p = topk_softmax.soft_topk_softmax(z, 3, 0.5, flat_top=False)
    assert (p[0, 0] / p[0, 2]).item() > 1.05 * torch.exp(z[0, 0] - z[0, 2]).item()
    q = soft(z, 3, 0.5)
    assert (q[0, 0] / q[0, 2]).item() == pytest.approx(torch.exp(z[0, 0] - z[0, 2]).item(), rel=1e-12)


# ---- limits ----

def test_large_tau_gives_softmax():
    z = randn(6, 20)
    assert_close(soft(z, 5, 1e4), torch.softmax(z, -1), 1e-4)


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.float64, 1e-6)])
def test_small_tau_gives_exact_top_k(dtype, atol):
    z = separated(4, 64, gap=0.1, dtype=dtype)
    assert_close(soft(z, 8, 1e-3), ref_hard(z, 8), atol)


def test_k_at_least_n_gives_softmax():
    z = randn(5, 4)
    assert_close(soft(z, 9, 0.5), torch.softmax(z, -1), 1e-12)


# ---- non-integer k ----

def test_non_integer_k_clamps_at_floor_k():
    z = randn(6, 12)
    m = mask(z, 2.5, 0.5)
    order = z.argsort(-1, descending=True)
    ranked = m.gather(-1, order)
    assert torch.equal(ranked[:, 0], ranked[:, 1])
    assert (ranked[:, 2] < ranked[:, 1]).all()


def test_non_integer_k_small_tau_limit():
    z = torch.tensor([[2.0, 1.5, 1.0, 0.5, 0.0, -0.5]], dtype=torch.float64)
    assert_close(mask(z, 2.5, 1e-3), torch.tensor([[1.0, 1.0, 0.5, 0.0, 0.0, 0.0]], dtype=torch.float64), 1e-9)


def test_k_below_one_is_not_clamped():
    z = randn(6, 12)
    assert torch.equal(mask(z, 0.5, 0.5), topk_softmax.soft_topk_mask(z, 0.5, 0.5))
    assert torch.equal(soft(z, 0.5, 0.5), topk_softmax.soft_topk_softmax(z, 0.5, 0.5))


# ---- monotonicity in tau ----

def test_top_k_entries_are_monotone_in_tau():
    # Observed on these rows, not proven: as tau falls, no top-k probability ever decreases.
    taus = torch.logspace(2, -3, 60, dtype=torch.float64)
    for seed in range(20):
        n, k = [(8, 3), (32, 1), (64, 16), (16, 15)][seed % 4]
        z = randn(n, seed=seed) * [0.5, 1.0, 3.0][seed % 3]
        p = torch.stack([soft(z, k, float(t)) for t in taus])[:, z.topk(k).indices]
        assert (p[1:] - p[:-1] >= -1e-12).all(), seed


def test_the_original_operator_is_not_monotone():
    z = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5], dtype=torch.float64)
    taus = torch.logspace(2, -3, 60, dtype=torch.float64)
    p1 = torch.stack([topk_softmax.soft_topk_softmax(z, 3, float(t), flat_top=False) for t in taus])[:, 0]
    assert p1.max().item() > p1[-1].item() + 0.01


# ---- gradients ----

@pytest.mark.parametrize("n,k,tau", [(7, 3, 1.0), (7, 1, 0.5), (12, 4, 0.2), (7, 2.5, 0.5)])
def test_gradcheck(n, k, tau):
    z = separated(2, n, gap=0.4).requires_grad_()  # no ties, so the clamp is differentiable here
    assert torch.autograd.gradcheck(lambda x: soft(x, k, tau), (z,), atol=1e-6)
    assert torch.autograd.gradcheck(lambda x: mask(x, k, tau), (z,), atol=1e-6)


def test_mask_ignores_how_far_the_top_entries_exceed_the_kth():
    z = separated(1, 10, gap=0.4)
    order = z[0].argsort(descending=True)
    jac = torch.autograd.functional.jacobian(lambda x: mask(x, 4, 0.5), z)[0, :, 0, :]
    assert_close(jac[:, order[:3]], torch.zeros(10, 3), 0.0)   # the 3 largest: no effect on any m
    assert jac[:, order[3]].abs().max().item() > 1e-3          # the 4th largest carries the gradient
    raised = z.clone()
    raised[0, order[0]] += 5.0
    assert torch.equal(mask(raised, 4, 0.5), mask(z, 4, 0.5))


@pytest.mark.parametrize("tau", [1.0, 1e-3])
def test_ties_at_the_kth_position_are_finite_and_symmetric(tau):
    z = torch.tensor([[3.0, 2.0, 1.0, 1.0, 1.0, 0.0, -1.0, -2.0]], dtype=torch.float64, requires_grad=True)
    m = mask(z, 3, tau)
    assert_close(m[:, 2:5], m[:, 2:3].expand(-1, 3), 1e-12)
    assert torch.equal(m[:, 0], m[:, 2]) and torch.equal(m[:, 1], m[:, 2])
    (soft(z, 3, tau) * randn(1, 8, seed=1)).sum().backward()
    assert torch.isfinite(z.grad).all()


# ---- invariances ----

def test_shift_invariance_and_permutation_equivariance():
    z = randn(5, 12)
    assert_close(soft(z + 7.5, 3, 0.5), soft(z, 3, 0.5), 1e-12)
    perm = torch.randperm(12, generator=torch.Generator().manual_seed(0))
    assert_close(soft(z[:, perm], 3, 0.5), soft(z, 3, 0.5)[:, perm], 1e-12)


# ---- modes, dtypes, module, compile ----

def test_hard_ignores_flat_top_and_st_uses_its_gradient():
    z = randn(5, 12)
    w = randn(5, 12, seed=1)
    assert torch.equal(soft(z, 3, 0.5, mode="hard"), topk_softmax.soft_topk_softmax(z, 3, 0.5, mode="hard"))
    assert torch.equal(soft(z, 3, 0.5, mode="st"), soft(z, 3, 0.5, mode="hard"))
    grads = []
    for mode in ("st", "soft"):
        x = z.clone().requires_grad_()
        (soft(x, 3, 0.5, mode=mode) * w).sum().backward()
        grads.append(x.grad)
    assert_close(grads[0], grads[1], 1e-12)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_16_bit_inputs(dtype):
    p = soft(randn(4, 64, dtype=dtype), 8, 0.5)
    assert p.dtype == dtype
    assert torch.isfinite(p).all()
    assert_close(p.float().sum(-1), torch.ones(4), 1e-2)


def test_module_passes_flat_top_on():
    z = randn(5, 12)
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.5, flat_top=True, track_stats=True)
    assert layer.flat_top is True
    assert torch.equal(layer(z), soft(z, 3, 0.5))
    m = mask(z, 3, 0.5)
    assert float(layer.stats["sum_v"]) == pytest.approx(float((m * (1 - m)).sum(-1).mean()), rel=1e-6)
    assert topk_softmax.SoftTopKSoftmax(3).flat_top is True


@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES)
def test_compiles_without_graph_breaks(mode):
    torch._dynamo.reset()
    z = randn(4, 64, dtype=torch.float32)
    assert_close(torch.compile(topk_softmax.soft_topk_softmax)(z, 8, 0.5, mode=mode, flat_top=True),
                 soft(z, 8, 0.5, mode=mode), 1e-6)
    explanation = torch._dynamo.explain(topk_softmax.soft_topk_softmax)(z, 8, 0.5, mode=mode, flat_top=True)
    assert explanation.graph_break_count == 0, explanation.break_reasons
    torch._dynamo.reset()


# ---- on a CUDA GPU: the Triton kernels agree with the PyTorch code ----

def _grad(z, k, tau, backend, mode="soft", dim=-1):
    w = randn(*z.shape, dtype=z.dtype, seed=1).to(z.device)
    x = z.clone().requires_grad_()
    (soft(x, k, tau, dim=dim, mode=mode, backend=backend) * w).sum().backward()
    return x.grad


@on_cuda
@pytest.mark.parametrize("n,k", [(2, 1), (7, 3), (64, 8), (1000, 100), (1000, 999), (1000, 2.5), (1000, 0.5),
                                 (16384, 2048), (131072, 64)])
@pytest.mark.parametrize("tau", [10.0, 1.0, 0.05])
def test_triton_matches_eager(n, k, tau):
    z = randn(3, n, dtype=torch.float32).cuda()
    assert_close(soft(z, k, tau, backend="triton").cpu(), soft(z, k, tau, backend="eager").cpu(), 1e-6)
    assert_close(mask(z, k, tau, backend="triton").cpu(), mask(z, k, tau, backend="eager").cpu(), 1e-5)
    assert_close(_grad(z, k, tau, "triton").cpu(), _grad(z, k, tau, "eager").cpu(), 1e-5)


@on_cuda
def test_triton_matches_the_reference():
    z = randn(6, 64, dtype=torch.float32)
    assert_close(soft(z.cuda(), 8, 0.3, backend="triton").cpu(), ref_soft(z, 8, 0.3, flat_top=True), 1e-6)


@on_cuda
@pytest.mark.parametrize("mode", ["soft", "st"])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_triton_modes_and_dims(mode, dim):
    z = randn(5, 6, 7, dtype=torch.float32).cuda()
    assert_close(soft(z, 2, 0.5, dim=dim, mode=mode, backend="triton").cpu(),
                 soft(z, 2, 0.5, dim=dim, mode=mode, backend="eager").cpu(), 1e-6)
    assert_close(_grad(z, 2, 0.5, "triton", mode, dim).cpu(), _grad(z, 2, 0.5, "eager", mode, dim).cpu(), 1e-5)


@on_cuda
def test_triton_hard_cases():
    for name, z, k, tau in hard_cases(torch.float32):
        z = z.cuda()
        p = soft(z, k, tau, backend="triton")
        assert torch.isfinite(p).all(), name
        assert (p.cpu() - soft(z, k, tau, backend="eager").cpu()).abs().max().item() <= 1e-5, name
        m = mask(z, k, tau, backend="triton")
        assert (m.double().sum(-1) - k).abs().max().item() <= mask_sum_tol(k, torch.float32), name
        assert torch.isfinite(_grad(z, k, tau, "triton")).all(), name


@on_cuda
@pytest.mark.slow
def test_triton_under_torch_compile():
    torch._dynamo.reset()
    z = randn(4, 64, dtype=torch.float32).cuda()
    fn = torch.compile(topk_softmax.soft_topk_softmax)
    assert_close(fn(z, 8, 0.5, flat_top=True, backend="triton").cpu(), soft(z, 8, 0.5, backend="eager").cpu(), 1e-6)
    explanation = torch._dynamo.explain(topk_softmax.soft_topk_softmax)(z, 8, 0.5, flat_top=True, backend="triton")
    assert explanation.graph_break_count == 0, explanation.break_reasons
    torch._dynamo.reset()


# ---- the kernel finds the k-th largest score itself ----

def _selection_cases():
    tie = torch.tensor([[3.0, 2.0, 1.0, 1.0, 1.0, 0.0, -1.0, -2.0]])
    high = randn(4, 1000).float()
    high[:, 0] = 1e4
    low = randn(4, 1000).float()
    low[:, 0] = -1e4
    ramp = torch.arange(64.0)[None]
    cases = [("floor(k) = 1", randn(4, 64).float(), 1), ("floor(k) = n - 1", randn(4, 64).float(), 63),
             ("constant row", torch.ones(2, 16), 5), ("n = 2", randn(4, 2).float(), 1),
             ("high outlier", high, 8), ("low outlier", low, 8), ("high outlier, k = 1", high, 1),
             ("ascending", ramp, 10), ("descending", ramp.flip(-1), 10),
             ("non-integer k", randn(4, 64).float(), 7.5), ("k below 1", randn(4, 64).float(), 0.5),
             ("two values", torch.tensor([[1.0, 0.0] * 8]), 3), ("two values, k past the split", torch.tensor([[1.0, 0.0] * 8]), 11)]
    cases += [(f"ties at the k-th value, k = {k}", tie, k) for k in (2, 3, 4, 5, 6)]
    return cases


@on_cuda
@pytest.mark.parametrize("tau", [1.0, 0.05])
def test_triton_selection_edge_cases(tau):
    for name, z, k in _selection_cases():
        z = z.cuda()
        assert_close(mask(z, k, tau, backend="triton").cpu(), mask(z, k, tau, backend="eager").cpu(), 1e-5)
        assert_close(soft(z, k, tau, backend="triton").cpu(), soft(z, k, tau, backend="eager").cpu(), 1e-6)
        assert torch.isfinite(_grad(z, k, tau, "triton")).all(), name


@on_cuda
@pytest.mark.parametrize("k", [3, 4, 5])
@pytest.mark.parametrize("fn", [soft, mask])
def test_triton_gradient_at_ties(fn, k):
    # Scores 2, 3 and 4 tie at the k-th value. The kernel splits the gradient routed through the
    # clamp equally among them; the eager code sends it to one of them. The totals agree.
    z = torch.tensor([[3.0, 2.0, 1.0, 1.0, 1.0, 0.0, -1.0, -2.0]], device="cuda")
    weights = torch.tensor([[0.3, -1.1, 0.7, 0.7, 0.7, 0.2, -0.4, 0.9]], device="cuda")
    grads = {}
    for backend in ("triton", "eager"):
        x = z.clone().requires_grad_()
        (fn(x, k, 0.5, backend=backend) * weights).sum().backward()
        grads[backend] = x.grad[0].cpu()
    assert torch.isfinite(grads["triton"]).all()
    assert_close(grads["triton"][2:5], grads["triton"][2:3].expand(3), 1e-6)
    assert grads["triton"][2:5].sum().item() == pytest.approx(grads["eager"][2:5].sum().item(), abs=1e-5)
    untied = [0, 1, 5, 6, 7]
    assert_close(grads["triton"][untied], grads["eager"][untied], 1e-5)


# ---- flat_top=False: the mask is computed from the scores themselves ----

def plain_soft(z, k, tau, **kwargs):
    return topk_softmax.soft_topk_softmax(z, k, tau, flat_top=False, **kwargs)


def plain_mask(z, k, tau, **kwargs):
    return topk_softmax.soft_topk_mask(z, k, tau, flat_top=False, **kwargs)


@pytest.mark.parametrize("n,k,tau", CASES)
def test_false_matches_the_reference(n, k, tau):
    z = randn(6, n)
    assert_close(plain_mask(z, k, tau), ref_mask(z, k, tau, flat_top=False), 1e-10)
    assert_close(plain_soft(z, k, tau), ref_soft(z, k, tau, flat_top=False), 1e-10)
    assert (plain_soft(z, k, tau) - soft(z, k, tau)).abs().max().item() > 1e-6 or k < 2


def test_false_module_matches_the_function():
    z = randn(5, 12)
    layer = topk_softmax.SoftTopKSoftmax(3, tau=0.5, flat_top=False)
    assert torch.equal(layer(z), plain_soft(z, 3, 0.5))


@pytest.mark.parametrize("fn", [plain_soft, plain_mask])
def test_false_gradcheck(fn):
    z = randn(3, 7).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: fn(x, 3, 0.5), (z,), eps=1e-6, atol=1e-5)


def _plain_grad(z, k, tau, backend):
    w = randn(*z.shape, dtype=z.dtype, seed=1).to(z.device)
    x = z.clone().requires_grad_()
    (plain_soft(x, k, tau, backend=backend) * w).sum().backward()
    return x.grad


@on_cuda
@pytest.mark.parametrize("n,k", [(2, 1), (7, 3), (64, 8), (1000, 100), (1000, 999), (1000, 2.5), (16384, 2048)])
@pytest.mark.parametrize("tau", [10.0, 1.0, 0.05])
def test_false_triton_matches_eager(n, k, tau):
    z = randn(3, n, dtype=torch.float32).cuda()
    assert_close(plain_soft(z, k, tau, backend="triton").cpu(), plain_soft(z, k, tau, backend="eager").cpu(), 1e-6)
    if k < n - 1:  # at k = n - 1 the one small mask entry is set by rounding in a sum near n
        assert_close(plain_mask(z, k, tau, backend="triton").cpu(), plain_mask(z, k, tau, backend="eager").cpu(), 1e-5)
    assert_close(_plain_grad(z, k, tau, "triton").cpu(), _plain_grad(z, k, tau, "eager").cpu(), 1e-5)


@on_cuda
def test_false_triton_hard_cases():
    for name, z, k, tau in hard_cases(torch.float32):
        z = z.cuda()
        p = plain_soft(z, k, tau, backend="triton")
        assert (p.cpu() - plain_soft(z, k, tau, backend="eager").cpu()).abs().max().item() <= 1e-5, name
        m = plain_mask(z, k, tau, backend="triton")
        assert (m.double().sum(-1) - k).abs().max().item() <= mask_sum_tol(k, torch.float32), name
        assert torch.isfinite(_plain_grad(z, k, tau, "triton")).all(), name


# ---- scores spread over many orders of magnitude ----

def _wide_cases():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(4, 16384, generator=g)
    extremes = z.clone()
    extremes[:, 0], extremes[:, 1] = -3e38, 3e38
    return [("exp(18 z)", torch.exp(18 * z)), ("exp(5 z)", torch.exp(5 * z)), ("-exp(18 z)", -torch.exp(18 * z)),
            ("powers of two", (2.0 ** -torch.arange(140.0))[None]), ("z cubed", z ** 3),
            ("half at -1e9", torch.where(torch.rand(4, 16384, generator=g) < 0.5, torch.tensor(-1e9), z)),
            ("offset 1e6", z + 1e6), ("near the largest float", extremes),
            ("tiny values", 1e-30 * torch.exp(10 * z)), ("signed wide", z * torch.exp(10 * z.roll(1, -1)))]


@on_cuda
def test_triton_finds_the_kth_score_for_wide_ranges():
    from topk_softmax import _triton

    assert _triton.available()
    for name, z in _wide_cases():
        n = z.shape[-1]
        assert torch.isfinite(z).all(), name
        for k in sorted({1, 8, 70, n // 8, n // 2, n - 1}):
            _, state = _triton._ops[0](z.cuda(), None, 1.0, float(k), k, 20, False)
            want = z.topk(k, -1).values[:, -1]
            assert torch.equal(state[:, 0].cpu(), want), f"{name}, k = {k}"


# ---- long rows: from 32768 scores, or 8192 for small k, the kernel finds the k-th score during
# its first pass ----

LONG = [(4096, 64), (8192, 32), (16384, 32), (32768, 16), (65536, 8), (262144, 4), (2097152, 2)]


def _long_families(rows, n):
    z = torch.randn(rows, n, generator=torch.Generator().manual_seed(0))
    spot = torch.arange(n)
    return [("gaussian", z), ("ties", (z * 10).round() / 10), ("two values", (z > 1.0).float()),
            ("constant", torch.ones(rows, n)), ("heavy tails", z ** 3), ("wide range", torch.exp(18 * z)),
            ("sorted", z.sort(-1).values), ("descending", z.sort(-1, descending=True).values),
            ("stride pattern", z + 3.0 * (spot % 64 == 0)), ("shifted halves", z + 3.0 * (spot >= n // 2))]


def _kth_scores(z, k):
    """The k-th largest score of each row as the forward kernel found it."""
    from topk_softmax import _triton

    assert _triton.available()
    _, state = _triton._ops[0](z, None, 1.0, float(k), k, 20, False)
    return state[:, 0]


def _assert_kth_scores(rows, n):
    for name, z in _long_families(rows, n):
        z = z.cuda()
        for k in sorted({100, 1000, n // 32, n // 8, n // 2}):
            assert torch.equal(_kth_scores(z, k), z.topk(k, -1).values[:, -1]), f"{name}, n = {n}, k = {k}"


@on_cuda
@pytest.mark.slow
@pytest.mark.parametrize("n,rows", LONG)
def test_triton_finds_the_kth_score_in_long_rows(n, rows):
    _assert_kth_scores(rows, n)


@on_cuda
@pytest.mark.slow
@pytest.mark.parametrize("sizes", [(16, 16), (16, 256), (256, 16)])
def test_triton_finds_the_kth_score_when_the_sample_or_buffer_is_too_small(sizes, monkeypatch):
    # a tiny sample makes the pivots miss and a tiny buffer overflows, so rows take the fallback
    from topk_softmax import _triton

    monkeypatch.setattr(_triton, "_oneread_sizes", lambda n, k: sizes)
    monkeypatch.setattr(_triton, "_forward_plan", lambda n, flat_k: (flat_k >= 1, 4096, 8))
    _assert_kth_scores(64, 4096)
    _assert_kth_scores(8, 32768)


@on_cuda
@pytest.mark.slow
@pytest.mark.parametrize("n,rows", LONG)
@pytest.mark.parametrize("tau", [1.0, 0.05])
def test_triton_matches_eager_in_long_rows(n, rows, tau):
    z = randn(rows, n, dtype=torch.float32).cuda()
    for k in (100, n // 8):
        assert_close(soft(z, k, tau, backend="triton").cpu(), soft(z, k, tau, backend="eager").cpu(), 1e-6)
        m = mask(z, k, tau, backend="triton")
        assert_close(m.cpu(), mask(z, k, tau, backend="eager").cpu(), 1e-5)
        assert (m.double().sum(-1) - k).abs().max().item() <= mask_sum_tol(k, torch.float32), (n, k)
        # Scores that tie at the k-th value share the routed gradient in the kernel and not in the
        # PyTorch code, so those entries are compared by their sum per row.
        tied = z == z.topk(k, -1).values[:, -1:]
        triton, eager = _grad(z, k, tau, "triton"), _grad(z, k, tau, "eager")
        assert_close(triton.masked_fill(tied, 0).cpu(), eager.masked_fill(tied, 0).cpu(), 1e-5)
        assert_close((triton * tied).sum(-1).cpu(), (eager * tied).sum(-1).cpu(), 1e-5)


@on_cuda
@pytest.mark.slow
def test_triton_long_rows_under_torch_compile():
    torch._dynamo.reset()
    z = randn(4, 32768, dtype=torch.float32).cuda()
    fn = torch.compile(topk_softmax.soft_topk_softmax)
    assert_close(fn(z, 512, 0.5, backend="triton").cpu(), soft(z, 512, 0.5, backend="eager").cpu(), 1e-6)
    explanation = torch._dynamo.explain(topk_softmax.soft_topk_softmax)(z, 512, 0.5, backend="triton")
    assert explanation.graph_break_count == 0, explanation.break_reasons
    torch._dynamo.reset()
