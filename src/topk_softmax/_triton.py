"""Fused GPU kernels for the soft top-k mask and softmax, written in Triton.

One kernel program handles one row. It reads the row in blocks, finds the threshold, and writes the
output; nothing of the row's size is stored in between. Unlike the PyTorch implementation, the
loops stop as soon as the row has converged, which is where the speed comes from.

triton is imported on first use, so importing topk_softmax does not need it.
"""
import math

import torch

# Launch settings chosen by timing on one GPU (benchmarks/results/m3.json). A row wider than the
# block is read in several steps. Threads come in groups of 32.
_FORWARD_BLOCK, _BACKWARD_BLOCK = 4096, 16384


def _forward_plan(n, flat_k):
    """How the forward kernel runs rows of n scores: whether it finds the flat_k-th largest score
    during its first pass, its largest block size, and its thread groups (warps) per row.

    The values are the fastest measured on one GPU (benchmarks/sweep.py) at each power of two from
    512 to 2,097,152; other lengths use the nearest power. Thread groups matter most from about
    32,768 scores: more groups per row leave fewer rows running together, and a row then stays in
    the GPU's cache between passes.
    """
    e = round(math.log2(n)) if n > 1 else 0
    if flat_k < 1:  # no k-th score to find
        groups = 1 if e <= 9 else 2 if e == 10 else 4 if e <= 12 else 8 if e <= 14 else 16 if e <= 16 else 32 if e == 17 else 8
        return False, _FORWARD_BLOCK, groups
    if e >= 15:
        return True, _FORWARD_BLOCK, 16 if e in (17, 18) else 8
    if e >= 13 and flat_k <= 256:  # shorter rows gain from the single pass only when k is small
        return True, 1024, 2
    return False, _FORWARD_BLOCK, 1 if e <= 10 else 2 if e == 11 else 4 if e == 12 else 8


def _backward_thread_groups(n):
    return 4 if n <= 1024 else 16


_MAX_POLISH = 6       # Newton steps after the bisection; most rows need 2
_MAX_SELECT = 96      # passes allowed for finding the k-th largest score; 24 + 2 * 33 always suffice
_BRACKET_WIDTH = 0.25  # bisection stops once the bracket is narrower than this many tau
_TAU_FLOOR = 1e-4     # as in functional.py, in units of the row standard deviation

_ops = None  # (forward op, backward op), or False if triton is not usable


def available():
    if _ops is None:
        _load()
    return _ops is not False


def faster_for(n):
    """Whether backend="auto" should pick these kernels for rows of length n.

    They beat the uncompiled PyTorch code at every length measured. Under torch.compile the
    compiled PyTorch code is faster for rows shorter than about 128.
    """
    return n >= 128 or not torch.compiler.is_compiling()


def _oneread_sizes(n, k):
    """Sample and buffer sizes (powers of two) for finding the k-th largest of n scores in one pass.

    The scores between the two pivots number about 6 n sqrt(p (1 - p) / S) for p = k/n and a
    sample of S, so the S that balances the sample's cost with theirs is about
    (3 n sqrt(p (1 - p)))^(2/3). The buffer holds their expected number plus four standard
    deviations.
    """
    def pow2_ceil(x):
        return 1 << max(int(x) - 1, 0).bit_length()

    p = k / n
    sample = pow2_ceil((3 * n * math.sqrt(max(p * (1 - p), 1 / n))) ** (2 / 3))
    sample = max(16, min(max(sample, 256), pow2_ceil(n // 4 + 1) // 2))
    r = k * sample / n
    d = 3 * math.sqrt(max(r * (1 - r / sample), 0)) + 1  # as in the kernel
    lowest = min(int(r + d) + 2, sample + 1)
    highest = int(r - d - 1) + 1 if r - d - 1 >= 0 else 0
    gaps = lowest - highest
    cap = pow2_ceil(n / sample * (gaps + 4 * math.sqrt(gaps) + 2))
    return sample, max(min(cap, pow2_ceil(n)), sample)


def run(x, k, tau, iters, want_mask, flat_top=True):
    """The mask (want_mask=True) or the masked softmax along the last dim of a 32-bit CUDA tensor."""
    n = x.shape[-1]
    z = x.contiguous().view(-1, n)
    if isinstance(tau, torch.Tensor):
        tau_tensor, tau_number = tau.detach().to(device=z.device, dtype=torch.float32), 0.0
    else:
        tau_tensor, tau_number = None, float(tau)
    out, _ = _ops[0](z, tau_tensor, tau_number, float(k), math.floor(k) if flat_top else 0, int(iters),
                     bool(want_mask))
    return out.view(x.shape)


def _load():
    global _ops
    try:
        import triton
        from torch.library import triton_op, wrap_triton

        from ._kernels import backward_kernel, forward_kernel
    except ImportError:
        _ops = False
        return

    def block(n, largest):
        return min(largest, max(16, triton.next_power_of_2(n)))

    @triton_op("topk_softmax::soft_topk_forward", mutates_args={})
    def forward_op(z: torch.Tensor, tau_tensor: torch.Tensor | None, tau_number: float, k: float, flat_k: int,
                   max_iters: int, want_mask: bool) -> tuple[torch.Tensor, torch.Tensor]:
        # flat_k is floor(k) for flat_top, which the kernel then finds the flat_k-th largest score
        # for, or 0 for the original operator
        rows, n = z.shape
        out = torch.empty_like(z)
        state = z.new_empty(rows, 4)  # per row: center (mean, or c for flat_top), threshold, floored tau, Newton shift
        # long rows find the flat_k-th largest score in one pass, using a sample and a buffer
        oneread, largest_block, groups = _forward_plan(n, flat_k)
        sample, cap = _oneread_sizes(n, flat_k) if oneread else (0, 1)
        scratch = z.new_empty(rows, cap) if sample else z
        wrap_triton(forward_kernel)[(rows,)](
            z, out, state, z if tau_tensor is None else tau_tensor, scratch, tau_number, k,
            math.log(k / (n - k)), flat_k, n, max_iters,
            WANT_MASK=want_mask, TAU_IS_TENSOR=tau_tensor is not None,
            MAX_POLISH=_MAX_POLISH, WIDTH=_BRACKET_WIDTH, FLOOR=_TAU_FLOOR,
            FLAT=flat_k >= 1, MAX_SELECT=_MAX_SELECT,
            S=sample, SC=min(max(sample, 1), 1024), C=cap, CB=min(cap, 4096),
            BLOCK=block(n, largest_block), num_warps=groups)
        return out, state

    @triton_op("topk_softmax::soft_topk_backward", mutates_args={})
    def backward_op(z: torch.Tensor, out: torch.Tensor, state: torch.Tensor, g: torch.Tensor,
                    want_mask: bool, flat: bool) -> torch.Tensor:
        rows, n = z.shape
        grad = torch.empty_like(z)
        wrap_triton(backward_kernel)[(rows,)](
            z, out, state, g, grad, n, WANT_MASK=want_mask, FLAT=flat,
            BLOCK=block(n, _BACKWARD_BLOCK), num_warps=_backward_thread_groups(n))
        return grad

    def setup_context(ctx, inputs, output):
        out, state = output
        ctx.save_for_backward(inputs[0], out, state)
        ctx.flat = inputs[4] >= 1
        ctx.want_mask = inputs[6]

    def backward(ctx, grad_out, grad_state):
        z, out, state = ctx.saved_tensors
        grad = backward_op(z, out, state, grad_out.contiguous(), ctx.want_mask, ctx.flat)
        return grad, None, None, None, None, None, None

    forward_op.register_autograd(backward, setup_context=setup_context)
    _ops = (forward_op, backward_op)
