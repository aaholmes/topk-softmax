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


def _forward_thread_groups(n):
    return 1 if n <= 512 else 2 if n <= 1024 else 8


def _backward_thread_groups(n):
    return 4 if n <= 1024 else 16


_MAX_POLISH = 6       # Newton steps after the bisection; most rows need 2
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


def run(x, k, tau, iters, want_mask, mask_scores=None):
    """The mask (want_mask=True) or the masked softmax along the last dim of a 32-bit CUDA tensor.

    mask_scores, if given, are the scores the mask is computed from in place of x (flat_top).
    """
    n = x.shape[-1]
    z = x.contiguous().view(-1, n)
    w = None if mask_scores is None else mask_scores.contiguous().view(-1, n)
    if isinstance(tau, torch.Tensor):
        tau_tensor, tau_number = tau.detach().to(device=z.device, dtype=torch.float32), 0.0
    else:
        tau_tensor, tau_number = None, float(tau)
    out, _ = _ops[0](z, w, tau_tensor, tau_number, float(k), int(iters), bool(want_mask))
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
    def forward_op(z: torch.Tensor, w: torch.Tensor | None, tau_tensor: torch.Tensor | None, tau_number: float,
                   k: float, max_iters: int, want_mask: bool) -> tuple[torch.Tensor, torch.Tensor]:
        rows, n = z.shape
        out = torch.empty_like(z)
        state = z.new_empty(rows, 4)  # per row: mean, threshold, floored tau, Newton shift
        wrap_triton(forward_kernel)[(rows,)](
            z if w is None else w, z, out, state, z if tau_tensor is None else tau_tensor, tau_number, k,
            math.log(k / (n - k)), n, max_iters,
            WANT_MASK=want_mask, TAU_IS_TENSOR=tau_tensor is not None,
            MAX_POLISH=_MAX_POLISH, WIDTH=_BRACKET_WIDTH, FLOOR=_TAU_FLOOR, FLAT=w is not None,
            BLOCK=block(n, _FORWARD_BLOCK), num_warps=_forward_thread_groups(n))
        return out, state

    @triton_op("topk_softmax::soft_topk_backward", mutates_args={})
    def backward_op(mask_scores: torch.Tensor, out: torch.Tensor, state: torch.Tensor, g: torch.Tensor,
                    want_mask: bool, flat: bool) -> tuple[torch.Tensor, torch.Tensor]:
        rows, n = mask_scores.shape
        grad = torch.empty_like(mask_scores)
        grad_mask = torch.empty_like(mask_scores) if flat else mask_scores.new_empty(0)
        wrap_triton(backward_kernel)[(rows,)](
            mask_scores, out, state, g, grad, grad_mask, n, WANT_MASK=want_mask, FLAT=flat,
            BLOCK=block(n, _BACKWARD_BLOCK), num_warps=_backward_thread_groups(n))
        return grad, grad_mask

    def setup_context(ctx, inputs, output):
        z, w = inputs[0], inputs[1]
        out, state = output
        ctx.save_for_backward(z if w is None else w, out, state)
        ctx.want_mask = inputs[6]
        ctx.flat = w is not None

    def backward(ctx, grad_out, grad_state):
        mask_scores, out, state = ctx.saved_tensors
        grad, grad_mask = backward_op(mask_scores, out, state, grad_out.contiguous(), ctx.want_mask, ctx.flat)
        return grad, (grad_mask if ctx.flat else None), None, None, None, None, None

    forward_op.register_autograd(backward, setup_context=setup_context)
    _ops = (forward_op, backward_op)
