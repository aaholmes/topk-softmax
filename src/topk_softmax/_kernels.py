"""The Triton kernels behind backend="triton". Imported only when triton is installed.

They are defined at module level so that torch.compile can find the helper `sums` when it
regenerates the kernel source.
"""
import triton
import triton.language as tl


@triton.jit
def sums(z_ptr, base, n, mean, t, shift, tau, BLOCK: tl.constexpr):
    """sum(sigma) and sum(sigma * (1 - sigma)) over one row, for u = (z - mean - t)/tau - shift."""
    s = 0.0
    sv = 0.0
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        z = tl.load(z_ptr + base + offs, mask=valid, other=0.0)
        sig = tl.where(valid, tl.sigmoid((z - mean - t) / tau - shift), 0.0)
        s += tl.sum(sig, axis=0)
        sv += tl.sum(sig * (1.0 - sig), axis=0)
    return s, sv

@triton.jit
def forward_kernel(z_ptr, out_ptr, state_ptr, tau_ptr, tau_number, k, logit, n, max_iters,
                   BLOCK: tl.constexpr, WANT_MASK: tl.constexpr, TAU_IS_TENSOR: tl.constexpr,
                   MAX_POLISH: tl.constexpr, WIDTH: tl.constexpr, FLOOR: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64) * n
    # scalar arguments arrive as 64-bit floats under torch.compile
    k = tl.cast(k, tl.float32)
    logit = tl.cast(logit, tl.float32)
    if TAU_IS_TENSOR:
        tau_in = tl.load(tau_ptr)
    else:
        tau_in = tl.cast(tau_number, tl.float32)

    # row statistics
    total = 0.0
    z_min = float("inf")
    z_max = float("-inf")
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        z = tl.load(z_ptr + base + offs, mask=valid, other=0.0)
        total += tl.sum(z, axis=0)
        z_min = tl.minimum(z_min, tl.min(tl.where(valid, z, float("inf")), axis=0))
        z_max = tl.maximum(z_max, tl.max(tl.where(valid, z, float("-inf")), axis=0))
    mean = total / n
    z_min -= mean
    z_max -= mean
    # tau is floored at FLOOR standard deviations. A standard deviation is at most half the range,
    # so the pass that computes it is skipped when the floor cannot apply.
    tau = tau_in
    if tau_in < FLOOR * 0.5 * (z_max - z_min):
        sq = 0.0
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            valid = offs < n
            zc = tl.load(z_ptr + base + offs, mask=valid, other=0.0) - mean
            sq += tl.sum(tl.where(valid, zc * zc, 0.0), axis=0)
        tau = tl.maximum(tau_in, FLOOR * tl.sqrt(sq / n))

    # bisection on t; sum(sigma) is decreasing in t and the bracket is as in _solve.py
    lo = z_min - tau * logit
    hi = z_max - tau * logit
    i = row * 0
    while i < max_iters:
        mid = 0.5 * (lo + hi)
        s, _ = sums(z_ptr, base, n, mean, mid, 0.0, tau, BLOCK)
        too_big = s > k
        lo = tl.where(too_big, mid, lo)
        hi = tl.where(too_big, hi, mid)
        i = tl.where(hi - lo > WIDTH * tau, i + 1, max_iters)
    t = 0.5 * (lo + hi)

    # Newton steps on a shift of u, each kept only if it reduces |sum(sigma) - k|
    shift = t * 0.0
    s, sv = sums(z_ptr, base, n, mean, t, shift, tau, BLOCK)
    resid = s - k
    tol = 2.5e-7 * k  # two to four 32-bit spacings at k
    j = row * 0
    while j < MAX_POLISH:
        shift_new = shift + resid / tl.maximum(sv, 1e-30)
        s_new, sv_new = sums(z_ptr, base, n, mean, t, shift_new, tau, BLOCK)
        resid_new = s_new - k
        better = tl.abs(resid_new) < tl.abs(resid)
        shift = tl.where(better, shift_new, shift)
        sv = tl.where(better, sv_new, sv)
        resid = tl.where(better, resid_new, resid)
        j = tl.where(better & (tl.abs(resid) > tol), j + 1, MAX_POLISH)

    # The largest score has the largest mask value, so the largest shifted logit is known
    # without another pass, which keeps the softmax to two passes.
    u_top = (z_max - t) / tau - shift
    y_top = z_max + tl.minimum(u_top, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(u_top)))
    denom = 0.0
    if not WANT_MASK:
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            valid = offs < n
            zc = tl.load(z_ptr + base + offs, mask=valid, other=0.0) - mean
            u = (zc - t) / tau - shift
            y = zc + tl.minimum(u, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(u)))
            denom += tl.sum(tl.where(valid, tl.exp(y - y_top), 0.0), axis=0)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        zc = tl.load(z_ptr + base + offs, mask=valid, other=0.0) - mean
        u = (zc - t) / tau - shift
        if WANT_MASK:
            out = tl.sigmoid(u)
        else:
            y = zc + tl.minimum(u, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(u)))
            out = tl.exp(y - y_top) / denom
        tl.store(out_ptr + base + offs, out, mask=valid)

    tl.store(state_ptr + row * 4, mean)
    tl.store(state_ptr + row * 4 + 1, t)
    tl.store(state_ptr + row * 4 + 2, tau)
    tl.store(state_ptr + row * 4 + 3, shift)

@triton.jit
def backward_kernel(z_ptr, out_ptr, state_ptr, g_ptr, grad_ptr, n,
                    BLOCK: tl.constexpr, WANT_MASK: tl.constexpr):
    # Implicit-function-theorem gradient, as in _autograd.py, fused with the softmax gradient:
    #   h = p (g - sum(p g)),  a = (1 - sigma) h,  grad = h + (a - v sum(a)/sum(v)) / tau
    # For the mask alone, h = 0 and a = v g.
    row = tl.program_id(0)
    base = row.to(tl.int64) * n
    mean = tl.load(state_ptr + row * 4)
    t = tl.load(state_ptr + row * 4 + 1)
    tau = tl.load(state_ptr + row * 4 + 2)
    shift = tl.load(state_ptr + row * 4 + 3)

    pg = 0.0
    if not WANT_MASK:
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            valid = offs < n
            p = tl.load(out_ptr + base + offs, mask=valid, other=0.0)
            pg += tl.sum(p * tl.load(g_ptr + base + offs, mask=valid, other=0.0), axis=0)
    sum_a = 0.0
    sum_v = 0.0
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        z = tl.load(z_ptr + base + offs, mask=valid, other=0.0)
        g = tl.load(g_ptr + base + offs, mask=valid, other=0.0)
        sig = tl.where(valid, tl.sigmoid((z - mean - t) / tau - shift), 0.0)
        v = sig * (1.0 - sig)
        if WANT_MASK:
            a = v * g
        else:
            p = tl.load(out_ptr + base + offs, mask=valid, other=0.0)
            a = (1.0 - sig) * p * (g - pg)
        sum_a += tl.sum(tl.where(valid, a, 0.0), axis=0)
        sum_v += tl.sum(v, axis=0)
    ratio = sum_a / tl.maximum(sum_v, 1e-30)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        z = tl.load(z_ptr + base + offs, mask=valid, other=0.0)
        g = tl.load(g_ptr + base + offs, mask=valid, other=0.0)
        sig = tl.sigmoid((z - mean - t) / tau - shift)
        v = sig * (1.0 - sig)
        if WANT_MASK:
            grad = (v * g - v * ratio) / tau
        else:
            h = tl.load(out_ptr + base + offs, mask=valid, other=0.0) * (g - pg)
            grad = h + ((1.0 - sig) * h - v * ratio) / tau
        tl.store(grad_ptr + base + offs, grad, mask=valid)
