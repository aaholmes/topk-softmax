"""The Triton kernels behind backend="triton". Imported only when triton is installed.

They are defined at module level so that torch.compile can find the helper `sums` when it
regenerates the kernel source.
"""
import triton
import triton.language as tl


@triton.jit
def sums(z_ptr, base, n, center, t, shift, tau, BLOCK: tl.constexpr, FLAT: tl.constexpr):
    """sum(sigma) and sum(sigma * (1 - sigma)) over one row, for u = (w - t)/tau - shift.

    w is the centered score z - center, clamped from above at 0 when FLAT (center is then the
    floor(k)-th largest score).
    """
    s = 0.0
    sv = 0.0
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        w = tl.load(z_ptr + base + offs, mask=valid, other=0.0) - center
        if FLAT:
            w = tl.minimum(w, 0.0)
        sig = tl.where(valid, tl.sigmoid((w - t) / tau - shift), 0.0)
        s += tl.sum(sig, axis=0)
        sv += tl.sum(sig * (1.0 - sig), axis=0)
    return s, sv


@triton.jit
def bit_midpoint(below, above):
    """A float halfway between two floats in the order of their bit patterns.

    Mapping the bit pattern to an integer that increases with the float makes every float one
    step, so halving that distance reaches any score within 32 halvings, however the scores are
    spread. The midpoint of the values can need one pass per score when they span many orders of
    magnitude.
    """
    lo = below.to(tl.int32, bitcast=True)
    hi = above.to(tl.int32, bitcast=True)
    lo = tl.where(lo < 0, lo ^ 0x7FFFFFFF, lo).to(tl.int64)
    hi = tl.where(hi < 0, hi ^ 0x7FFFFFFF, hi).to(tl.int64)
    mid = (lo + (hi - lo) // 2).to(tl.int32)
    mid = tl.where(mid < 0, mid ^ 0x7FFFFFFF, mid)
    return mid.to(tl.float32, bitcast=True)


@triton.jit
def single_pass(ptr, off, m, mid, BLOCK: tl.constexpr):
    """Over the m scores at ptr + off: how many exceed mid, the smallest that does, and the
    largest that does not."""
    count = m * 0
    next_up = float("inf")
    next_down = float("-inf")
    for start in range(0, m, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < m
        z = tl.load(ptr + off + offs, mask=valid, other=0.0)
        over = valid & (z > mid)
        count += tl.sum(over.to(tl.int32), axis=0)
        next_up = tl.minimum(next_up, tl.min(tl.where(over, z, float("inf")), axis=0))
        next_down = tl.maximum(next_down, tl.max(tl.where(valid & (z <= mid), z, float("-inf")), axis=0))
    return count, next_up, next_down


@triton.jit
def kth_in_buffer(buf_ptr, off, m, need, below, above, CB: tl.constexpr, MAX_SELECT: tl.constexpr):
    """The need-th largest of the m scores at buf_ptr + off, whose smallest and largest are
    below and above. The trial values are chosen as in forward_kernel."""
    over_below = m
    over_above = m * 0
    j = tl.where(below < above, m * 0, MAX_SELECT)
    while j < MAX_SELECT:
        frac = (over_below.to(tl.float32) - need + 0.5) / (over_below - over_above).to(tl.float32)
        frac = tl.where(j % 2 == 0, frac, 0.5)
        mid = below + frac * (above - below)
        mid = tl.where((j % 2 == 1) & (j >= 24), bit_midpoint(below, above), mid)
        mid = tl.where((mid >= above) | (mid < below), below, mid)
        count, next_up, next_down = single_pass(buf_ptr, off, m, mid, CB)
        enough = count >= need
        below = tl.where(enough, next_up, below)
        above = tl.where(enough, above, next_down)
        over_below = tl.where(enough, count, over_below)
        over_above = tl.where(enough, over_above, count)
        j = tl.where(below < above, j + 1, MAX_SELECT)
    return above


@triton.jit
def forward_kernel(z_ptr, out_ptr, state_ptr, tau_ptr, buf_ptr, tau_number, k, logit, kf, n, max_iters,
                   BLOCK: tl.constexpr, WANT_MASK: tl.constexpr, TAU_IS_TENSOR: tl.constexpr,
                   MAX_POLISH: tl.constexpr, WIDTH: tl.constexpr, FLOOR: tl.constexpr,
                   FLAT: tl.constexpr, MAX_SELECT: tl.constexpr,
                   S: tl.constexpr, SC: tl.constexpr, C: tl.constexpr, CB: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64) * n
    # scalar arguments arrive as 64-bit floats under torch.compile
    k = tl.cast(k, tl.float32)
    logit = tl.cast(logit, tl.float32)
    if TAU_IS_TENSOR:
        tau_in = tl.load(tau_ptr)
    else:
        tau_in = tl.cast(tau_number, tl.float32)

    # For long rows (S > 0), c, the kf-th largest score, is found during the pass for the row
    # statistics. S scores, one at a pseudo-random position in each of S equal stretches of the
    # row, give two pivots: sample order statistics three standard deviations of the sample rank
    # to either side of where c is expected. The pass counts the scores over the upper pivot and
    # copies those between the pivots to a buffer of C scores, which then holds c.
    buf = row.to(tl.int64) * C
    low = float("-inf")
    high = float("inf")
    if S > 0:
        stride = n // S
        s_min = float("inf")
        s_max = float("-inf")
        for first in range(0, S, SC):
            offs = first + tl.arange(0, SC)
            # a hash of the sample index and the row picks the position inside each stretch
            h = offs.to(tl.int64) * 2654435761 + row.to(tl.int64) * 40503 + 12345
            h = (h ^ (h >> 15)) * 2246822519
            h = (h ^ (h >> 13)) & 0x7FFFFFFF
            part = tl.load(z_ptr + base + offs.to(tl.int64) * stride + h % stride)
            tl.store(buf_ptr + buf + offs, part)
            s_min = tl.minimum(s_min, tl.min(part, axis=0))
            s_max = tl.maximum(s_max, tl.max(part, axis=0))
        r = kf * (1.0 * S) / n  # expected number of sample scores over c
        d = 3.0 * tl.sqrt(tl.maximum(r * (1.0 - r / S), 0.0)) + 1.0
        size = row * 0 + S
        low = low + s_min * 0.0
        high = high + s_min * 0.0
        if (r + d).to(tl.int32) + 2 <= S:
            low = kth_in_buffer(buf_ptr, buf, size, (r + d).to(tl.int32) + 2, s_min, s_max, CB, MAX_SELECT)
        if r - d - 1.0 >= 0.0:
            high = kth_in_buffer(buf_ptr, buf, size, (r - d - 1.0).to(tl.int32) + 1, s_min, s_max, CB, MAX_SELECT)

    # row statistics
    total = 0.0
    z_min = float("inf")
    z_max = float("-inf")
    over_high = row * 0       # scores over the upper pivot
    filled = row * 0          # scores between the pivots
    in_min = float("inf")     # the smallest and largest of those
    in_max = float("-inf")
    high_next = float("inf")  # the smallest score over the upper pivot
    low_prev = float("-inf")  # the largest score at or under the lower pivot
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        z = tl.load(z_ptr + base + offs, mask=valid, other=0.0)
        total += tl.sum(z, axis=0)
        z_min = tl.minimum(z_min, tl.min(tl.where(valid, z, float("inf")), axis=0))
        z_max = tl.maximum(z_max, tl.max(tl.where(valid, z, float("-inf")), axis=0))
        if S > 0:
            inside = valid & (z > low) & (z <= high)
            spot = filled + tl.cumsum(inside.to(tl.int32), axis=0) - 1
            tl.store(buf_ptr + buf + spot, z, mask=inside & (spot < C))
            filled += tl.sum(inside.to(tl.int32), axis=0)
            over_high += tl.sum((valid & (z > high)).to(tl.int32), axis=0)
            in_min = tl.minimum(in_min, tl.min(tl.where(inside, z, float("inf")), axis=0))
            in_max = tl.maximum(in_max, tl.max(tl.where(inside, z, float("-inf")), axis=0))
            high_next = tl.minimum(high_next, tl.min(tl.where(valid & (z > high), z, float("inf")), axis=0))
            low_prev = tl.maximum(low_prev, tl.max(tl.where(valid & (z <= low), z, float("-inf")), axis=0))

    if FLAT:
        # Find c, the kf-th largest score. below and above are scores of the row with
        # below <= c <= above. Each pass counts the scores over a trial value and moves one end to
        # the nearest score on the correct side, so the loop ends exactly at c, ties included.
        below = z_min
        above = z_max
        # count_below scores are >= below and count_above are > above; kf lies between them
        count_below = row * 0 + n
        count_above = row * 0
        if S > 0:
            between = (over_high < kf) & (kf <= over_high + filled)
            if between & (filled <= C):
                below = kth_in_buffer(buf_ptr, buf, filled, kf - over_high, in_min, in_max, CB, MAX_SELECT)
                above = below
            else:
                # The buffer overflowed or the pivots missed. The loop below then starts from
                # the part of the row that the pass showed to hold c.
                higher = kf <= over_high
                below = tl.where(between, in_min, tl.where(higher, high_next, z_min))
                above = tl.where(between, in_max, tl.where(higher, z_max, low_prev))
                count_below = tl.where(between, over_high + filled, tl.where(higher, over_high, count_below))
                count_above = tl.where(between, over_high, tl.where(higher, count_above, over_high + filled))
        i = tl.where(below < above, row * 0, MAX_SELECT)
        while i < MAX_SELECT:
            # Trial value: on even passes, where a straight line through the two counts crosses
            # kf; on odd passes the midpoint. From pass 24 on, the odd passes take the midpoint
            # of the bit patterns, which bounds the passes left at 2 * 33 for any scores.
            frac = (count_below.to(tl.float32) - kf + 0.5) / (count_below - count_above).to(tl.float32)
            frac = tl.where(i % 2 == 0, frac, 0.5)
            mid = below + frac * (above - below)
            mid = tl.where((i % 2 == 1) & (i >= 24), bit_midpoint(below, above), mid)
            mid = tl.where((mid >= above) | (mid < below), below, mid)
            count = row * 0
            next_up = float("inf")
            next_down = float("-inf")
            for start in range(0, n, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                valid = offs < n
                z = tl.load(z_ptr + base + offs, mask=valid, other=0.0)
                over = valid & (z > mid)
                count += tl.sum(over.to(tl.int32), axis=0)
                next_up = tl.minimum(next_up, tl.min(tl.where(over, z, float("inf")), axis=0))
                next_down = tl.maximum(next_down, tl.max(tl.where(valid & (z <= mid), z, float("-inf")), axis=0))
            enough = count >= kf  # then c is above mid
            below = tl.where(enough, next_up, below)
            above = tl.where(enough, above, next_down)
            count_below = tl.where(enough, count, count_below)
            count_above = tl.where(enough, count_above, count)
            i = tl.where(below < above, i + 1, MAX_SELECT)
        # the mask is computed from w = min(z, c) - c, which runs from z_min - c up to 0
        center = above
        s_top = z_max - center
        z_min = z_min - center
        z_max = center * 0.0
    else:
        center = total / n
        z_min -= center
        z_max -= center
        s_top = z_max

    # tau is floored at FLOOR standard deviations of the scores the mask is computed from. A
    # standard deviation is at most half the range, so the pass that computes it is skipped when
    # the floor cannot apply.
    tau = tau_in
    if tau_in < FLOOR * 0.5 * (z_max - z_min):
        sq = 0.0
        lin = 0.0
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            valid = offs < n
            w = tl.load(z_ptr + base + offs, mask=valid, other=0.0) - center
            if FLAT:
                w = tl.minimum(w, 0.0)
            w = tl.where(valid, w, 0.0)
            sq += tl.sum(w * w, axis=0)
            lin += tl.sum(w, axis=0)
        var = tl.maximum(sq / n - (lin / n) * (lin / n), 0.0)
        tau = tl.maximum(tau_in, FLOOR * tl.sqrt(var))

    # bisection on t; sum(sigma) is decreasing in t and the bracket is as in _solve.py
    lo = z_min - tau * logit
    hi = z_max - tau * logit
    i = row * 0
    while i < max_iters:
        mid = 0.5 * (lo + hi)
        s, _ = sums(z_ptr, base, n, center, mid, 0.0, tau, BLOCK, FLAT)
        too_big = s > k
        lo = tl.where(too_big, mid, lo)
        hi = tl.where(too_big, hi, mid)
        i = tl.where(hi - lo > WIDTH * tau, i + 1, max_iters)
    t = 0.5 * (lo + hi)

    # Newton steps on a shift of u, each kept only if it reduces |sum(sigma) - k|
    shift = t * 0.0
    s, sv = sums(z_ptr, base, n, center, t, shift, tau, BLOCK, FLAT)
    resid = s - k
    tol = 2.5e-7 * k  # two to four 32-bit spacings at k
    j = row * 0
    while j < MAX_POLISH:
        shift_new = shift + resid / tl.maximum(sv, 1e-30)
        s_new, sv_new = sums(z_ptr, base, n, center, t, shift_new, tau, BLOCK, FLAT)
        resid_new = s_new - k
        better = tl.abs(resid_new) < tl.abs(resid)
        shift = tl.where(better, shift_new, shift)
        sv = tl.where(better, sv_new, sv)
        resid = tl.where(better, resid_new, resid)
        j = tl.where(better & (tl.abs(resid) > tol), j + 1, MAX_POLISH)

    # The largest score has the largest mask value, so the largest shifted logit is known
    # without another pass, which keeps the softmax to two passes.
    u_top = (z_max - t) / tau - shift
    y_top = s_top + tl.minimum(u_top, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(u_top)))
    denom = 0.0
    if not WANT_MASK:
        for start in range(0, n, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            valid = offs < n
            zc = tl.load(z_ptr + base + offs, mask=valid, other=0.0) - center
            if FLAT:
                w = tl.minimum(zc, 0.0)
            else:
                w = zc
            u = (w - t) / tau - shift
            y = zc + tl.minimum(u, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(u)))
            denom += tl.sum(tl.where(valid, tl.exp(y - y_top), 0.0), axis=0)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        zc = tl.load(z_ptr + base + offs, mask=valid, other=0.0) - center
        if FLAT:
            w = tl.minimum(zc, 0.0)
        else:
            w = zc
        u = (w - t) / tau - shift
        if WANT_MASK:
            out = tl.sigmoid(u)
        else:
            y = zc + tl.minimum(u, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(u)))
            out = tl.exp(y - y_top) / denom
        tl.store(out_ptr + base + offs, out, mask=valid)

    tl.store(state_ptr + row * 4, center)
    tl.store(state_ptr + row * 4 + 1, t)
    tl.store(state_ptr + row * 4 + 2, tau)
    tl.store(state_ptr + row * 4 + 3, shift)


@triton.jit
def backward_kernel(z_ptr, out_ptr, state_ptr, g_ptr, grad_ptr, n,
                    BLOCK: tl.constexpr, WANT_MASK: tl.constexpr, FLAT: tl.constexpr):
    # Implicit-function-theorem gradient, as in _autograd.py, fused with the softmax gradient:
    #   h = p (g - sum(p g)),  a = (1 - sigma) h,  grad = h + (a - v sum(a)/sum(v)) / tau
    # For the mask alone, h = 0 and a = v g.
    # With FLAT the mask sees min(z, c). The second term of every score above c is then a gradient
    # for c, so it is summed and given to the scores equal to c, split equally if several tie.
    row = tl.program_id(0)
    base = row.to(tl.int64) * n
    center = tl.load(state_ptr + row * 4)  # the row mean, or c when FLAT
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
    sum_a_above = 0.0
    sum_v_above = 0.0
    at_c = 0.0
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        zc = tl.load(z_ptr + base + offs, mask=valid, other=0.0) - center
        g = tl.load(g_ptr + base + offs, mask=valid, other=0.0)
        if FLAT:
            w = tl.minimum(zc, 0.0)
        else:
            w = zc
        sig = tl.where(valid, tl.sigmoid((w - t) / tau - shift), 0.0)
        v = sig * (1.0 - sig)
        if WANT_MASK:
            a = v * g
        else:
            p = tl.load(out_ptr + base + offs, mask=valid, other=0.0)
            a = (1.0 - sig) * p * (g - pg)
        a = tl.where(valid, a, 0.0)
        sum_a += tl.sum(a, axis=0)
        sum_v += tl.sum(v, axis=0)
        if FLAT:
            over = valid & (zc > 0.0)
            sum_a_above += tl.sum(tl.where(over, a, 0.0), axis=0)
            sum_v_above += tl.sum(tl.where(over, v, 0.0), axis=0)
            at_c += tl.sum(tl.where(valid & (zc == 0.0), 1.0, 0.0), axis=0)
    ratio = sum_a / tl.maximum(sum_v, 1e-30)
    routed = (sum_a_above - sum_v_above * ratio) / tau / tl.maximum(at_c, 1.0)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        zc = tl.load(z_ptr + base + offs, mask=valid, other=0.0) - center
        g = tl.load(g_ptr + base + offs, mask=valid, other=0.0)
        if FLAT:
            w = tl.minimum(zc, 0.0)
        else:
            w = zc
        sig = tl.sigmoid((w - t) / tau - shift)
        v = sig * (1.0 - sig)
        if WANT_MASK:
            h = zc * 0.0
            a = v * g
        else:
            h = tl.load(out_ptr + base + offs, mask=valid, other=0.0) * (g - pg)
            a = (1.0 - sig) * h
        through_mask = (a - v * ratio) / tau
        if FLAT:
            through_mask = tl.where(zc > 0.0, 0.0, through_mask) + tl.where(zc == 0.0, routed, 0.0)
        tl.store(grad_ptr + base + offs, h + through_mask, mask=valid)
