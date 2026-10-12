"""Ways to find the k-th largest score of each row on the GPU, timed as standalone kernels.

    python benchmarks/select.py --out benchmarks/results/NAME.json     # time (GPU; --n, --arms, --k100 choose the grid)
    python benchmarks/select.py --report benchmarks/results/NAME.json  # tables from a saved file

results/select.json holds every arm for n up to 131072; results/select_wide.json holds the
leading arms for n from 64 to 2097152, with the bit-pattern midpoint described below.

Every arm returns one value per row and is checked to equal torch.topk's k-th value on every row
before it is timed. One Triton program handles one row.

    minmax        one pass for the row minimum and maximum; every Triton arm below starts with it
    current       the library's rule: one trial value per pass, alternately interpolated and the
                  midpoint; each pass counts the scores over the trial and moves one end of the
                  interval to the nearest score. From pass 24 on, the midpoint is taken on the
                  bit patterns, which bounds the passes for any input
    bits          `current` with the bit-pattern midpoint from the first pass
    multi4/8/16   that rule with 4, 8 or 16 evenly spaced trial values per pass
    sample64/256/1024  two pivots from that many evenly strided scores bracket the target (ranked by
                  comparing every pair in the sample), then `current`
    sample64+multi8  the pivots from 64 scores, then `multi8`
    finish64/256/1024  `current` until at most that many scores remain between the two ends; one
                  pass then copies those scores to a small buffer, and the rule continues on the
                  buffer alone
    sample64+finish64/256/1024  the pivots from 64 scores, then that
    aim+finish64/256/1024  as finish, but each trial aims a quarter of the buffer size short of the
                  target rank, on whichever side has more scores left, so that both ends close in;
                  every third pass uses the midpoint
    sample64+aim+finish256/1024  the pivots from 64 scores, then that
    oneread       one pass over the row: a sample of S scores gives two pivots that bracket the
                  target with high probability (three standard deviations of the sample rank);
                  the pass counts the scores over the upper pivot and copies those between the
                  pivots to a buffer of C scores, where the rule finishes. The sample takes one
                  score at a pseudo-random position in each of S equal stretches of the row, and
                  S and C are computed from n and k (`_oneread_sizes` in the library). A row whose pivots miss,
                  or whose buffer overflows, falls back to `current`, started from the interval
                  and counts the pass found (reported as the fraction of rows).
    oneread_even  `oneread` with the sample at an even stride
    onereadS/C    `oneread_even` with S and C fixed (shrunk to n/4 and n for short rows)
    hybrid        `current` for rows shorter than 16384, `oneread` otherwise

    python benchmarks/select.py --census benchmarks/results/select_census.json   # fallback rates over many rows
    python benchmarks/select.py --plot benchmarks/results/select_wide.json OUT.png  # time per score against row length
    compact       `current`, but each pass also writes the scores on both sides of the trial to a
                  scratch row, and the next pass reads only the side that holds the target
                  (quickselect with value pivots)
    heap          one sequential pass with a min-heap of the k largest scores seen so far
    topk          torch.topk(z, k).values[:, -1]
    kthvalue      torch.kthvalue(z, n - k + 1).values

Block size and thread groups are chosen per arm and per n as the fastest of a small grid, on the
input that is then timed.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from topk_softmax._kernels import bit_midpoint, kth_in_buffer as _kth_in_buffer, single_pass as _single_pass
from topk_softmax._triton import _oneread_sizes

NS = [64, 512, 4096, 32768, 262144, 2097152]
MAX_BATCH = 4096
MAX_INPUT_BYTES = 2**29
MAX_SELECT = 128  # 24 + 3 * 33 passes always suffice, also for the `aim` arms
INPUTS = ["gaussian", "rounded", "heavy_tail", "wide", "sorted", "periodic", "halves"]
CENSUS_INPUTS = INPUTS + ["descending", "periodic4096", "constant", "two_values", "spikes"]


@triton.jit
def _minmax(z_ptr, base, n, BLOCK: tl.constexpr):
    z_min = float("inf")
    z_max = float("-inf")
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        z = tl.load(z_ptr + base + offs, mask=valid, other=0.0)
        z_min = tl.minimum(z_min, tl.min(tl.where(valid, z, float("inf")), axis=0))
        z_max = tl.maximum(z_max, tl.max(tl.where(valid, z, float("-inf")), axis=0))
    return z_min, z_max


@triton.jit
def _count_pass(z_ptr, base, n, trials, BLOCK: tl.constexpr, M: tl.constexpr):
    """For each trial value: the number of scores over it, the smallest score over it, and the
    largest score at or under it."""
    counts = tl.zeros((M,), dtype=tl.int32)
    up = tl.full((M,), float("inf"), dtype=tl.float32)
    down = tl.full((M,), float("-inf"), dtype=tl.float32)
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        z = tl.load(z_ptr + base + offs, mask=valid, other=0.0)
        zz = z[:, None]
        over = valid[:, None] & (zz > trials[None, :])
        under = valid[:, None] & (zz <= trials[None, :])
        counts += tl.sum(over.to(tl.int32), axis=0)
        up = tl.minimum(up, tl.min(tl.where(over, zz, float("inf")), axis=0))
        down = tl.maximum(down, tl.max(tl.where(under, zz, float("-inf")), axis=0))
    return counts, up, down


@triton.jit
def minmax_kernel(z_ptr, c_ptr, n, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    z_min, z_max = _minmax(z_ptr, row.to(tl.int64) * n, n, BLOCK)
    tl.store(c_ptr + row, z_max - z_min)


@triton.jit
def select_kernel(z_ptr, buf_ptr, c_ptr, passes_ptr, kf, n,
                  BLOCK: tl.constexpr, M: tl.constexpr, S: tl.constexpr, SC: tl.constexpr, C: tl.constexpr,
                  AIM: tl.constexpr, BITS_FROM: tl.constexpr, MAX_SELECT: tl.constexpr):
    """M trial values per pass (M = 1 is the library's rule); S > 0 first takes pivots from S scores;
    C > 0 finishes on a buffer of at most C scores."""
    row = tl.program_id(0)
    base = row.to(tl.int64) * n
    below, above = _minmax(z_ptr, base, n, BLOCK)
    # count_below scores are >= below and count_above are > above; kf lies between them
    count_below = row * 0 + n
    count_above = row * 0
    passes = row * 0

    if S > 0:
        stride = n // S
        s = tl.load(z_ptr + base + tl.arange(0, S).to(tl.int64) * stride)
        rank = tl.zeros((S,), dtype=tl.int32)  # sample values over each one
        for first in range(0, S, SC):
            part = tl.load(z_ptr + base + (first + tl.arange(0, SC)).to(tl.int64) * stride)
            rank += tl.sum((part[None, :] > s[:, None]).to(tl.int32), axis=1)
        r = kf * (1.0 * S) / n
        d = 2.0 * tl.sqrt(tl.maximum(r * (1.0 - r / S), 0.0)) + 1.0
        low = tl.max(tl.where(rank.to(tl.float32) >= r + d, s, float("-inf")), axis=0)
        high = tl.min(tl.where(rank.to(tl.float32) <= r - d - 1.0, s, float("inf")), axis=0)
        low = tl.where(low > below, low, below)
        high = tl.where(high < above, high, above)
        pair = tl.where(tl.arange(0, 2) == 0, low, high)
        pair_counts, pair_up, pair_down = _count_pass(z_ptr, base, n, pair, BLOCK, 2)
        pair_ok = pair_counts >= kf
        some_ok = tl.max(pair_ok.to(tl.int32), axis=0) > 0
        some_not = tl.min(pair_ok.to(tl.int32), axis=0) < 1
        below = tl.where(some_ok, tl.max(tl.where(pair_ok, pair_up, float("-inf")), axis=0), below)
        count_below = tl.where(some_ok, tl.min(tl.where(pair_ok, pair_counts, 2147483647), axis=0), count_below)
        above = tl.where(some_not, tl.min(tl.where(pair_ok, float("inf"), pair_down), axis=0), above)
        count_above = tl.where(some_not, tl.max(tl.where(pair_ok, -1, pair_counts), axis=0), count_above)
        passes += 1

    i = tl.where(below < above, row * 0, MAX_SELECT)
    while (i < MAX_SELECT) & (count_below - count_above > C):
        if M == 1:
            if AIM:
                aim = tl.where(count_below - kf >= kf - count_above, kf + C // 4, kf - C // 4)
                frac = (count_below - aim).to(tl.float32) / (count_below - count_above).to(tl.float32)
                frac = tl.where(i % 3 == 2, 0.5, frac)
                use_bits = (i % 3 == 2) & (i >= BITS_FROM)
            else:
                frac = (count_below.to(tl.float32) - kf + 0.5) / (count_below - count_above).to(tl.float32)
                frac = tl.where(i % 2 == 0, frac, 0.5)
                use_bits = (i % 2 == 1) & (i >= BITS_FROM)
            mid = below + frac * (above - below)
            mid = tl.where(use_bits, bit_midpoint(below, above), mid)
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
            enough = count >= kf
            below = tl.where(enough, next_up, below)
            above = tl.where(enough, above, next_down)
            count_below = tl.where(enough, count, count_below)
            count_above = tl.where(enough, count_above, count)
        else:
            trials = below + (tl.arange(0, M) + 1).to(tl.float32) / (M + 1) * (above - below)
            trials = tl.where((trials >= above) | (trials < below), below, trials)
            counts, up, down = _count_pass(z_ptr, base, n, trials, BLOCK, M)
            ok = counts >= kf  # then the target is over that trial
            any_ok = tl.max(ok.to(tl.int32), axis=0) > 0
            any_not = tl.min(ok.to(tl.int32), axis=0) < 1
            below = tl.where(any_ok, tl.max(tl.where(ok, up, float("-inf")), axis=0), below)
            count_below = tl.where(any_ok, tl.min(tl.where(ok, counts, 2147483647), axis=0), count_below)
            above = tl.where(any_not, tl.min(tl.where(ok, float("inf"), down), axis=0), above)
            count_above = tl.where(any_not, tl.max(tl.where(ok, -1, counts), axis=0), count_above)
        passes += 1
        i = tl.where(below < above, i + 1, MAX_SELECT)

    if C > 0:
        m = count_below - count_above  # scores in [below, above]; the target is the need-th largest of them
        if (below < above) & (m <= C):
            buf = row.to(tl.int64) * C
            filled = row * 0
            for start in range(0, n, BLOCK):
                offs = start + tl.arange(0, BLOCK)
                valid = offs < n
                z = tl.load(z_ptr + base + offs, mask=valid, other=0.0)
                inside = valid & (z >= below) & (z <= above)
                tl.store(buf_ptr + buf + filled + tl.cumsum(inside.to(tl.int32), axis=0) - 1, z, mask=inside)
                filled += tl.sum(inside.to(tl.int32), axis=0)
            passes += 1
            need = kf - count_above
            over_below = m
            over_above = row * 0
            j = row * 0
            while j < MAX_SELECT:
                frac = (over_below.to(tl.float32) - need + 0.5) / (over_below - over_above).to(tl.float32)
                frac = tl.where(j % 2 == 0, frac, 0.5)
                mid = below + frac * (above - below)
                mid = tl.where((j % 2 == 1) & (j >= 24), bit_midpoint(below, above), mid)
                mid = tl.where((mid >= above) | (mid < below), below, mid)
                count, next_up, next_down = _single_pass(buf_ptr, buf, m, mid, C)
                enough = count >= need
                below = tl.where(enough, next_up, below)
                above = tl.where(enough, above, next_down)
                over_below = tl.where(enough, count, over_below)
                over_above = tl.where(enough, over_above, count)
                j = tl.where(below < above, j + 1, MAX_SELECT)
    tl.store(c_ptr + row, above)
    tl.store(passes_ptr + row, passes)


@triton.jit
def oneread_kernel(z_ptr, buf_ptr, c_ptr, passes_ptr, kf, n, BLOCK: tl.constexpr, S: tl.constexpr, SC: tl.constexpr,
                   C: tl.constexpr, CB: tl.constexpr, SCATTER: tl.constexpr, MAX_SELECT: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64) * n
    buf = row.to(tl.int64) * C

    # the sample goes to the buffer, where two of its order statistics become the pivots
    stride = n // S
    s_min = float("inf")
    s_max = float("-inf")
    for first in range(0, S, SC):
        offs = first + tl.arange(0, SC)
        spots = offs.to(tl.int64) * stride
        if SCATTER:  # a hash of the sample index and the row picks the position inside each stretch
            h = offs.to(tl.int64) * 2654435761 + row.to(tl.int64) * 40503 + 12345
            h = (h ^ (h >> 15)) * 2246822519
            h = (h ^ (h >> 13)) & 0x7FFFFFFF
            spots += h % stride
        part = tl.load(z_ptr + base + spots)
        tl.store(buf_ptr + buf + offs, part)
        s_min = tl.minimum(s_min, tl.min(part, axis=0))
        s_max = tl.maximum(s_max, tl.max(part, axis=0))
    r = kf * (1.0 * S) / n  # expected number of sample scores over the target
    d = 3.0 * tl.sqrt(tl.maximum(r * (1.0 - r / S), 0.0)) + 1.0
    size = row * 0 + S
    need_low = (r + d).to(tl.int32) + 2
    low = float("-inf") + s_min * 0.0
    if need_low <= S:
        low = _kth_in_buffer(buf_ptr, buf, size, need_low, s_min, s_max, CB, MAX_SELECT)
    high = float("inf") + s_min * 0.0
    if r - d - 1.0 >= 0.0:
        high = _kth_in_buffer(buf_ptr, buf, size, (r - d - 1.0).to(tl.int32) + 1, s_min, s_max, CB, MAX_SELECT)

    # the one pass over the row
    over_high = row * 0
    filled = row * 0
    in_min = float("inf")
    in_max = float("-inf")
    z_min = float("inf")
    z_max = float("-inf")
    high_next = float("inf")  # the smallest score over the upper pivot
    low_prev = float("-inf")  # the largest score at or under the lower pivot
    for start in range(0, n, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < n
        z = tl.load(z_ptr + base + offs, mask=valid, other=0.0)
        inside = valid & (z > low) & (z <= high)
        spot = filled + tl.cumsum(inside.to(tl.int32), axis=0) - 1
        tl.store(buf_ptr + buf + spot, z, mask=inside & (spot < C))
        filled += tl.sum(inside.to(tl.int32), axis=0)
        over_high += tl.sum((valid & (z > high)).to(tl.int32), axis=0)
        high_next = tl.minimum(high_next, tl.min(tl.where(valid & (z > high), z, float("inf")), axis=0))
        low_prev = tl.maximum(low_prev, tl.max(tl.where(valid & (z <= low), z, float("-inf")), axis=0))
        in_min = tl.minimum(in_min, tl.min(tl.where(inside, z, float("inf")), axis=0))
        in_max = tl.maximum(in_max, tl.max(tl.where(inside, z, float("-inf")), axis=0))
        z_min = tl.minimum(z_min, tl.min(tl.where(valid, z, float("inf")), axis=0))
        z_max = tl.maximum(z_max, tl.max(tl.where(valid, z, float("-inf")), axis=0))

    passes = row * 0 + 1
    c = z_max
    if (over_high < kf) & (kf <= over_high + filled) & (filled <= C):
        c = _kth_in_buffer(buf_ptr, buf, filled, kf - over_high, in_min, in_max, CB, MAX_SELECT)
    else:
        # The buffer overflowed or the pivots missed: the library's rule on the whole row, started
        # from the part of the row that the pass showed to hold the target.
        length = row * 0 + n
        between = (over_high < kf) & (kf <= over_high + filled)
        higher = kf <= over_high
        below = tl.where(between, in_min, tl.where(higher, high_next, z_min))
        above = tl.where(between, in_max, tl.where(higher, z_max, low_prev))
        count_below = tl.where(between, over_high + filled, tl.where(higher, over_high, length))
        count_above = tl.where(between, over_high, tl.where(higher, row * 0, over_high + filled))
        i = tl.where(below < above, row * 0, MAX_SELECT)
        while i < MAX_SELECT:
            frac = (count_below.to(tl.float32) - kf + 0.5) / (count_below - count_above).to(tl.float32)
            frac = tl.where(i % 2 == 0, frac, 0.5)
            mid = below + frac * (above - below)
            mid = tl.where((i % 2 == 1) & (i >= 24), bit_midpoint(below, above), mid)
            mid = tl.where((mid >= above) | (mid < below), below, mid)
            count, next_up, next_down = _single_pass(z_ptr, base, length, mid, BLOCK)
            enough = count >= kf
            below = tl.where(enough, next_up, below)
            above = tl.where(enough, above, next_down)
            count_below = tl.where(enough, count, count_below)
            count_above = tl.where(enough, count_above, count)
            passes += 1
            i = tl.where(below < above, i + 1, MAX_SELECT)
        c = above
    tl.store(c_ptr + row, c)
    tl.store(passes_ptr + row, passes)


@triton.jit
def _partition(src_ptr, src, m, dst_ptr, dst, mid, BLOCK: tl.constexpr):
    """Write the m scores at src to dst: those over mid from the front, the others from the back."""
    count = m * 0
    rest = m * 0
    next_up = float("inf")
    next_down = float("-inf")
    for start in range(0, m, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        valid = offs < m
        z = tl.load(src_ptr + src + offs, mask=valid, other=0.0)
        over = valid & (z > mid)
        under = valid & (z <= mid)
        tl.store(dst_ptr + dst + count + tl.cumsum(over.to(tl.int32), axis=0) - 1, z, mask=over)
        tl.store(dst_ptr + dst + (m - 1) - (rest + tl.cumsum(under.to(tl.int32), axis=0) - 1), z, mask=under)
        count += tl.sum(over.to(tl.int32), axis=0)
        rest += tl.sum(under.to(tl.int32), axis=0)
        next_up = tl.minimum(next_up, tl.min(tl.where(over, z, float("inf")), axis=0))
        next_down = tl.maximum(next_down, tl.max(tl.where(under, z, float("-inf")), axis=0))
    return count, next_up, next_down


@triton.jit
def compact_kernel(z_ptr, scratch_ptr, c_ptr, passes_ptr, kf, n, BLOCK: tl.constexpr, MAX_SELECT: tl.constexpr):
    row = tl.program_id(0)
    base = row.to(tl.int64) * n
    halves = row.to(tl.int64) * 2 * n  # two scratch rows, written alternately
    below, above = _minmax(z_ptr, base, n, BLOCK)
    m = row * 0 + n  # candidates left; the target is the kk-th largest of them
    kk = row * 0 + kf
    src = halves
    passes = row * 0
    i = tl.where(below < above, row * 0, MAX_SELECT)
    if i < MAX_SELECT:
        # the first pass reads the input row
        mid = below + (m.to(tl.float32) - kk + 0.5) / m.to(tl.float32) * (above - below)
        mid = tl.where((mid >= above) | (mid < below), below, mid)
        count, next_up, next_down = _partition(z_ptr, base, m, scratch_ptr, halves, mid, BLOCK)
        enough = count >= kk
        below = tl.where(enough, next_up, below)
        above = tl.where(enough, above, next_down)
        src = tl.where(enough, halves, halves + count)
        kk = tl.where(enough, kk, kk - count)
        m = tl.where(enough, count, m - count)
        passes += 1
        i = tl.where(below < above, i + 1, MAX_SELECT)
    while i < MAX_SELECT:
        frac = (m.to(tl.float32) - kk + 0.5) / m.to(tl.float32)
        frac = tl.where(i % 2 == 0, frac, 0.5)
        mid = below + frac * (above - below)
        mid = tl.where((mid >= above) | (mid < below), below, mid)
        dst = halves + (i % 2).to(tl.int64) * n
        count, next_up, next_down = _partition(scratch_ptr, src, m, scratch_ptr, dst, mid, BLOCK)
        enough = count >= kk
        below = tl.where(enough, next_up, below)
        above = tl.where(enough, above, next_down)
        src = tl.where(enough, dst, dst + count)
        kk = tl.where(enough, kk, kk - count)
        m = tl.where(enough, count, m - count)
        passes += 1
        i = tl.where(below < above, i + 1, MAX_SELECT)
    tl.store(c_ptr + row, above)
    tl.store(passes_ptr + row, passes)


@triton.jit
def heap_kernel(z_ptr, heap_ptr, c_ptr, kf, n):
    """A min-heap of the kf largest scores seen so far; its root at the end is the target."""
    row = tl.program_id(0)
    base = row.to(tl.int64) * n
    heap = row.to(tl.int64) * kf
    for i in range(0, kf):  # insert, sifting up
        v = tl.load(z_ptr + base + i)
        j = row * 0 + i
        go = j > 0
        while go:
            parent = (j - 1) // 2
            pv = tl.load(heap_ptr + heap + parent)
            swap = pv > v
            tl.store(heap_ptr + heap + j, tl.where(swap, pv, v))
            j = tl.where(swap, parent, j)
            go = swap & (j > 0)
        tl.store(heap_ptr + heap + j, v)
    for i in range(kf, n):  # replace the root if the score is larger, sifting down
        v = tl.load(z_ptr + base + i)
        j = row * 0
        go = v > tl.load(heap_ptr + heap)
        while go:
            left = 2 * j + 1
            right = left + 1
            lv = tl.where(left < kf, tl.load(heap_ptr + heap + tl.minimum(left, kf - 1)), float("inf"))
            rv = tl.where(right < kf, tl.load(heap_ptr + heap + tl.minimum(right, kf - 1)), float("inf"))
            child = tl.where(rv < lv, right, left)
            cv = tl.minimum(lv, rv)
            swap = cv < v
            tl.store(heap_ptr + heap + j, tl.where(swap, cv, v))
            j = tl.where(swap, child, j)
            go = swap
    tl.store(c_ptr + row, tl.load(heap_ptr + heap))


def _pow2_floor(x):
    return triton.next_power_of_2(int(x) + 1) // 2


def _arm(name, z, kf, block, warps):
    """A function () -> (c, passes or None) for the arm."""
    rows, n = z.shape
    c = z.new_empty(rows)
    passes = torch.zeros(rows, dtype=torch.int32, device=z.device)
    if name == "topk":
        return lambda: (z.topk(kf, -1).values[:, -1], None)
    if name == "kthvalue":
        return lambda: (z.kthvalue(n - kf + 1, -1).values, None)
    if name == "minmax":
        def run():
            minmax_kernel[(rows,)](z, c, n, BLOCK=block, num_warps=warps)
            return None, None
        return run
    if name == "heap":
        heap = z.new_empty(rows, kf)

        def run():
            heap_kernel[(rows,)](z, heap, c, kf, n, num_warps=1)
            return c, None
        return run
    if name == "compact":
        scratch = z.new_empty(rows, 2 * n)

        def run():
            compact_kernel[(rows,)](z, scratch, c, passes, kf, n, BLOCK=block, MAX_SELECT=MAX_SELECT, num_warps=warps)
            return c, passes
        return run
    if name == "hybrid":
        name = "current" if n < 16384 else "oneread"
    if name.startswith("oneread"):
        if "/" in name:
            sample, cap = (int(x) for x in name[7:].split("/"))
            sample = min(sample, max(16, _pow2_floor(n // 4)))
            cap = max(min(cap, triton.next_power_of_2(n)), sample)
        else:
            sample, cap = _oneread_sizes(n, kf)
        scratch = z.new_empty(rows, cap)

        def run():
            oneread_kernel[(rows,)](z, scratch, c, passes, kf, n, BLOCK=block, S=sample, SC=min(sample, 1024), C=cap,
                                    CB=min(cap, 4096), SCATTER=name == "oneread", MAX_SELECT=MAX_SELECT, num_warps=warps)
            return c, passes
        return run
    m = {"multi4": 4, "multi8": 8, "multi16": 16, "sample64+multi8": 8}.get(name, 1)
    s = int(name.split("+")[0][6:]) if name.startswith("sample") else 0
    s = s if n >= 4 * s else 0  # shorter rows skip the sample
    cap = int(name.split("finish")[1]) if "finish" in name else 0
    buf = z.new_empty(rows, max(cap, 1))

    def run():
        select_kernel[(rows,)](z, buf, c, passes, kf, n, BLOCK=block, M=m, S=s, SC=min(max(s, 1), 64), C=cap, AIM="aim" in name, BITS_FROM=0 if name == "bits" else 24,
                               MAX_SELECT=MAX_SELECT,
                               num_warps=warps)
        return c, passes
    return run


ARMS = ["minmax", "current", "bits", "multi4", "multi8", "multi16", "sample64", "sample256", "sample64+multi8",
        "finish64", "finish256", "finish1024", "sample64+finish64", "sample64+finish256", "sample64+finish1024",
        "aim+finish64", "aim+finish256", "aim+finish1024", "sample64+aim+finish256", "sample64+aim+finish1024",
        "sample1024", "finish4096", "sample256+aim+finish1024", "sample1024+aim+finish1024", "sample1024+aim+finish4096",
        "oneread1024/16384", "oneread4096/16384", "oneread4096/65536", "oneread_even", "oneread", "hybrid",
        "compact", "heap", "topk", "kthvalue"]
TUNED = set(ARMS) - {"heap", "topk", "kthvalue"}


def _time(run, repeats, budget_s):
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    times, begun = [], time.perf_counter()
    while len(times) < repeats and (len(times) < 3 or time.perf_counter() - begun < budget_s):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return times


def _rows(kind, rows, n, seed=0):
    z = torch.randn(rows, n, device="cuda", generator=torch.Generator("cuda").manual_seed(seed))
    if kind == "rounded":  # many exact ties
        return (z * 10).round() / 10
    if kind == "heavy_tail":
        return z ** 3
    if kind == "wide":  # about 70 orders of magnitude
        return torch.exp(18 * z)
    if kind == "sorted":
        return z.sort(-1).values
    if kind == "periodic":  # every 64th score is raised, which every evenly strided sample hits
        return z + 3.0 * (torch.arange(n, device="cuda") % 64 == 0)
    if kind == "periodic4096":
        return z + 3.0 * (torch.arange(n, device="cuda") % 4096 < 64)
    if kind == "halves":  # the second half of the row is shifted up
        return z + 3.0 * (torch.arange(n, device="cuda") >= n // 2)
    if kind == "descending":
        return z.sort(-1, descending=True).values
    if kind == "constant":
        return torch.ones_like(z)
    if kind == "two_values":
        return (z > 1.0).float()
    if kind == "spikes":  # a few scores far above the rest
        return z + 1e6 * (torch.rand(rows, n, device="cuda", generator=torch.Generator("cuda").manual_seed(seed + 1)) < 1e-4)
    return z


def _grid(n):
    """Block sizes and thread groups to try; those that earlier full grids chose."""
    top = max(16, triton.next_power_of_2(n))
    if n < 4096:
        return sorted({(top, 1), (min(top, 256), 1)})
    if n < 32768:
        return [(1024, 1), (1024, 2), (4096, 8)]
    if n < 262144:
        return [(1024, 2), (4096, 2), (4096, 8)]
    return [(4096, 8), (4096, 16), (4096, 32)]


def census(path, ns, ks, rows_wanted=1024):
    """Fallback rates of `oneread` over many rows, with every result checked."""
    out = []
    for kind in CENSUS_INPUTS:
        for n in ns:
            rows = MAX_BATCH
            while rows * n * 4 > MAX_INPUT_BYTES:
                rows //= 2
            seeds = max(1, -(-rows_wanted // rows))
            for arm in ("oneread", "oneread_even"):
                tally = {k: [0, 0, 0] for k in _ks(n, kind, False, ks)}
                for seed in range(seeds):
                    z = _rows(kind, rows, n, seed)
                    for k in tally:
                        c, passes = _arm(arm, z, k, _grid(n)[-1][0], _grid(n)[-1][1])()
                        wrong = int((c != z.topk(k, -1).values[:, -1]).sum())
                        assert wrong == 0, f"{arm} {kind} n={n} k={k} seed={seed}: {wrong} rows differ from torch.topk"
                        tally[k][0] += rows
                        tally[k][1] += int((passes > 1).sum())
                        tally[k][2] = max(tally[k][2], int(passes.max()))
                for k, (total, fell, worst) in tally.items():
                    out.append({"input": kind, "n": n, "k": k, "arm": arm, "rows": total, "fell_back": fell, "passes_max": worst})
                    print(f"{kind} n={n} k={k} {arm}: {fell} of {total} rows fell back, at most {worst} passes", flush=True)
                Path(path).write_text(json.dumps(out, indent=1))


def _ks(n, kind, k100, ks=None):
    if ks:  # entries are numbers or fractions of n written as "n/8"
        values = [n // int(k[2:]) if str(k).startswith("n/") else int(k) for k in ks]
        return sorted({k for k in values if 0 < k < n})
    if k100:
        return [n // 8] + ([100] if n > 100 else [])
    return sorted({min(8, n // 2), n // 8, n // 2}) if kind == "gaussian" else [n // 8]


def _settings(kind, ns, k100, ks=None):
    """(n, rows, k): rows fill 512 MB at most; with k100, long Gaussian rows are also timed in batches of 8 and 1."""
    for n in ns:
        rows = MAX_BATCH
        while rows * n * 4 > MAX_INPUT_BYTES:
            rows //= 2
        for batch in [rows] + ([8, 1] if k100 and kind == "gaussian" and n >= 262144 else []):
            for kf in _ks(n, kind, k100, ks):
                yield n, batch, kf


def collect(ns, arms, repeats, budget_s, save=lambda cells: None, k100=False, cells=(), ks=None, resume=False):
    cells = list(cells)
    done = {(c["input"], c["n"], c["batch"], c["k"], c["arm"]) for c in cells} if resume else set()
    for kind in INPUTS:
        z = None
        for n, rows, kf in _settings(kind, ns, k100, ks):
            if z is None or z.shape != (rows, n):
                z = _rows(kind, rows, n)
            if True:
                want = z.topk(kf, -1).values[:, -1]
                for name in arms:
                    if (kind, n, rows, kf, name) in done:
                        continue
                    grid = [(None, None)]
                    if name in TUNED:
                        grid = _grid(n)
                    best = None
                    for block, warps in grid:
                        run = _arm(name, z, kf, block, warps)
                        c, passes = run()
                        if c is not None:
                            wrong = int((c != want).sum())
                            assert wrong == 0, f"{name} n={n} k={kf} {kind}: {wrong} of {rows} rows differ from torch.topk"
                        quick = statistics.median(_time(run, 5, budget_s / 4))
                        if best is None or quick < best[0]:
                            best = (quick, block, warps, run, passes)
                    _, block, warps, run, passes = best
                    times = _time(run, repeats, budget_s)
                    cell = {"input": kind, "n": n, "batch": rows, "k": kf, "arm": name, "block": block, "thread_groups": warps,
                            "times_ms": times}
                    if passes is not None:
                        run()
                        cell["passes_mean"] = passes.float().mean().item()
                        cell["passes_max"] = int(passes.max())
                        if name.startswith("oneread"):
                            cell["fallback_frac"] = (passes > 1).float().mean().item()
                    cells = [c for c in cells if (c["input"], c["n"], c["batch"], c["k"], c["arm"]) != (kind, n, rows, kf, name)]
                    cells.append(cell)
                    save(cells)
                    print(f"{kind} n={n} k={kf} {name}: {statistics.median(times):.3f} ms"
                          + (f", {cell['passes_mean']:.1f} passes" if passes is not None else ""), flush=True)
    return cells


def report(path):
    data = json.loads(Path(path).read_text())
    cells = data["cells"]
    arms = [a for a in ARMS if any(c["arm"] == a for c in cells)]
    groups = []
    for c in cells:
        key = (c["input"], c["n"], c["batch"], c["k"])
        if key not in groups:
            groups.append(key)
    print(f"\nmedian time in ms (mean passes over the row after the min/max pass), {data['meta']}")
    for kind in INPUTS:
        cols = [g for g in groups if g[0] == kind]
        if not cols:
            continue
        print(f"\n{kind}\n\n| arm | " + " | ".join(f"n={n} x {b}, k={k}" for _, n, b, k in cols) + " |")
        print("| --- |" + " --- |" * len(cols))
        for a in arms:
            texts = []
            for _, n, b, k in cols:
                cell = next((c for c in cells if (c["input"], c["n"], c["batch"], c["k"], c["arm"]) == (kind, n, b, k, a)), None)
                text = "" if cell is None else f"{statistics.median(cell['times_ms']):.3f}"
                if cell is not None and "passes_mean" in cell:
                    text += f" ({cell['passes_mean']:.1f})"
                if cell is not None and cell.get("fallback_frac"):
                    text += f" [{100 * cell['fallback_frac']:.0f}% fell back]"
                texts.append(text)
            print(f"| {a} | " + " | ".join(texts) + " |")


PLOT_KS = [("100", "k = 100"), ("1000", "k = 1000"), ("10000", "k = 10,000"), ("n/32", "k = n/32"), ("n/8", "k = n/8"), ("n/2", "k = n/2")]
PLOT_ARMS = [("minmax", "one pass over the data", "0.55", "--"), ("current", "current rule", "#d95f02", "-"),
             ("oneread", "one-read selection", "#1b9e77", "-"), ("topk", "torch.topk", "#7570b3", "-")]
PLOT_INPUTS = ["gaussian", "rounded", "heavy_tail", "wide", "sorted", "periodic", "halves"]


def plot(path, out):
    """Six panels, one per k: time per score against row length for full batches. Thick lines
    are Gaussian rows; each thin line is one of the other six timed input families."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = json.loads(Path(path).read_text())["cells"]
    full = {}
    for c in cells:
        full[c["n"]] = max(full.get(c["n"], 0), c["batch"])

    def per_score(kind, n, k, arm):  # nanoseconds
        cell = next((c for c in cells if (c["input"], c["n"], c["batch"], c["k"], c["arm"]) == (kind, n, full[n], k, arm)), None)
        return None if cell is None else statistics.median(cell["times_ms"]) * 1e6 / (full[n] * n)

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4), sharey=True)
    for ax, (key, label) in zip(axes.flat, PLOT_KS):
        for arm, name, color, style in PLOT_ARMS:
            # thin lines for the other input families, then the Gaussian rows on top
            for kind in ([] if arm == "minmax" else PLOT_INPUTS[1:]) + ["gaussian"]:
                points = []
                for n in sorted(full):
                    k = _ks(n, "gaussian", False, [key])
                    if k and per_score(kind, n, k[0], arm) is not None:
                        points.append((n, per_score(kind, n, k[0], arm)))
                if not points:
                    continue
                if kind == "gaussian":
                    ax.plot(*zip(*points), style, color=color, linewidth=1.5, marker="o", markersize=2.5, label=name)
                else:
                    ax.plot(*zip(*points), "-", color=color, linewidth=0.5, alpha=0.55)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(48, 2**21.4)
        ax.set_xticks([2**6, 2**9, 2**12, 2**15, 2**18, 2**21])
        ax.set_xlabel(f"scores per row n ({label})")
    for ax in axes[:, 0]:
        ax.set_ylabel("time per score (ns), median of 20 calls")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--report", metavar="FILE")
    ap.add_argument("--n", type=int, nargs="+", default=NS)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--plot", nargs=2, metavar=("FILE", "OUT"), help="draw the six-panel figure from a saved file")
    ap.add_argument("--census", metavar="FILE", help="count fallbacks of the oneread arms over many rows")
    ap.add_argument("--resume", action="store_true", help="with --merge, skip the cells already in --out")
    ap.add_argument("--merge", action="store_true", help="keep the cells already in --out, replacing those re-timed")
    ap.add_argument("--k100", action="store_true", help="use k = n/8 and k = 100, and small batches of long rows")
    ap.add_argument("--k", nargs="+", help='time these k (those under n) in place of the default ones, e.g. 100 n/8')
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--budget", type=float, default=10.0, help="seconds per arm before stopping early (at least 3 calls)")
    a = ap.parse_args()
    if a.out:
        meta = {"torch": torch.__version__, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(0),
                "repeats": a.repeats}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(Path(a.out).read_text())["cells"] if a.merge and Path(a.out).exists() else []
        collect(a.n, a.arms, a.repeats, a.budget,
                lambda cells: Path(a.out).write_text(json.dumps({"meta": meta, "cells": cells}, indent=1)), a.k100, old, a.k, a.resume)
        report(a.out)
    if a.census:
        census(a.census, a.n, a.k or ["100", "1000", "10000", "n/8", "n/2"])
    if a.report:
        report(a.report)
    if a.plot:
        plot(*a.plot)
