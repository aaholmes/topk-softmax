"""Forward and backward time of the library on the GPU, over row length, k and kind of input.

    python benchmarks/sweep.py --out benchmarks/results/sweep.json      # time the forced settings (about 40 min)
    python benchmarks/sweep.py --out benchmarks/results/sweep.json --library   # then the library as configured
    python benchmarks/sweep.py --report benchmarks/results/sweep.json   # best setting per row length and k
    python benchmarks/sweep.py --plot benchmarks/results/sweep.json docs  # docs/speed.png and docs/selection.png

With flat_top=True the forward kernel has two ways to find each row's k-th largest score: counting
passes over the row (`count`), or one pass that copies the scores between two sampled pivots
(`oneread`). Each is timed with several block sizes and thread groups per row, written
`count/4096/8`. `plain` is flat_top=False, which needs no k-th score, and `softmax` is
torch.softmax. `library` is the operator as the package configures it; for it the backward pass is
also timed alone, as one torch.autograd.grad call on a saved output.

Batches hold at most 512 MB of scores. Times are medians of 20 calls.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import topk_softmax
from topk_softmax import _triton

NS = [2**e for e in range(9, 22)]
KS = ["100", "1000", "10000", "n/32", "n/8", "n/2"]
MAX_BATCH = 4096
MAX_INPUT_BYTES = 2**29
INPUTS = ["gaussian", "ties", "heavy_tails"]


def _rows(kind, rows, n):
    z = torch.randn(rows, n, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    if kind == "ties":
        return (z * 10).round() / 10
    if kind == "heavy_tails":
        z = z ** 3
        return z / z.std(-1, keepdim=True)
    return z


def _k(n, key):
    return n // int(key[2:]) if key.startswith("n/") else int(key)


def _settings(n):
    """(block, thread groups) to try for rows of n scores."""
    if n <= 1024:
        return [(4096, 1), (4096, 2), (4096, 4)]
    if n < 4096:
        return [(4096, 2), (4096, 4), (4096, 8)]
    if n < 32768:
        return [(1024, 2), (4096, 2), (4096, 4), (4096, 8)]
    if n <= 65536:
        return [(1024, 2), (4096, 4), (4096, 8), (4096, 16), (4096, 32)]
    return [(4096, 8), (4096, 16), (4096, 32)]


def _median_ms(call, repeats, budget_s=15.0):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    times, begun = [], time.perf_counter()
    while len(times) < repeats and (len(times) < 3 or time.perf_counter() - begun < budget_s):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def _forward(z, k, tau, flat_top=True):
    def call():
        with torch.no_grad():
            topk_softmax.soft_topk_softmax(z, k, tau, backend="triton", flat_top=flat_top)
    return call


def _backward(fn, z):
    x = z.clone().requires_grad_()
    p = fn(x)
    g = torch.randn_like(z)
    return lambda: torch.autograd.grad(p, x, g, retain_graph=True)


def collect(path, ns, library, repeats):
    cells = json.loads(Path(path).read_text())["cells"] if Path(path).exists() else []
    done = {(c["input"], c["n"], c["k"], c["tau"], c["arm"]) for c in cells}
    default_plan = _triton._forward_plan

    def record(kind, n, rows, key, tau, arm, ms):
        cells.append({"input": kind, "n": n, "batch": rows, "k": key, "tau": tau, "arm": arm, "ms": ms})
        meta = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "repeats": repeats}
        Path(path).write_text(json.dumps({"meta": meta, "cells": cells}, indent=1))
        print(f"{kind} n={n} k={key} tau={tau} {arm}: {ms:.3f} ms", flush=True)

    for kind in INPUTS:
        for n in ns:
            rows = MAX_BATCH
            while rows * n * 4 > MAX_INPUT_BYTES:
                rows //= 2
            z = None
            for key in KS if kind == "gaussian" else ["100", "n/8"]:
                k = _k(n, key)
                if not 0 < k < n:
                    continue
                for tau in [1.0, 0.01] if kind == "gaussian" and key in ("100", "n/8") else [1.0]:
                    arms = {}
                    if library:
                        _triton._forward_plan = default_plan
                        arms["library"] = lambda: _forward(z, k, tau)
                        arms["library backward"] = lambda: _backward(
                            lambda x: topk_softmax.soft_topk_softmax(x, k, tau, backend="triton"), z)
                        arms["plain library"] = lambda: _forward(z, k, tau, False)
                        if key == "n/8" and tau == 1.0:
                            arms["softmax"] = lambda: lambda: torch.softmax(z, -1)
                            arms["softmax backward"] = lambda: _backward(lambda x: torch.softmax(x, -1), z)
                    else:
                        for block, groups in _settings(n):
                            for rule in ("count", "oneread"):
                                if rule == "oneread" and n < 4096:
                                    continue
                                arms[f"{rule}/{block}/{groups}"] = (
                                    lambda rule=rule, block=block, groups=groups: _forced(z, k, tau, rule, block, groups))
                            if key == "n/8":
                                arms[f"plain/{block}/{groups}"] = (
                                    lambda block=block, groups=groups: _forced(z, k, tau, "plain", block, groups))
                    for arm, make in arms.items():
                        if (kind, n, key, tau, arm) in done:
                            continue
                        if z is None:
                            z = _rows(kind, rows, n)
                        record(kind, n, rows, key, tau, arm, _median_ms(make(), repeats))
    _triton._forward_plan = default_plan


def _forced(z, k, tau, rule, block, groups):
    def call():
        _triton._forward_plan = lambda n, flat_k: (rule == "oneread", block, groups)
        with torch.no_grad():
            topk_softmax.soft_topk_softmax(z, k, tau, backend="triton", flat_top=rule != "plain")
    return call


def _load(path):
    return {(c["input"], c["n"], c["k"], c["tau"], c["arm"]): c for c in json.loads(Path(path).read_text())["cells"]}


def best(cells, kind, n, key, tau, rule):
    """(ms, arm) of the fastest forced setting of a rule, or None."""
    found = [(c["ms"], arm) for (i, m, kk, t, arm), c in cells.items()
             if (i, m, kk, t) == (kind, n, key, tau) and arm.startswith(rule + "/")]
    return min(found) if found else None


def report(path):
    cells = _load(path)
    ns = sorted({key[1] for key in cells})
    for kind in INPUTS:
        for tau in (1.0, 0.01):
            keys = [k for k in KS if any((kind, n, k, tau, a) in cells for n in ns for a in ("library", "count/4096/8"))]
            if not keys:
                continue
            print(f"\n{kind}, tau = {tau}: forward ms as best count / best oneread (settings of the faster) / library\n")
            print("| n | " + " | ".join(f"k = {k}" for k in keys) + " |\n| --- |" + " --- |" * len(keys))
            for n in ns:
                row = []
                for key in keys:
                    c, o, lib = best(cells, kind, n, key, tau, "count"), best(cells, kind, n, key, tau, "oneread"), cells.get((kind, n, key, tau, "library"))
                    if c is None:
                        row.append("")
                        continue
                    win = min(x for x in (c, o) if x)
                    text = f"{c[0]:.2f} / " + (f"{o[0]:.2f}" if o else "–") + f" ({win[1]})"
                    row.append(text + (f" / {lib['ms']:.2f}" if lib else ""))
                print(f"| {n} | " + " | ".join(row) + " |")
    gaps = []
    for (kind, n, key, tau, arm), c in cells.items():
        if arm == "library":
            forced = [x for x in (best(cells, kind, n, key, tau, "count"), best(cells, kind, n, key, tau, "oneread")) if x]
            if forced:
                gaps.append((c["ms"] / min(forced)[0] - 1, kind, n, key, tau))
    if gaps:
        gaps.sort()
        within = sum(g[0] <= 0.05 for g in gaps)
        print(f"\nlibrary compared with the best forced setting: within 5% on {within} of {len(gaps)} cells; "
              f"median {100 * statistics.median(g[0] for g in gaps):+.1f}%, worst {100 * gaps[-1][0]:+.1f}% at {gaps[-1][1:]}")


def plot(path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = _load(path)
    ns = sorted({key[1] for key in cells})
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

    def per_score(n, key, arm, tau=1.0):  # nanoseconds
        c = cells.get(("gaussian", n, key, tau, arm))
        return None if c is None else c["ms"] * 1e6 / (c["batch"] * n)

    def line(ax, key, arm, **style):
        points = [(n, per_score(n, key, arm)) for n in ns if per_score(n, key, arm) is not None]
        if points:
            ax.plot(*zip(*points), **style)

    # time per score against row length: forward and backward, compared with torch.softmax
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
    for ax, suffix, word in zip(axes, ("", " backward"), ("forward", "backward")):
        for key in KS:
            if key != "n/8":
                line(ax, key, "library" + suffix, color="#1b9e77", linewidth=0.5, alpha=0.55)
        line(ax, "n/8", "library" + suffix, color="#1b9e77", linewidth=1.5, marker="o", markersize=2.5, label="soft_topk_softmax")
        if not suffix:
            line(ax, "n/8", "plain library", color="#d95f02", linewidth=1.2, marker="o", markersize=2.5, label="flat_top=False")
        line(ax, "n/8", "softmax" + suffix, color="0.45", linestyle="--", linewidth=1.2, marker="o", markersize=2.5, label="torch.softmax")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([2**9, 2**12, 2**15, 2**18, 2**21])
        ax.set_xlabel(f"scores per row n ({word} pass)")
    axes[0].set_ylabel("time per score (ns), median of 20 calls")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(Path(out) / "speed.png", dpi=150)
    plt.close(fig)

    # where finding the k-th score in the first pass overtakes counting passes
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    colors = plt.get_cmap("viridis")
    for i, key in enumerate(KS):
        points = []
        for n in ns:
            c, o = best(cells, "gaussian", n, key, 1.0, "count"), best(cells, "gaussian", n, key, 1.0, "oneread")
            if c and o:
                points.append((n, o[0] / c[0]))
        if points:
            ax.plot(*zip(*points), color=colors(i / (len(KS) - 1) * 0.9), linewidth=1.2, marker="o", markersize=2.5, label=f"k = {key}")
    ax.axhline(1.0, color="0.8", linewidth=0.6, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1, 2], ["0.25", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_xticks([2**12, 2**15, 2**18, 2**21])
    ax.set_xlabel("scores per row n")
    ax.set_ylabel("forward time, one pass ÷ counting passes")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(Path(out) / "selection.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--library", action="store_true", help="time the library as configured instead of the forced settings")
    ap.add_argument("--report", metavar="FILE")
    ap.add_argument("--plot", nargs=2, metavar=("FILE", "DIR"))
    ap.add_argument("--n", type=int, nargs="+", default=NS)
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    if a.out:
        collect(a.out, a.n, a.library, a.repeats)
        report(a.out)
    if a.report:
        report(a.report)
    if a.plot:
        plot(*a.plot)
