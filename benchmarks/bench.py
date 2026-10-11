"""Timing, profiling and solver-accuracy measurements for topk_softmax.

    python benchmarks/bench.py --out benchmarks/results/NAME.json    # time the grid, save raw timings
    python benchmarks/bench.py --report benchmarks/results/NAME.json # print tables from a saved file
    python benchmarks/bench.py --profile                             # kernel launches of one eager call
    python benchmarks/bench.py --iters                               # mask-sum error versus solver iterations

Arms: `softmax` is torch.softmax, the lower bound for one pass over the data. `eager` is topk_softmax
without compilation, `compiled` is torch.compile of the eager code, and `triton` is the fused kernel
(skipped until the `backend` argument exists). All arms in a cell run on one input tensor.
`copy` times a tensor copy of the input, as a measured stand-in for the memory-bandwidth limit.
"""
import argparse
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import topk_softmax

NS = [64, 1024, 16384, 131072]
TAUS = [1.0, 0.01]
MAX_BATCH = 4096
MAX_INPUT_BYTES = 2**29  # the batch is halved until the input fits
ARMS = ["copy", "softmax", "eager", "compiled", "triton"]
HAS_BACKEND = "backend" in inspect.signature(topk_softmax.soft_topk_softmax).parameters


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _make_arm(arm, k, tau):
    """A function z -> p for the arm, or None if the arm cannot run here."""
    eager_kw = {"backend": "eager"} if HAS_BACKEND else {}
    if arm == "softmax":
        return lambda z: torch.softmax(z, -1)
    if arm == "eager":
        return lambda z: topk_softmax.soft_topk_softmax(z, k, tau, **eager_kw)
    if arm == "compiled":
        torch._dynamo.reset()
        fn = torch.compile(topk_softmax.soft_topk_softmax)
        return lambda z: fn(z, k, tau, **eager_kw)
    if arm == "triton":
        if not HAS_BACKEND:
            return None
        return lambda z: topk_softmax.soft_topk_softmax(z, k, tau, backend="triton")
    raise ValueError(arm)


def _time(call, device, repeats, budget_s):
    for _ in range(5):
        call()
    _sync(device)
    times, start = [], time.perf_counter()
    while len(times) < repeats and (len(times) < 3 or time.perf_counter() - start < budget_s):
        t0 = time.perf_counter()
        call()
        _sync(device)
        times.append(1e3 * (time.perf_counter() - t0))
    return times


def collect(devices, ns, taus, arms, repeats, budget_s):
    cells = []
    for device in devices:
        for n in ns:
            batch = MAX_BATCH
            while batch * n * 4 > MAX_INPUT_BYTES:
                batch //= 2
            g = torch.Generator().manual_seed(0)
            z = torch.randn(batch, n, generator=g).to(device)
            w = torch.randn(batch, n, generator=g).to(device)
            k = max(1, n // 8)
            for tau in taus:
                for arm in arms:
                    if arm == "triton" and device != "cuda":
                        continue
                    if arm in ("copy", "softmax") and tau != taus[0]:
                        continue  # these do not depend on tau
                    base = dict(device=device, n=n, batch=batch, k=k, tau=tau, arm=arm)
                    if arm == "copy":
                        dst = torch.empty_like(z)
                        cells.append(dict(base, direction="forward",
                                          times_ms=_time(lambda: dst.copy_(z), device, repeats, budget_s)))
                        del dst
                        continue
                    fn = _make_arm(arm, k, tau)
                    if fn is None:
                        continue

                    def forward():
                        with torch.no_grad():
                            fn(z)

                    def forward_backward():
                        x = z.detach().requires_grad_()
                        (fn(x) * w).sum().backward()

                    for direction, call in (("forward", forward), ("forward+backward", forward_backward)):
                        if device == "cuda":
                            torch.cuda.empty_cache()
                            torch.cuda.reset_peak_memory_stats()
                            before = torch.cuda.memory_allocated()
                        times = _time(call, device, repeats, budget_s)
                        peak = torch.cuda.max_memory_allocated() - before if device == "cuda" else None
                        cells.append(dict(base, direction=direction, times_ms=times, peak_extra_bytes=peak))
                        print(f"{device} n={n} tau={tau:g} {arm} {direction}: "
                              f"{statistics.median(times):.3f} ms (n = {len(times)})", flush=True)
            del z, w
    return cells


def _meta():
    meta = {"torch": torch.__version__, "has_backend": HAS_BACKEND, "cpu_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["cuda"] = torch.version.cuda
    try:
        import triton
        meta["triton"] = triton.__version__
    except ImportError:
        meta["triton"] = None
    return meta


def _q(times, q):
    s = sorted(times)
    return s[min(len(s) - 1, int(q * len(s)))]


def report(path):
    data = json.loads(Path(path).read_text())
    print("meta:", json.dumps(data["meta"]))
    cells = data["cells"]
    med = {(c["device"], c["n"], c["tau"], c["arm"], c["direction"]): statistics.median(c["times_ms"]) for c in cells}
    tau0 = next(c["tau"] for c in cells if c["arm"] == "softmax")
    print("\nMedian time in ms [10th-90th percentile], input throughput in GB/s, time relative to softmax, "
          "and peak extra GPU memory relative to the input size.\n")
    print("| device | n | batch | tau | direction | arm | ms | GB/s | x softmax | x copy | peak mem / input | repeats |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in cells:
        t = c["times_ms"]
        m = statistics.median(t)
        nbytes = c["n"] * c["batch"] * 4
        soft = med.get((c["device"], c["n"], tau0, "softmax", c["direction"]))
        copy = med.get((c["device"], c["n"], tau0, "copy", "forward"))
        mem = c.get("peak_extra_bytes")
        print(f"| {c['device']} | {c['n']} | {c['batch']} | {c['tau']:g} | {c['direction']} | {c['arm']} | "
              f"{m:.3f} [{_q(t, 0.1):.3f}-{_q(t, 0.9):.3f}] | {nbytes / 1e9 / (m / 1e3):.2f} | "
              f"{m / soft:.1f} | {m / copy:.1f} | {'' if mem is None else f'{mem / nbytes:.1f}'} | {len(t)} |")


def profile(n=1024, tau=1.0):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile as torch_profile

    z = torch.randn(MAX_BATCH, n, device="cuda")
    k = n // 8
    for name, fn in (("softmax", lambda: torch.softmax(z, -1)),
                     ("eager topk_softmax", lambda: topk_softmax.soft_topk_softmax(z, k, tau))):
        with torch.no_grad():
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                fn()
                torch.cuda.synchronize()
        kernels = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
        launches = [e for e in kernels if not e.name.lower().startswith(("memcpy", "memset"))]
        print(f"\n{name}, forward, n = {n}, batch = {MAX_BATCH}: {len(launches)} GPU kernel launches, "
              f"{sum(e.device_time for e in launches) / 1e3:.3f} ms of GPU kernel time")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=40))


def iters_sweep():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from _reference import hard_cases

    counts = [4, 6, 8, 10, 12, 16, 20, 24, 32, 60]
    for dtype in (torch.float32, torch.float64):
        print(f"\n{dtype}: worst |sum(m) - k| over rows, by solver iterations\n")
        print("| case | " + " | ".join(str(c) for c in counts) + " |")
        print("| --- |" + " --- |" * len(counts))
        worst = [0.0] * len(counts)
        for name, z, k, tau in hard_cases(dtype):
            errs = [(topk_softmax.soft_topk_mask(z, k, tau, iters=c).double().sum(-1) - k).abs().max().item() for c in counts]
            worst = [max(a, b) for a, b in zip(worst, errs)]
            print(f"| {name} | " + " | ".join(f"{e:.1e}" for e in errs) + " |")
        print("| **worst case** | " + " | ".join(f"{e:.1e}" for e in worst) + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="time the grid and write raw timings to this JSON file")
    ap.add_argument("--report", metavar="FILE", help="print tables from a saved JSON file")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--iters", action="store_true")
    ap.add_argument("--devices", nargs="+", default=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
    ap.add_argument("--n", type=int, nargs="+", default=NS)
    ap.add_argument("--tau", type=float, nargs="+", default=TAUS)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--budget", type=float, default=20.0, help="seconds per cell before stopping early (at least 3 repeats)")
    a = ap.parse_args()
    if a.out:
        cells = collect(a.devices, a.n, a.tau, a.arms, a.repeats, a.budget)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"meta": _meta(), "cells": cells}, indent=1))
        report(a.out)
    if a.report:
        report(a.report)
    if a.profile:
        profile()
    if a.iters:
        iters_sweep()


if __name__ == "__main__":
    main()
