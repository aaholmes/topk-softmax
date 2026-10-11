"""Error versus iteration for several root-finders for the threshold t.

    python benchmarks/solvers.py --out benchmarks/results/solvers.json     # collect (about 5 min, CPU)
    python benchmarks/solvers.py --report benchmarks/results/solvers.json  # tables from a saved file
    python benchmarks/solvers.py --plot benchmarks/results/solvers.json DIR
    python benchmarks/solvers.py --time bisection:20 newton_progress:12    # compiled GPU forward time

The equation is f(t) = sum_i sigmoid((z_i - t)/tau) - k = 0, with f decreasing in t. One iteration
is one evaluation of f, which is one pass over the row. Every solver runs a fixed number of
iterations with no early exit. Each is swapped into the library in place of its bisection, so it
sees the centered scores and the floored tau, and so the "polished" error includes the library's
two final Newton steps on u = (z - t)/tau.
"""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

import softk
import softk._autograd as autograd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from _reference import hard_cases  # noqa: E402

TAUS = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]  # in units of the row standard deviation
NS = [64, 1024, 32768]
ROWS = 16
MAX_ITERS = {torch.float32: 32, torch.float64: 60}


def tolerance(k, dtype):
    """Allowed |sum(m) - k|. In 32-bit floats the sum near k is spaced k * 1.2e-7 apart, and no
    solver gets the residual below a few of those spacings, so the tolerance is 8 of them."""
    return 1e-10 if dtype in (torch.float64, "float64") else max(1e-5, 1e-6 * k)


def _f(z, t, k, tau):
    s = torch.sigmoid((z - t) / tau)
    return s.sum(-1, keepdim=True) - k, (s * (1 - s)).sum(-1, keepdim=True)


def _bracket(z, k, tau):
    c = tau * math.log((k / z.shape[-1]) / (1 - k / z.shape[-1]))
    return z.amin(-1, keepdim=True) - c, z.amax(-1, keepdim=True) - c


def bisection(z, k, tau, iters, trace=None):
    lo, hi = _bracket(z, k, tau)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f, _ = _f(z, mid, k, tau)
        lo, hi = torch.where(f > 0, mid, lo), torch.where(f > 0, hi, mid)
        if trace is not None:
            trace.append(0.5 * (lo + hi))
    return 0.5 * (lo + hi)


def _newton(z, k, tau, iters, trace, progress_check, t=None):
    lo, hi = _bracket(z, k, tau)
    t = 0.5 * (lo + hi) if t is None else t
    step_before_last = hi - lo
    last_step = hi - lo
    for _ in range(iters):
        f, sum_v = _f(z, t, k, tau)
        lo, hi = torch.where(f > 0, t, lo), torch.where(f > 0, hi, t)
        step = tau * f / sum_v.clamp_min(1e-30)
        # inclusive: at convergence the step is zero and t is itself an end of the bracket
        ok = (t + step >= lo) & (t + step <= hi)
        if progress_check:
            # reject a step that is not at least halving the one before last (as in rtsafe)
            ok = ok & (step.abs() <= 0.5 * step_before_last.abs())
        t_new = torch.where(ok, t + step, 0.5 * (lo + hi))
        step_before_last, last_step = last_step, t_new - t
        t = t_new
        if trace is not None:
            trace.append(t)
    return t


def newton_bracket(z, k, tau, iters, trace=None):
    return _newton(z, k, tau, iters, trace, progress_check=False)


def newton_progress(z, k, tau, iters, trace=None):
    return _newton(z, k, tau, iters, trace, progress_check=True)


def _topk_start(z, k):
    # as tau -> 0 the root lies between the (floor(k)+1)-th and the ceil(k)-th largest scores
    top = z.topk(math.floor(k) + 1, -1).values
    return 0.5 * (top[..., -1:] + top[..., math.ceil(k) - 1:math.ceil(k)])


def newton_topk_start(z, k, tau, iters, trace=None):
    return _newton(z, k, tau, iters, trace, progress_check=True, t=_topk_start(z, k))


def newton_bracket_topk_start(z, k, tau, iters, trace=None):
    """Added after seeing the first results, to separate the start from the progress check."""
    return _newton(z, k, tau, iters, trace, progress_check=False, t=_topk_start(z, k))


def illinois(z, k, tau, iters, trace=None):
    """Bracketing secant method. Its first two evaluations are the ends of the bracket."""
    a, b = _bracket(z, k, tau)

    def secant(a, fa, b, fb):
        x = (a * fb - b * fa) / (fb - fa)
        inside = torch.isfinite(x) & (x >= torch.minimum(a, b)) & (x <= torch.maximum(a, b))
        return torch.where(inside, x, 0.5 * (a + b))

    x = 0.5 * (a + b)
    fa = fb = None
    for i in range(iters):
        if i == 0:
            fa, _ = _f(z, a, k, tau)
        elif i == 1:
            fb, _ = _f(z, b, k, tau)
            x = secant(a, fa, b, fb)
        else:
            fx, _ = _f(z, x, k, tau)
            crossed = fx * fb < 0  # the root is between x and b: keep b as the far end
            a, fa = torch.where(crossed, b, a), torch.where(crossed, fb, 0.5 * fa)
            b, fb = x, fx
            x = secant(a, fa, b, fb)
        if trace is not None:
            trace.append(x)
    return x


SOLVERS = {"bisection": bisection, "newton_bracket": newton_bracket, "newton_progress": newton_progress,
           "illinois": illinois, "newton_topk_start": newton_topk_start,
           "newton_bracket_topk_start": newton_bracket_topk_start}
LABELS = {"bisection": "bisection", "newton_bracket": "Newton, bracket check only",
          "newton_progress": "Newton with progress check", "illinois": "Illinois",
          "newton_topk_start": "Newton with progress check, top-k start",
          "newton_bracket_topk_start": "Newton, bracket check only, top-k start"}


def _random_rows(n, dtype):
    g = torch.Generator().manual_seed(n)
    z = torch.randn(ROWS, n, generator=g, dtype=torch.float64)
    z = (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True, correction=0)
    return z.to(dtype)


def _cases(dtype):
    for n in NS:
        z = _random_rows(n, dtype)
        for k in sorted({1, n // 8, n // 2, n - 1}):
            for tau in TAUS:
                yield dict(group="random", n=n, k=k, tau=tau), z, k, tau
    for name, z, k, tau in hard_cases(dtype):
        yield dict(group="hard", n=z.shape[-1], k=k, tau=tau, name=name), z, k, tau


def _summary(errors):
    """errors: one tensor of per-row errors per iteration."""
    r = lambda x: float(f"{x:.3g}")
    return {"worst": [r(e.max().item()) for e in errors],
            "p90": [r(e.quantile(0.9).item()) for e in errors],
            "median": [r(e.median().item()) for e in errors]}


def collect():
    records = []
    original = autograd._solve_threshold
    try:
        for dtype in MAX_ITERS:
            n_iters = MAX_ITERS[dtype]
            for meta, z, k, tau in _cases(dtype):
                for name, solver in SOLVERS.items():
                    seen = {}

                    def record(zc, k_, tau_, iters):
                        seen["trace"] = []
                        seen["z"], seen["tau"] = zc, tau_
                        return solver(zc, k_, tau_, iters, seen["trace"])

                    autograd._solve_threshold = record
                    softk.soft_topk_mask(z, k, tau, iters=n_iters)
                    raw = [_f(seen["z"], t, k, seen["tau"])[0].double().abs().flatten() for t in seen["trace"]]
                    polished = []
                    for t in seen["trace"]:
                        autograd._solve_threshold = lambda *a, t=t: t
                        m = softk.soft_topk_mask(z, k, tau, iters=n_iters)
                        polished.append((m.double().sum(-1) - k).abs().flatten())
                    records.append(dict(meta, solver=name, dtype=str(dtype).split(".")[-1], rows=z.shape[0],
                                        raw=_summary(raw), polished=_summary(polished)))
            print(f"collected {dtype}", flush=True)
    finally:
        autograd._solve_threshold = original
    return records


def _iters_to_tol(errors, tol):
    """Smallest iteration count from which the error stays within tol; None if it never does."""
    needed = None
    for i in range(len(errors), 0, -1):
        if errors[i - 1] > tol:
            break
        needed = i
    return needed


def _needed(records, solver, dtype, key, group, tau=None, stat="worst"):
    """Worst case over the matching records of the iterations needed; None if any never converges."""
    counts = [_iters_to_tol(r[key][stat], tolerance(r["k"], r["dtype"])) for r in records
              if r["solver"] == solver and r["dtype"] == dtype and r["group"] == group
              and (tau is None or r["tau"] == tau)]
    return None if None in counts else max(counts)


def report(path):
    records = json.loads(Path(path).read_text())["records"]
    fmt = lambda c, limit: f">{limit}" if c is None else str(c)
    for dtype, limit in (("float32", 32), ("float64", 60)):
        for key in ("polished", "raw"):
            what = "after the two final Newton steps" if key == "polished" else "solver alone"
            print(f"\n{dtype}, {what}: iterations until the worst row stays within tolerance "
                  f"(random rows: n in {NS}, 4 values of k, {ROWS} rows each)\n")
            print("| solver | " + " | ".join(f"tau={t:g}" for t in TAUS) + " | all tau | hard cases |")
            print("| --- |" + " --- |" * (len(TAUS) + 2))
            for s in SOLVERS:
                per_tau = [_needed(records, s, dtype, key, "random", t) for t in TAUS]
                cells = [fmt(c, limit) for c in per_tau]
                cells.append(fmt(None if None in per_tau else max(per_tau), limit))
                cells.append(fmt(_needed(records, s, dtype, key, "hard"), limit))
                print(f"| {LABELS[s]} | " + " | ".join(cells) + " |")
    print("\nfloat32, after the two final Newton steps, median row instead of worst row\n")
    print("| solver | " + " | ".join(f"tau={t:g}" for t in TAUS) + " |")
    print("| --- |" + " --- |" * len(TAUS))
    for s in SOLVERS:
        print(f"| {LABELS[s]} | " + " | ".join(
            fmt(_needed(records, s, "float32", "polished", "random", t, "median"), 32) for t in TAUS) + " |")


def plot(path, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = json.loads(Path(path).read_text())["records"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = ROWS * sum(len({1, n // 8, n // 2, n - 1}) for n in NS)
    panel_taus = [1e-4, 1e-2, 1.0, 100.0]

    fig, axes = plt.subplots(1, len(panel_taus), figsize=(10, 2.9), sharey=True)
    for ax, tau in zip(axes, panel_taus):
        for s in SOLVERS:
            rs = [r for r in records if r["solver"] == s and r["dtype"] == "float32"
                  and r["group"] == "random" and r["tau"] == tau]
            ratio = [max(r["polished"]["worst"][i] / tolerance(r["k"], "float32") for r in rs) for i in range(32)]
            ax.plot(range(1, 33), ratio, lw=1.4, label=LABELS[s])
        ax.axhline(1.0, color="0.7", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(f"iterations (τ = {tau:g})")
    axes[0].set_ylabel(f"worst |Σm − k| / tolerance\n(32-bit, n = {rows} rows)", fontsize=9)
    axes[-1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(out_dir / "solver_error_vs_iteration.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for s in SOLVERS:
        counts = [_needed(records, s, "float32", "polished", "random", t) for t in TAUS]
        ax.plot(TAUS, [33 if c is None else c for c in counts], marker="o", ms=3, lw=1.4, label=LABELS[s])
    ax.set_xscale("log")
    ax.set_xlabel("τ (row standard deviations)")
    ax.set_ylabel(f"iterations to tolerance\n(worst of n = {rows} rows)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "solver_iterations_vs_tau.png", dpi=150, bbox_inches="tight")


def time_compiled(specs, ns=(1024, 16384), batch=4096, tau=1.0):
    original = autograd._solve_threshold
    try:
        for n in ns:
            z = torch.randn(batch, n, generator=torch.Generator().manual_seed(0)).cuda()
            for spec in specs:
                name, iters = spec.split(":")
                autograd._solve_threshold = lambda zc, k, t, i, solver=SOLVERS[name]: solver(zc, k, t, i)
                torch._dynamo.reset()
                fn = torch.compile(softk.soft_topk_softmax)
                with torch.no_grad():
                    for _ in range(5):
                        fn(z, n // 8, tau, iters=int(iters))
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(20):
                        t0 = time.perf_counter()
                        fn(z, n // 8, tau, iters=int(iters))
                        torch.cuda.synchronize()
                        times.append(1e3 * (time.perf_counter() - t0))
                times.sort()
                print(f"n={n} batch={batch} {name} iters={iters}: compiled forward "
                      f"{statistics.median(times):.3f} ms [{times[2]:.3f}-{times[17]:.3f}], n = 20 calls")
    finally:
        autograd._solve_threshold = original
        torch._dynamo.reset()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--report", metavar="FILE")
    ap.add_argument("--plot", nargs=2, metavar=("FILE", "DIR"))
    ap.add_argument("--time", nargs="+", metavar="SOLVER:ITERS")
    a = ap.parse_args()
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"torch": torch.__version__, "records": collect()}, separators=(",", ":")))
        report(a.out)
    if a.report:
        report(a.report)
    if a.plot:
        plot(*a.plot)
    if a.time:
        time_compiled(a.time)


if __name__ == "__main__":
    main()
