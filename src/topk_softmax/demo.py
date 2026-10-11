"""Figures showing how p = soft_topk_softmax(z, k, tau) moves from softmax to top-k softmax.

    python -m topk_softmax.demo --k 1 2 4 --input spread --out figs/

writes figs/sweep_spread.png: one panel per k, one line per output entry p_i, with tau on a log
axis. The markers at the right edge are softmax(z), the large-tau limit.

    --flow PATH    one input and one k, with tau decreasing left to right (the README figure)
    --train PATH   a toy training run with tau annealed by TauScheduler

Needs matplotlib, which the `demo` extra installs.
"""
import argparse
import math
import os

import torch

from . import SoftTopKSoftmax, TauScheduler, soft_topk_softmax

INPUTS = {
    "spread": torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5]),
    "tie": torch.tensor([1.0, 0.95, 0.9, 0.85, -1.0, -1.1, -1.2, -1.3]),
    "spike": torch.tensor([4.0, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]),
    "random": torch.randn(8, generator=torch.Generator().manual_seed(0)),
}


def sweep(plt, out, name, ks, tau_min, tau_max):
    z = INPUTS[name].double()
    n = z.numel()
    taus = torch.logspace(math.log10(tau_min), math.log10(tau_max), 200, dtype=torch.float64)
    sm = torch.softmax(z, -1)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(ks), figsize=(2.4 * len(ks) + 1.6, 2.8), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], ks):
        P = torch.stack([soft_topk_softmax(z, k, float(t)) for t in taus])
        for i in range(n):
            color = cmap(i / (n - 1))
            ax.plot(taus, P[:, i], color=color, lw=1.6, label=f"z[{i}] = {z[i]:.2f}")
            ax.scatter([taus[-1] * 1.4], [sm[i]], color=color, s=10, marker="<")
        ax.set_xscale("log")
        ax.set_xlabel(f"tau (k = {k:g})")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel(f"p_i (n = {n})")
    axes[0, -1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")


def flow(plt, out, k=3):
    z = INPUTS["spread"].double()
    n = z.numel()
    taus = torch.logspace(2, -3, 300, dtype=torch.float64)
    P = torch.stack([soft_topk_softmax(z, k, float(t)) for t in taus])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i in range(n):
        top = i < k  # INPUTS["spread"] is sorted in decreasing order
        ax.plot(taus, P[:, i], color="tab:blue" if top else "0.6", lw=1.8,
                label={0: f"{k} largest scores", k: f"other {n - k} scores"}.get(i))
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlim(taus[0], taus[-1])
    ax.set_xlabel("temperature τ (decreasing)")
    ax.set_ylabel(f"output probability (n = {n} scores)")
    ax.text(0, 1.03, "large τ: softmax", transform=ax.transAxes, ha="left", va="bottom")
    ax.text(1, 1.03, f"τ → 0: softmax over the {k} largest scores", transform=ax.transAxes, ha="right", va="bottom")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.06))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def train(plt, out, n=64, k=4, steps=3000, tau_start=10.0, tau_end=0.3, seed=0):
    torch.manual_seed(seed)
    d = 16
    X = torch.randn(4096, d)
    # teacher: a target distribution supported on the k largest scores of a random linear map
    zt = X @ torch.randn(d, n)
    idx = zt.topk(k, -1).indices
    target = torch.zeros_like(zt).scatter(-1, idx, torch.softmax(zt.gather(-1, idx), -1))
    layer = SoftTopKSoftmax(k, tau=tau_start, track_stats=True)
    # the LayerNorm fixes the scale of the scores, so tau keeps its meaning during training
    model = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.GELU(), torch.nn.Linear(64, n),
                                torch.nn.LayerNorm(n, elementwise_affine=False), layer)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    sched = TauScheduler(layer, tau_start, tau_end, steps)
    log = {"step": [], "tau": [], "kl": [], "out": []}
    for s in range(steps):
        b = torch.randint(0, X.shape[0], (256,))
        p, tb = model(X[b]), target[b]
        kl = (tb * (tb.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(-1).mean()
        opt.zero_grad()
        kl.backward()
        opt.step()
        if s % 25 == 0:
            log["step"].append(s)
            log["tau"].append(sched.tau / tau_start)
            log["kl"].append(kl.item())
            log["out"].append(float(layer.stats["mass_outside_mean"]))
        sched.step()
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
    ax[0].plot(log["step"], log["kl"])
    ax[0].set_ylabel("KL(target || p) (nats)")
    ax[1].plot(log["step"], log["out"], label="mass outside top-k")
    ax[1].plot(log["step"], log["tau"], "--", label="tau / tau_start")
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        a.set_yscale("log")
        a.set_xlabel(f"step (n = {n}, k = {k})")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"final KL={log['kl'][-1]:.4f}, mass outside top-k={log['out'][-1]:.2e}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m topk_softmax.demo", description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=float, nargs="+", default=[1, 2, 4], help="one panel per value")
    ap.add_argument("--tau-min", type=float, default=1e-3)
    ap.add_argument("--tau-max", type=float, default=1e2)
    ap.add_argument("--input", choices=list(INPUTS), default="spread")
    ap.add_argument("--out", help="directory for sweep_<input>.png (default: figs)")
    ap.add_argument("--flow", metavar="PATH", help="also write the README figure to PATH")
    ap.add_argument("--train", metavar="PATH", help="also write the toy training figure to PATH")
    a = ap.parse_args(argv)
    if not 0 < a.tau_min < a.tau_max:
        ap.error("need 0 < --tau-min < --tau-max")
    if min(a.k) <= 0:
        ap.error("--k values must be positive")

    try:
        import matplotlib
    except ImportError:
        raise SystemExit("topk_softmax.demo needs matplotlib: pip install 'topk-softmax[demo]'")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if a.out or not (a.flow or a.train):
        out = a.out or "figs"
        os.makedirs(out, exist_ok=True)
        sweep(plt, os.path.join(out, f"sweep_{a.input}.png"), a.input, a.k, a.tau_min, a.tau_max)
    if a.flow:
        flow(plt, a.flow)
    if a.train:
        train(plt, a.train)


if __name__ == "__main__":
    main()
