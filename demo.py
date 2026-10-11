"""softk demo: how p = soft_topk_softmax(z, k, tau) moves from softmax to top-k softmax.

Figure 1: rows = input vectors, cols = k; each line is one output entry p_i vs tau (log axis).
Figure 2: toy training run with tau annealing: KL to a k-sparse target and mass outside top-k vs step.
"""
import argparse
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from softk_proto import soft_topk_softmax

INPUTS = {
    "spread":    torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5]),
    "near-tie":  torch.tensor([1.0, 0.95, 0.9, 0.85, -1.0, -1.1, -1.2, -1.3]),
    "one spike": torch.tensor([4.0, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3]),
}


def sweep(out):
    ks = [1, 2, 4]
    taus = torch.logspace(-3, 2, 200, dtype=torch.float64)
    fig, axes = plt.subplots(len(INPUTS), len(ks), figsize=(7.5, 6), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    for r, (name, z) in enumerate(INPUTS.items()):
        z = z.double()
        P = torch.stack([soft_topk_softmax(z, k, float(t)) for k in ks for t in taus]).view(len(ks), len(taus), -1)
        sm = torch.softmax(z, -1)
        for c, k in enumerate(ks):
            ax = axes[r, c]
            for i in range(z.numel()):
                ax.plot(taus, P[c, :, i], color=cmap(i / (z.numel() - 1)), lw=1.6, label=f"z[{i}]={z[i]:.2f}")
                ax.scatter([taus[-1] * 1.4], [sm[i]], color=cmap(i / (z.numel() - 1)), s=10, marker="<")
            ax.set_xscale("log")
            if r == 0:
                ax.set_title(f"k = {k}")
            if c == 0:
                ax.set_ylabel(f"{name}\np_i (n = {z.numel()})")
            if r == len(INPUTS) - 1:
                ax.set_xlabel("tau")
            ax.grid(alpha=0.3)
    axes[0, -1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")


def train(out, n=64, k=4, steps=3000, tau0=10.0, tau_end=0.01, seed=0):
    torch.manual_seed(seed)
    d = 16
    X = torch.randn(4096, d)
    # teacher: k-sparse target distribution from a random linear scorer
    W_t = torch.randn(d, n)
    zt = X @ W_t
    idx = zt.topk(k, -1).indices
    target = torch.zeros_like(zt).scatter(-1, idx, torch.softmax(zt.gather(-1, idx), -1))
    model = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.GELU(), torch.nn.Linear(64, n),
                                torch.nn.LayerNorm(n, elementwise_affine=False))
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    hold = int(0.05 * steps)
    log = {"step": [], "tau": [], "kl": [], "out": []}
    for s in range(steps):
        tau = tau0 if s < hold else tau0 * (tau_end / tau0) ** ((s - hold) / (steps - hold))
        b = torch.randint(0, X.shape[0], (256,))
        p = soft_topk_softmax(model(X[b]), k, tau)
        tb = target[b]
        kl = (tb * (tb.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(-1).mean()
        opt.zero_grad(); kl.backward(); opt.step()
        if s % 25 == 0:
            with torch.no_grad():
                outside = 1 - p.topk(k, -1).values.sum(-1).mean()
            log["step"].append(s); log["tau"].append(tau); log["kl"].append(kl.item()); log["out"].append(outside.item())
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
    ax[0].plot(log["step"], log["kl"]); ax[0].set_yscale("log"); ax[0].set_ylabel("KL(target || p) (nats)"); ax[0].set_xlabel(f"step (n = {n}, k = {k})")
    ax[1].plot(log["step"], log["out"], label="mass outside top-k")
    ax[1].plot(log["step"], [t / tau0 for t in log["tau"]], "--", label="tau / tau0")
    ax[1].set_yscale("log"); ax[1].set_xlabel(f"step (n = {n}, k = {k})"); ax[1].legend(fontsize=8, frameon=False)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"final KL={log['kl'][-1]:.4f}, mass outside top-k={log['out'][-1]:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="sweep.png")
    ap.add_argument("--train", default="train.png")
    a = ap.parse_args()
    sweep(a.sweep)
    train(a.train)
