"""Four ways to train a network whose output should be supported on its k largest scores.

    python benchmarks/toy.py --out benchmarks/results/toy.json     # train (about 2 min, CPU)
    python benchmarks/toy.py --report benchmarks/results/toy.json  # table from a saved file

The task is the one in tests/test_toy_learning.py: a fixed random linear teacher gives each input a
target distribution supported on its K largest of N scores. Every arm uses identical data, initial
weights, batch order, optimizer and step count for a given seed, and none is tuned separately. All
arms are scored the way the model would be used: with exact top-k at test time, by total variation
distance to the target on held-out inputs.

    annealed   soft top-k with tau lowered from 10 to 0.3 by TauScheduler
    annealed_flat  the annealed arm with flat_top=True (the top k share one mask value)
    fixed_tau  soft top-k with tau = 0.3 throughout
    softmax    a plain softmax during training
    hard       exact top-k during training
"""
import argparse
import json
import random
import statistics
from pathlib import Path

import torch

import topk_softmax

N, K, D, STEPS, SEEDS = 64, 4, 16, 2000, 10
TAU_START, TAU_END = 10.0, 0.3
ARMS = ["annealed", "annealed_flat", "fixed_tau", "softmax", "hard"]


def held_out_distance(seed, arm):
    g = torch.Generator().manual_seed(seed)
    teacher = torch.randn(D, N, generator=g)

    def data(rows):
        x = torch.randn(rows, D, generator=g)
        scores = x @ teacher
        idx = scores.topk(K, -1).indices
        return x, torch.zeros_like(scores).scatter(-1, idx, torch.softmax(scores.gather(-1, idx), -1))

    x, target = data(4096)
    x_held, target_held = data(1024)

    torch.manual_seed(seed)
    annealed = arm in ("annealed", "annealed_flat")
    layer = topk_softmax.SoftTopKSoftmax(K, tau=TAU_START if annealed else TAU_END,
                                         mode="hard" if arm == "hard" else "soft", flat_top=arm == "annealed_flat")
    body = torch.nn.Sequential(
        torch.nn.Linear(D, 64), torch.nn.GELU(), torch.nn.Linear(64, N),
        torch.nn.LayerNorm(N, elementwise_affine=False),
    )
    head = torch.nn.Softmax(-1) if arm == "softmax" else layer
    opt = torch.optim.Adam(body.parameters(), 3e-3)
    sched = topk_softmax.TauScheduler(layer, TAU_START, TAU_END, STEPS) if annealed else None

    for _ in range(STEPS):
        b = torch.randint(0, x.shape[0], (256,), generator=g)
        p, t = head(body(x[b])), target[b]
        loss = (t * (t.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

    with torch.no_grad():
        p = topk_softmax.soft_topk_softmax(body(x_held), K, 1.0, mode="hard")
        return 0.5 * (p - target_held).abs().sum(-1).mean().item()


def _bootstrap_interval(diffs, draws=10000):
    rng = random.Random(0)
    medians = sorted(statistics.median(rng.choices(diffs, k=len(diffs))) for _ in range(draws))
    return medians[int(0.025 * draws)], medians[int(0.975 * draws)]


def report(path):
    data = json.loads(Path(path).read_text())
    d = data["distance"]
    n = len(d["annealed"])
    print(f"Held-out total variation distance with exact top-k at test time, n = {n} paired seeds.\n")
    print("| arm | median | min | max | annealed minus this arm, median [95% bootstrap interval] | seeds where annealed is better |")
    print("| --- | --- | --- | --- | --- | --- |")
    for arm in ARMS:
        v = d[arm]
        row = f"| {arm} | {statistics.median(v):.3f} | {min(v):.3f} | {max(v):.3f} |"
        if arm == "annealed":
            print(row + " | |")
            continue
        diffs = [a - b for a, b in zip(d["annealed"], v)]
        lo, hi = _bootstrap_interval(diffs)
        print(row + f" {statistics.median(diffs):+.3f} [{lo:+.3f}, {hi:+.3f}] | {sum(x < 0 for x in diffs)} of {n} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--report", metavar="FILE")
    a = ap.parse_args()
    if a.out:
        distance = {arm: [held_out_distance(s, arm) for s in range(SEEDS)] for arm in ARMS}
        settings = dict(n=N, k=K, d=D, steps=STEPS, tau_start=TAU_START, tau_end=TAU_END, torch=torch.__version__)
        Path(a.out).write_text(json.dumps({"settings": settings, "distance": distance}, indent=1))
        report(a.out)
    if a.report:
        report(a.report)


if __name__ == "__main__":
    main()
