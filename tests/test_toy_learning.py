"""Annealed soft top-k compared with hard top-k from step 0, on a synthetic teacher task.

A fixed random linear teacher gives each input a target distribution supported on its K largest
scores. A small network is trained to match it twice per seed, with identical data, initial
weights and batch order: once with tau annealed from 10 to 0.3, once with hard top-k throughout.
Both are scored in hard mode on held-out inputs by total variation distance to the target.

This shows only that annealing beats hard top-k on this one task. There is no fixed-tau arm, so
it does not separate annealing from starting soft.
"""
import pytest
import torch

import softk

pytestmark = pytest.mark.slow

N, K, D, STEPS, SEEDS = 64, 4, 16, 2000, 10


def _held_out_distance(seed, anneal):
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
    layer = softk.SoftTopKSoftmax(K, tau=10.0, mode="soft" if anneal else "hard")
    model = torch.nn.Sequential(
        torch.nn.Linear(D, 64), torch.nn.GELU(), torch.nn.Linear(64, N),
        torch.nn.LayerNorm(N, elementwise_affine=False), layer,
    )
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    sched = softk.TauScheduler(layer, 10.0, 0.3, STEPS) if anneal else None

    for _ in range(STEPS):
        b = torch.randint(0, x.shape[0], (256,), generator=g)
        p, t = model(x[b]), target[b]
        loss = (t * (t.clamp_min(1e-12).log() - p.clamp_min(1e-12).log())).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

    layer.mode = "hard"
    with torch.no_grad():
        return 0.5 * (model(x_held) - target_held).abs().sum(-1).mean().item()


@pytest.fixture(scope="module")
def distances():
    return [(_held_out_distance(s, anneal=True), _held_out_distance(s, anneal=False)) for s in range(SEEDS)]


def test_annealing_beats_hard_topk_on_at_least_nine_of_ten_seeds(distances):
    # under no difference, 9 or more wins out of 10 has probability 0.011
    wins = sum(annealed < hard for annealed, hard in distances)
    assert wins >= 9, distances


def test_annealed_model_recovers_most_of_the_target(distances):
    # a model that put its mass on entirely wrong entries would score 1.0
    annealed = sorted(a for a, _ in distances)
    assert annealed[len(annealed) // 2] < 0.25, annealed
