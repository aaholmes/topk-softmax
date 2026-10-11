"""Quantities to log while annealing tau."""
import math

import torch


def mass_outside_topk(p, k, dim=-1):
    """Probability that p puts outside its k largest entries along `dim`. Zero for a hard output."""
    return (1 - p.topk(math.ceil(k), dim).values.sum(dim)).clamp_min(0)


def effective_support(p, dim=-1):
    """exp(entropy of p) along `dim`: n for a uniform row, 1 for a one-hot row."""
    return torch.special.entr(p).sum(dim).exp()


def topk_churn(z_old, z_new, k, dim=-1):
    """Fraction of the top-k indices of z_old along `dim` that are not in the top k of z_new.

    Call it on the scores of a fixed probe batch at two points in training. A value near zero
    while the loss is still falling suggests the selection stopped changing too early.
    """
    k = math.ceil(k)
    old = torch.zeros_like(z_old, dtype=torch.bool).scatter(dim, z_old.topk(k, dim).indices, True)
    new = torch.zeros_like(z_new, dtype=torch.bool).scatter(dim, z_new.topk(k, dim).indices, True)
    return 1 - (old & new).sum(dim).to(z_old.dtype) / k
