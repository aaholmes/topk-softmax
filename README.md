# topkmax

A softmax turns a vector of n scores into a probability distribution in which every entry gets
some probability. Sparse attention and expert routing instead want only the k highest-scoring
entries to receive probability. Selecting those k entries outright is discontinuous in the scores,
and the entries that are not selected receive no gradient, so a model cannot learn to promote them.

This repository is an early prototype of a PyTorch operator that interpolates between the two. It
has a temperature τ: at large τ it equals the ordinary softmax, and as τ → 0 it becomes a softmax
over only the k largest scores. It is differentiable at every τ > 0, so τ can start large and be
lowered during training. The planned package name is `softk`.

## Definition

Each score z_i gets a weight m_i between 0 and 1, called the mask, and the output p is the softmax
reweighted by that mask:

```math
m_i = \sigma\!\left(\frac{z_i - t}{\tau}\right), \qquad \sum_i m_i = k, \qquad
p_i = \frac{m_i \, e^{z_i}}{\sum_j m_j \, e^{z_j}}
```

Here σ is the logistic sigmoid, and the threshold t is the single number per row that makes the
mask sum to k. A small τ makes the sigmoid a step, so the mask is 1 on the k largest scores and 0
elsewhere. A large τ makes every m_i equal to k/n, a constant that cancels in the ratio. k does
not have to be an integer.

Two details make this cheap to compute:

- The sum of the mask decreases monotonically in t, so t is found by bisection, followed by one
  Newton step.
- The backward pass does not differentiate through the bisection. It uses the implicit function
  theorem on the constraint Σm = k, which gives a closed-form gradient costing two sums per row.

A mask of this form, a sigmoid with a threshold chosen so the entries sum to k, appears in
[The Limited Multi-Label Projection Layer](https://arxiv.org/abs/1906.08707) (Amos, Koltun and
Kolter, 2019).

## Example

```python
import torch
from softk_prototype import soft_topk_softmax

z = torch.randn(4, 10)
p = soft_topk_softmax(z, k=3, tau=0.1)   # each row sums to 1; mass concentrates on the top 3
```

## Status

This is a prototype, not yet an installable package.

- `softk_prototype.py` implements the operator and its custom backward pass in about 100 lines.
- `tests/` is a suite of 303 tests that defines what the first packaged version must do. It was
  written before the package, and it imports a `softk` module that does not exist yet, so it
  cannot run as-is.
- Run on the prototype through a temporary import alias, 220 of the 303 tests pass. These cover
  finite-difference gradient checks, agreement with an independent double-precision reference,
  the softmax limit at large τ, and the top-k limit at small τ.
- The other 83 tests cover features that are not implemented: an exact top-k mode for inference, a
  straight-through mode (exact top-k forward, soft gradient backward), half-precision inputs, a
  lower bound on τ relative to the spread of the scores, and input validation.
- One known numerical limit: in single precision, when several scores tie exactly at the k-th
  position and τ is below about 1e-3, the mask sum misses k by more than the 1e-5 target.
- `demo.py` plots the output as a function of τ and k, and runs a small training example. It
  currently fails on import because it expects the prototype under a different module name.

There is no evidence yet that the operator improves a real model; it has only been checked for
numerical correctness.

## Running the prototype's checks

```sh
uv sync
uv run python softk_prototype.py
```
