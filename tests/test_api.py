import inspect

import softk


def _params(fn):
    return inspect.signature(fn).parameters


def test_soft_topk_softmax_signature():
    params = _params(softk.soft_topk_softmax)
    assert list(params)[:5] == ["z", "k", "tau", "dim", "mode"]
    assert params["dim"].default == -1
    assert params["mode"].default == "soft"


def test_soft_topk_mask_signature():
    params = _params(softk.soft_topk_mask)
    assert list(params)[:4] == ["z", "k", "tau", "dim"]
    assert params["dim"].default == -1
