import io
import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

import softk
from _reference import randn

KINDS = ["exp", "cosine", "linear"]


def _layer(**kwargs):
    return softk.SoftTopKSoftmax(3, **kwargs)


def _run(sched, steps, mass_outside=None):
    """tau after each of `steps` calls to step()."""
    out = []
    for _ in range(steps):
        sched.step() if mass_outside is None else sched.step(mass_outside=mass_outside)
        out.append(sched.tau)
    return out


def _expected(kind, s, tau0, tau1, total, hold):
    """tau after s steps: tau0 through the hold, then the named curve, then tau1."""
    if s <= hold:
        return tau0
    if s >= total:
        return tau1
    x = (s - hold) / (total - hold)
    if kind == "exp":
        return tau0 * (tau1 / tau0) ** x
    if kind == "cosine":  # a half cosine in log tau
        return math.exp(math.log(tau0) + (math.log(tau1) - math.log(tau0)) * (1 - math.cos(math.pi * x)) / 2)
    return tau0 + (tau1 - tau0) * x  # linear in tau


# ---- the schedule itself ----

def test_tau_starts_at_tau_start_and_is_written_to_the_module():
    layer = _layer(tau=3.0)
    sched = softk.TauScheduler(layer, 10.0, 0.5, 100)
    assert sched.tau == 10.0
    assert float(layer.tau) == 10.0


# hold_frac * total_steps is an exact integer in every case, so the hold length is unambiguous
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("total,hold_frac", [(100, 0.25), (40, 0.5), (100, 0.0), (64, 0.125)])
def test_tau_follows_the_curve_at_every_step(kind, total, hold_frac):
    hold = int(hold_frac * total)
    sched = softk.TauScheduler(_layer(), 10.0, 0.3, total, kind=kind, hold_frac=hold_frac)
    taus = _run(sched, total + 20)
    for s, tau in enumerate(taus, start=1):
        assert tau == pytest.approx(_expected(kind, s, 10.0, 0.3, total, hold), rel=1e-9), s


def test_kind_defaults_to_exp_and_hold_to_five_percent():
    sched = softk.TauScheduler(_layer(), 10.0, 0.3, 200)
    taus = _run(sched, 200)
    for s, tau in enumerate(taus, start=1):
        assert tau == pytest.approx(_expected("exp", s, 10.0, 0.3, 200, 10), rel=1e-9), s


@pytest.mark.parametrize("kind", KINDS)
def test_tau_stays_at_tau_end_after_the_last_step(kind):
    sched = softk.TauScheduler(_layer(), 10.0, 0.3, 50, kind=kind)
    taus = _run(sched, 120)
    assert all(tau == pytest.approx(0.3, rel=1e-9) for tau in taus[49:])


def test_equal_start_and_end_gives_a_constant():
    sched = softk.TauScheduler(_layer(), 2.0, 2.0, 30)
    assert all(tau == pytest.approx(2.0, rel=1e-12) for tau in _run(sched, 40))


@given(
    kind=st.sampled_from(KINDS),
    tau_end=st.floats(1e-3, 1.0),
    ratio=st.floats(1.0, 1e3),
    total=st.integers(1, 300),
    hold_frac=st.floats(0.0, 0.9),
)
def test_every_kind_is_non_increasing_between_its_endpoints(kind, tau_end, ratio, total, hold_frac):
    tau_start = tau_end * ratio
    sched = softk.TauScheduler(_layer(), tau_start, tau_end, total, kind=kind, hold_frac=hold_frac)
    taus = [sched.tau] + _run(sched, total + 5)
    assert taus[0] == pytest.approx(tau_start, rel=1e-12)
    assert taus[total] == pytest.approx(tau_end, rel=1e-9)
    assert taus[-1] == pytest.approx(tau_end, rel=1e-9)
    for a, b in zip(taus, taus[1:]):
        assert b <= a * (1 + 1e-12)
    assert all(tau_end * (1 - 1e-9) <= tau <= tau_start * (1 + 1e-9) for tau in taus)


# ---- writing into modules ----

def test_every_registered_module_receives_tau():
    layers = [_layer(tau=1.0), _layer(tau=2.0), _layer(tau=3.0)]
    sched = softk.TauScheduler(layers, 10.0, 0.3, 100)
    for _ in range(60):
        sched.step()
        for layer in layers:
            assert float(layer.tau) == pytest.approx(sched.tau, rel=1e-6)


def test_a_single_module_is_accepted():
    layer = _layer()
    sched = softk.TauScheduler(layer, 10.0, 0.3, 100)
    _run(sched, 100)
    assert float(layer.tau) == pytest.approx(0.3, rel=1e-6)


def test_the_module_output_follows_the_schedule():
    z = randn(5, 12)
    layer = _layer()
    sched = softk.TauScheduler(layer, 10.0, 0.3, 100)
    _run(sched, 100)
    expected = softk.soft_topk_softmax(z, 3, float(layer.tau))
    assert torch.allclose(layer(z), expected, atol=1e-12)


def test_the_buffer_keeps_its_dtype_and_identity():
    layer = _layer()
    buffer = layer.tau
    sched = softk.TauScheduler(layer, 10.0, 0.3, 100)
    _run(sched, 10)
    assert layer.tau is buffer
    assert layer.tau.ndim == 0


# ---- early stop ----

def test_early_stop_is_off_by_default():
    a = softk.TauScheduler(_layer(), 10.0, 0.3, 200)
    b = softk.TauScheduler(_layer(), 10.0, 0.3, 200)
    assert _run(a, 200, mass_outside=0.0) == _run(b, 200)


def test_a_high_metric_never_stops_the_schedule():
    a = softk.TauScheduler(_layer(), 10.0, 0.3, 200, early_stop_eps=1e-3)
    b = softk.TauScheduler(_layer(), 10.0, 0.3, 200)
    assert _run(a, 200, mass_outside=0.5) == _run(b, 200)


# The metric is constant in each phase, so these hold for any form of running mean.
def test_a_low_metric_freezes_tau_before_the_end():
    sched = softk.TauScheduler(_layer(), 10.0, 0.3, 200, early_stop_eps=1e-3)
    taus = _run(sched, 200, mass_outside=1e-5)
    assert taus[-1] > 0.3 * 1.5
    assert len(set(taus[-50:])) == 1


def test_a_frozen_tau_never_moves_again():
    layer = _layer()
    sched = softk.TauScheduler(layer, 10.0, 0.3, 1000, early_stop_eps=1e-3)
    frozen = _run(sched, 200, mass_outside=1e-5)[-1]
    later = _run(sched, 900, mass_outside=0.5)
    assert set(later) == {frozen}
    assert float(layer.tau) == pytest.approx(frozen, rel=1e-6)


def test_early_stop_reads_the_module_stats_by_default():
    # widely separated scores and a small tau_end leave almost no mass outside the top 3
    z = 50.0 * randn(8, 12)
    layer = _layer(track_stats=True)
    sched = softk.TauScheduler(layer, 1.0, 1e-3, 200, hold_frac=0.0, early_stop_eps=1e-3)
    taus = []
    for _ in range(200):
        layer(z)
        sched.step()
        taus.append(sched.tau)
    assert taus[-1] > 1e-3 * 1.5
    assert len(set(taus[-50:])) == 1


# ---- straight-through tail ----

def test_st_tail_is_off_by_default_and_leaves_mode_alone():
    layer = _layer(mode="hard")
    sched = softk.TauScheduler(layer, 10.0, 0.3, 100)
    for _ in range(120):
        sched.step()
        assert layer.mode == "hard"


def test_st_tail_switches_mode_for_the_final_steps():
    layers = [_layer(), _layer()]
    sched = softk.TauScheduler(layers, 10.0, 0.3, 100, st_frac=0.1)
    modes = []
    for _ in range(130):
        sched.step()
        assert layers[0].mode == layers[1].mode
        modes.append(layers[0].mode)
    assert set(modes[:85]) == {"soft"}  # steps 1-85
    assert set(modes[94:]) == {"st"}    # steps 95 onward, including past the end
    assert sum(a != b for a, b in zip(modes, modes[1:])) == 1


def test_st_tail_does_not_change_tau():
    a = softk.TauScheduler(_layer(), 10.0, 0.3, 100, st_frac=0.1)
    b = softk.TauScheduler(_layer(), 10.0, 0.3, 100)
    assert _run(a, 120) == _run(b, 120)


# ---- saving and resuming ----

def _roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer)


RESUME_CASES = {
    "mid-hold": (dict(hold_frac=0.25), 10, None),
    "mid-decay": (dict(hold_frac=0.25), 37, None),
    "cosine": (dict(kind="cosine"), 37, None),
    "linear": (dict(kind="linear"), 37, None),
    "past-the-end": (dict(), 150, None),
    "frozen": (dict(early_stop_eps=1e-3), 60, 1e-5),
    "not-yet-frozen": (dict(early_stop_eps=1e-3), 60, 0.5),
    "st-tail": (dict(st_frac=0.2), 90, None),
    "before-st-tail": (dict(st_frac=0.2), 70, None),
}


@pytest.mark.parametrize("case", RESUME_CASES)
def test_resuming_from_state_dict_continues_the_identical_sequence(case):
    kwargs, steps_before, metric = RESUME_CASES[case]

    layer = _layer()
    sched = softk.TauScheduler(layer, 10.0, 0.3, 100, **kwargs)
    _run(sched, steps_before, mass_outside=metric)
    saved = _roundtrip(sched.state_dict())
    saved_layer = _roundtrip(layer.state_dict())

    layer2 = _layer()
    layer2.load_state_dict(saved_layer)
    sched2 = softk.TauScheduler(layer2, 10.0, 0.3, 100, **kwargs)
    sched2.load_state_dict(saved)
    assert sched2.tau == sched.tau
    assert float(layer2.tau) == float(layer.tau)
    assert layer2.mode == layer.mode

    for _ in range(80):
        for s in (sched, sched2):
            s.step() if metric is None else s.step(mass_outside=metric)
        assert sched2.tau == sched.tau
        assert float(layer2.tau) == float(layer.tau)
        assert layer2.mode == layer.mode


def test_state_dict_does_not_hold_the_modules():
    sched = softk.TauScheduler(_layer(), 10.0, 0.3, 100)
    _run(sched, 5)
    assert not any(isinstance(v, torch.nn.Module) for v in sched.state_dict().values())


# ---- argument checks ----

@pytest.mark.parametrize("kwargs", [
    dict(tau_start=0.0), dict(tau_start=-1.0), dict(tau_end=0.0), dict(tau_end=-1.0),
    dict(tau_start=0.1, tau_end=0.3),
    dict(total_steps=0), dict(total_steps=-5),
    dict(kind="linear_log"), dict(kind="step"),
    dict(hold_frac=-0.1), dict(hold_frac=1.5),
    dict(st_frac=-0.1), dict(st_frac=1.5),
    dict(early_stop_eps=0.0), dict(early_stop_eps=-1e-3),
])
def test_invalid_arguments_raise(kwargs):
    args = dict(tau_start=10.0, tau_end=0.3, total_steps=100)
    args.update(kwargs)
    with pytest.raises(ValueError):
        softk.TauScheduler(_layer(), **args)
