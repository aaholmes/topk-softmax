import subprocess
import sys

import pytest

pytestmark = pytest.mark.slow


def _demo(*args):
    return subprocess.run([sys.executable, "-m", "topk_softmax.demo", *args], capture_output=True, text=True)


@pytest.mark.parametrize("name", ["spread", "tie", "spike", "random"])
def test_demo_writes_a_figure_for_each_input(name, tmp_path):
    out = tmp_path / "figs"  # does not exist yet; the command creates it
    result = _demo("--k", "1", "2", "--input", name, "--out", str(out))
    assert result.returncode == 0, result.stderr
    figures = list(out.glob("*.png"))
    assert figures
    assert all(f.stat().st_size > 0 for f in figures)


def test_demo_accepts_a_tau_range(tmp_path):
    result = _demo("--k", "2", "--tau-min", "1e-2", "--tau-max", "1e1", "--input", "spread", "--out", str(tmp_path))
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.glob("*.png"))


def test_demo_rejects_an_unknown_input(tmp_path):
    result = _demo("--k", "1", "--input", "no-such-input", "--out", str(tmp_path))
    assert result.returncode != 0
    assert not list(tmp_path.glob("*.png"))
