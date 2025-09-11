import pytest

from train.runner import adapter_gate_step_smoke


def test_coverage_tracks_think_only():
    out = adapter_gate_step_smoke()
    assert "coverage" in out
    cov = out.get("coverage")
    assert cov is not None
    assert cov > 0.0
