import pytest

from train.runner import adapter_gate_step_smoke


def test_gate_coverage_fraction():
    out = adapter_gate_step_smoke()
    cov = out.get("coverage")
    assert cov is not None
    assert 0.0 <= cov <= 1.0
    # In this synthetic setup, coverage should be less than 1.0 since some non-think positions exist
    assert cov < 1.0


def test_gate_coverage_value():
    out = adapter_gate_step_smoke()
    cov = out.get("coverage")
    assert cov is not None
    assert 0.0 < cov <= 1.0
