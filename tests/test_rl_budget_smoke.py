import pytest
torch = pytest.importorskip("torch")

from train.rl_loop import dry_run_budget_rl


def test_dry_run_budget_rl_decreases_budget():
    out = dry_run_budget_rl(steps=40, seed=1)
    assert out["mu_after"] <= out["mu_before"] + 1e-6
