import pytest

from train.eval_loop import noise_probe


def test_no_gain_from_markers():
    recs = [
        {"body": "<think> reason briefly </think><answer> ok </answer>", "gold": "ok"},
        {"body": "<think> reason more </think><answer> ok </answer>", "gold": "ok"},
    ]
    out = noise_probe(recs, markers=["Step 1:", "===="])
    # Markers should not improve accuracy; they may increase budget proxy
    assert out["delta_accuracy"] <= 0.0
    assert out["delta_budget"] >= 0.0
