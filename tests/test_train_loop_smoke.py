import pytest
torch = pytest.importorskip("torch")
from train.runner import sft_one_step_smoke

def test_one_step_sft():
    out = sft_one_step_smoke()
    assert isinstance(out["loss"], float)
    # Hidden extraction should strip tags
    assert "<think>" not in out["extracted"] and "</think>" not in out["extracted"]
    assert "<answer>" not in out["extracted"] and "</answer>" not in out["extracted"]

