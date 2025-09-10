import pytest
torch = pytest.importorskip("torch")
from metacog_heads import MetacogHeads

def test_budget_bounds():
    hs = torch.randn(2, 8, 16)
    heads = MetacogHeads(hidden_size=16, taps=[1,3,5], proj_dim=32)
    out = heads(hs, B_max=128, temperature=0.7)
    assert out["budget"].min().item() >= 0
    assert out["budget"].max().item() <= 128
