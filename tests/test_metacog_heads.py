import pytest
torch = pytest.importorskip("torch")
from metacog_heads import MetacogHeads, LegacyMetacogHeads

def test_shapes_and_bounds():
    B, T, H = 2, 7, 16
    taps = (1, 3)
    hs_list = [torch.randn(B, T, H) for _ in range(max(taps) + 1)]
    heads = MetacogHeads(hidden_size=H, taps=taps, plan_k=3)
    plan, budget, conf = heads(hs_list)
    assert plan.shape == (B, 3)
    assert budget.shape == (B, 1)
    assert conf.shape == (B, 1)
    assert float(budget.min()) >= 0.0 and float(budget.max()) <= 4096.0
    assert float(conf.min()) >= -20.0 and float(conf.max()) <= 20.0

