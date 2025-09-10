import pytest
torch = pytest.importorskip("torch")
from side_adapters import LowRankAdapter, ResidualAdapterConfig
from train.hooks import think_mask_context

def test_mask_context_does_not_leak_between_calls():
    B, T, H = 1, 3, 8
    adap = LowRankAdapter(ResidualAdapterConfig(hidden_size=H, rank=4))
    with torch.no_grad():
        adap.gate.copy_(torch.tensor(1.0))
    adap.eval()
    h = torch.randn(B, T, H)

    # First call: mask only position 0 active
    m1 = torch.zeros(B, T); m1[0, 0] = 1
    with think_mask_context(m1):
        out1 = adap(h)
    d1 = out1 - h
    nz1 = [torch.norm(d1[0, i]).item() > 0 for i in range(T)]

    # Second call: different mask, position 1 active
    m2 = torch.zeros(B, T); m2[0, 1] = 1
    with think_mask_context(m2):
        out2 = adap(h)
    d2 = out2 - h
    nz2 = [torch.norm(d2[0, i]).item() > 0 for i in range(T)]

    # Assert positions changed exactly according to each mask (no carry-over)
    assert nz1[0] and not nz1[1] and not nz1[2]
    assert not nz2[0] and nz2[1] and not nz2[2]

