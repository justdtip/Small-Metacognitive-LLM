import pytest
torch = pytest.importorskip("torch")
from side_adapters import LowRankAdapter, ResidualAdapterConfig
from train.hooks import think_mask_context

def test_gate_applies_only_on_think_tokens():
    B, T, H = 2, 4, 8
    adap = LowRankAdapter(ResidualAdapterConfig(hidden_size=H, rank=4))
    # make sure scalar gate is on
    with torch.no_grad():
        adap.gate.copy_(torch.tensor(1.0))
    adap.eval()  # deterministic hard-concrete
    h = torch.randn(B, T, H)

    # mask: only (0,0) and (0,1) are think tokens; all tokens in batch 1 are answer (mask 0)
    mask = torch.zeros(B, T)
    mask[0, 0] = 1.0
    mask[0, 1] = 1.0

    with think_mask_context(mask):
        out = adap(h)

    delta = out - h
    # Positions with mask==1 should have non-zero deltas
    assert torch.norm(delta[0, 0]).item() > 0
    assert torch.norm(delta[0, 1]).item() > 0
    # Positions with mask==0 should have near-zero deltas (hard gate off)
    assert torch.allclose(delta[0, 2], torch.zeros_like(delta[0, 2]), atol=1e-6)
    assert torch.allclose(delta[0, 3], torch.zeros_like(delta[0, 3]), atol=1e-6)
    assert torch.allclose(delta[1], torch.zeros_like(delta[1]), atol=1e-6)

