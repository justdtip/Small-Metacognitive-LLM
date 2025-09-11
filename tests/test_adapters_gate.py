import pytest
torch = pytest.importorskip("torch")
from side_adapters import LowRankAdapter, ResidualAdapterConfig
from train.hooks import think_mask_context
from train.losses import compute_losses

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


def test_gate_reg_loss_positive():
    B, T, H, V = 1, 4, 8, 11
    adap = LowRankAdapter(ResidualAdapterConfig(hidden_size=H, rank=4))
    with torch.no_grad():
        adap.gate.copy_(torch.tensor(1.0))
    adap.eval()
    h = torch.randn(B, T, H)

    mask = torch.zeros(B, T)
    mask[0, 0] = 1.0
    with think_mask_context(mask):
        _ = adap(h)

    logits = torch.randn(B, T, V)
    labels = torch.full((B, T), -100, dtype=torch.long)
    labels[0, 1] = 3  # one valid label to avoid degenerate CE
    out = compute_losses(logits, labels, gate_modules=[adap], weights={"answer_ce": 0.0, "gate_reg": 1.0})
    assert out["gate_reg"].item() > 0.0
