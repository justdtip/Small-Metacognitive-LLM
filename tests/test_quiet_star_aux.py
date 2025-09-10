import pytest
torch = pytest.importorskip("torch")

from train.losses import compute_losses
from train.schedules import quiet_star_context

def test_quiet_star_aux_adds_loss():
    B, T, V = 2, 5, 7
    torch.manual_seed(0)
    logits = torch.randn(B, T, V)
    labels = torch.full((B, T), -100)
    # mark two positions with labels (simulate answer tokens)
    labels[0, 2] = 3
    labels[1, 4] = 1
    think_mask = torch.zeros(B, T)
    think_mask[0, 0] = 1
    think_mask[1, 1] = 1

    # Without aux
    out0 = compute_losses(logits, labels, weights={"answer_ce": 1.0, "aux_mix": 0.0})
    # With aux
    with quiet_star_context(think_mask, tau=2.0, sample_ratio=1.0):
        out1 = compute_losses(logits, labels, weights={"answer_ce": 1.0, "aux_mix": 0.5})

    assert "aux_loss" in out1
    assert out1["aux_loss"].item() >= 0.0
    assert out1["total"].item() > out0["total"].item()

