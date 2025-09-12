import pytest
import torch

from train.losses import compute_losses


def test_answer_unaffected_by_wrong_think():
    torch.manual_seed(0)
    B, T, V, H = 1, 6, 11, 8
    # Identical answer CE setup: labels ignored
    logits = torch.randn(B, T, V)
    labels = torch.full((B, T), -100, dtype=torch.long)
    # Think hidden and mask
    think_hidden = torch.randn(B, T, H)
    think_mask = torch.zeros(B, T); think_mask[0, :3] = 1.0

    # Non-adversarial sample (label=0) → penalty should be zero
    out_clean = compute_losses(
        logits, labels,
        weights={"answer_ce": 0.0, "wrongthink": 0.1},
        think_hidden=think_hidden, think_mask=think_mask, wrongthink_labels=torch.tensor([0.0]),
    )
    assert float(out_clean["wrongthink_penalty"].item()) == 0.0
    # Adversarial sample (label=1) → penalty positive
    out_adv = compute_losses(
        logits, labels,
        weights={"answer_ce": 0.0, "wrongthink": 0.1},
        think_hidden=think_hidden, think_mask=think_mask, wrongthink_labels=torch.tensor([1.0]),
    )
    assert float(out_adv["wrongthink_penalty"].item()) > 0.0
