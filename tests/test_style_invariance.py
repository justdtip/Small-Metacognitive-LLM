import pytest
import torch

from train.losses import compute_losses


def test_kl_lower_for_identical_answers():
    torch.manual_seed(0)
    B, T, V = 1, 5, 7
    logits_a = torch.randn(B, T, V)
    logits_b = logits_a.clone()
    labels = torch.full((B, T), -100, dtype=torch.long)  # ignore CE
    ans_mask = torch.zeros(B, T)
    ans_mask[0, 2:] = 1.0  # last positions are answers

    out_same = compute_losses(
        logits_a, labels,
        weights={"answer_ce": 0.0, "style_inv": 1.0},
        style_pair={"logits_a": logits_a, "logits_b": logits_b, "answer_mask": ans_mask},
    )
    si_same = float(out_same["style_inv"].item())
    assert si_same <= 1e-6

    # Perturb b logits → increase KL
    logits_b2 = logits_a + 0.5 * torch.randn_like(logits_a)
    out_diff = compute_losses(
        logits_a, labels,
        weights={"answer_ce": 0.0, "style_inv": 1.0},
        style_pair={"logits_a": logits_a, "logits_b": logits_b2, "answer_mask": ans_mask},
    )
    si_diff = float(out_diff["style_inv"].item())
    assert si_diff > si_same + 1e-4
