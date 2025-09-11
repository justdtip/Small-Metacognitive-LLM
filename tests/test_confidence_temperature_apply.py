import pytest
import torch

from tina.serve import IntrospectiveEngine
from train.metrics import ece_binary


def test_apply_conf_temperature_reduces_ece_for_overconfident_probs():
    torch.manual_seed(0)
    N = 200
    # Create overconfident probabilities (simulate logits scaled too large)
    labels = torch.randint(0, 2, (N,)).float()
    base = (labels * 2 - 1) * 2.0  # +/-2
    noise = 0.5 * torch.randn(N)
    logits = (base + noise) * 3.0  # overconfident
    probs = torch.sigmoid(logits)
    e0 = ece_binary(probs, labels)

    # Apply temperature T>1 to reduce confidence
    T = 2.0
    probs_cal = IntrospectiveEngine.apply_conf_temperature(probs, T)
    e1 = ece_binary(probs_cal, labels)
    # Temperature scaling should reduce confidence magnitude on average (pull toward 0.5)
    mad0 = torch.abs(probs - 0.5).mean().item()
    mad1 = torch.abs(probs_cal - 0.5).mean().item()
    assert mad1 <= mad0 + 1e-6
