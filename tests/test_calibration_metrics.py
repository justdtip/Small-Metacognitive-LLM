import pytest
torch = pytest.importorskip("torch")
from train.metrics import temperature_fit, ece_binary

def test_temperature_scaling_reduces_ece_binary():
    # Create an overconfident binary classifier: logits scaled too large
    torch.manual_seed(0)
    N = 200
    labels = torch.randint(0, 2, (N,))
    # Base logits proportional to labels with noise, then scaled up (overconfident)
    base = (labels.float() * 2 - 1) * 2.0  # +/-2
    noise = 0.5 * torch.randn(N)
    logits = (base + noise) * 3.0  # overconfident scaling
    probs = torch.sigmoid(logits)
    ece0 = ece_binary(probs, labels)

    T = temperature_fit(logits, labels, max_iter=200)
    probs_cal = torch.sigmoid(logits / max(T, 1e-6))
    ece1 = ece_binary(probs_cal, labels)

    assert ece1 <= ece0 + 1e-6  # calibration should not worsen ECE
    # In most cases it should strictly improve
    assert ece1 < ece0 or abs(ece1 - ece0) < 1e-3

