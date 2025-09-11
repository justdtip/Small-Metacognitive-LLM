import pytest
torch = pytest.importorskip("torch")

from train.eval_loop import fit_confidence_temperature_and_save
from train.metrics import ece_binary


def test_ece_drops_after_temp(tmp_path):
    torch.manual_seed(42)
    N = 500
    # Construct overconfident logits aligned with labels to ensure improvement via temperature scaling
    labels = torch.randint(0, 2, (N,))
    base = (labels.float() * 2 - 1) * 2.0
    noise = 0.3 * torch.randn(N)
    logits = (base + noise) * 5.0  # deliberately overconfident

    # ECE before
    e0 = ece_binary(torch.sigmoid(logits), labels)
    out = fit_confidence_temperature_and_save(logits, labels, str(tmp_path / "cal.json"))
    T = float(out["conf_temp"]) if isinstance(out, dict) else None
    e1 = float(out["ece_after"]) if isinstance(out, dict) else ece_binary(torch.sigmoid(logits / max(T or 1.0, 1e-6)), labels)

    # Require a strict improvement (allow tiny tolerance for numerical noise)
    assert e1 < e0 - 1e-6
