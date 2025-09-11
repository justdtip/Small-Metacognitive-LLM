import json
import pytest

from tina.serve import _extract_answer, SlackStop
from train.eval_loop import fit_confidence_temperature_and_save
from train.metrics import ece_binary


def test_hidden_leakage_zero_extraction():
    # Hidden mode extraction removes tags from answer
    body = "<think>some chain</think><answer>final</answer>"
    out = _extract_answer(body, include_think=False)
    assert out == "final"
    assert "<think>" not in out and "</think>" not in out
    assert "<answer>" not in out and "</answer>" not in out


def test_budget_soft_cap_thresholds():
    torch = pytest.importorskip("torch")
    class DummyTok:
        def encode(self, s, add_special_tokens=False):
            # only used by StopOnTags elsewhere; SlackStop does not require tokenizer
            return [1, 2]

    base_len = 7
    slack = 0.2
    for B in (8, 16, 32):
        soft = SlackStop(base_len=base_len, budget=B, slack_ratio=slack)
        limit = int(B * (1.0 + slack))
        # At limit: not stopped yet
        ids_at = torch.tensor([[0] * (base_len + limit)], dtype=torch.long)
        assert soft(ids_at, None) is False
        # Beyond limit: stop
        ids_over = torch.tensor([[0] * (base_len + limit + 1)], dtype=torch.long)
        assert soft(ids_over, None) is True


def test_confidence_ece_below_after_calibration(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    N = 300
    # Overconfident synthetic logits
    labels = torch.randint(0, 2, (N,))
    base = (labels.float() * 2 - 1) * 1.5
    noise = 0.7 * torch.randn(N)
    logits = (base + noise) * 2.5

    # ECE before
    e0 = ece_binary(torch.sigmoid(logits), labels)
    blob_path = tmp_path / "cal.json"
    blob = fit_confidence_temperature_and_save(logits, labels, str(blob_path))
    assert blob_path.exists()
    T = float(json.loads(blob_path.read_text())['conf_temp'])
    e1 = ece_binary(torch.sigmoid(logits / max(T, 1e-6)), labels)

    # Allow a small tolerance on synthetic draws
    assert e1 <= e0 + 1e-2
    # Acceptance threshold (loose for synthetic data)
    assert e1 <= 0.5
