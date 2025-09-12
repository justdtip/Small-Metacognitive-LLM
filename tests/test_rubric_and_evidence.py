import pytest
import torch

from train.losses import compute_losses
from train.eval_loop import compute_eval_metrics


def test_detects_evidence_and_trains():
    body = "<think> use heat from incandescent filament </think><answer> ok </answer>"
    rec = {"body": body, "gold": "ok", "evidence_keys": ["heat", "incandescent"]}
    m = compute_eval_metrics([rec])
    # Coverage appended into think_score_mean aggregator ~ 1.0
    assert abs(float(m.get("think_score_mean") or 0.0) - 1.0) < 1e-6

    # Evidence loss near zero when all keys present
    present = torch.ones(1, 2)
    logits = torch.zeros(1, 3, 5)
    labels = torch.full((1, 3), -100, dtype=torch.long)
    out = compute_losses(logits, labels, weights={"answer_ce": 0.0, "evidence": 1.0}, evidence_present=present)
    assert float(out["evidence_loss"].item()) <= 1e-6


essage = None

def test_penalizes_missing_keys():
    body = "<think> use heat </think><answer> ok </answer>"
    rec = {"body": body, "gold": "ok", "evidence_keys": ["heat", "incandescent"]}
    m = compute_eval_metrics([rec])
    # one of two keys → 0.5 coverage
    assert abs(float(m.get("think_score_mean") or 0.0) - 0.5) < 1e-6

    # Evidence loss positive when a key is missing
    present = torch.tensor([[1.0, 0.0]])
    logits = torch.zeros(1, 3, 5)
    labels = torch.full((1, 3), -100, dtype=torch.long)
    out = compute_losses(logits, labels, weights={"answer_ce": 0.0, "evidence": 1.0}, evidence_present=present)
    assert float(out["evidence_loss"].item()) > 0.0
