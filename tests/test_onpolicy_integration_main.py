import json
import types
import pytest
import torch

from train import runner as rn


def test_rl_called_and_think_len_present(monkeypatch):
    calls = {"n": 0}
    _orig_rl = rn.rl_phase_step
    def _spy_rl(*a, **k):
        calls["n"] += 1
        return _orig_rl(*a, **k)
    monkeypatch.setattr(rn, "rl_phase_step", _spy_rl)

    # Stub decode_with_budget to avoid heavy generate and return a fixed think length
    def _stub_decode(tok, model, input_ids, **kw):
        return {"text": "<think>x</think><answer>ok</answer>", "think_tokens_used": 4}
    monkeypatch.setattr(rn, "decode_with_budget", _stub_decode)

    out = rn.main_train_loop(steps=3, sample_every=1, budget_cap=8)
    batch = out.get("last_batch")
    assert batch is not None and isinstance(batch.get("think_tokens_used"), torch.Tensor)
    assert int(batch["think_tokens_used"][0].item()) > 0
    assert calls["n"] >= 1

