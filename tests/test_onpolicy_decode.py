import json
import types
import pytest
torch = pytest.importorskip("torch")

from train import runner as rn


def test_onpolicy_integration_records_think_len(monkeypatch):
    # Spy on rl_phase_step calls
    calls = {"n": 0}
    _orig_rl = rn.rl_phase_step
    def _spy_rl(*a, **k):
        calls["n"] += 1
        return _orig_rl(*a, **k)
    monkeypatch.setattr(rn, "rl_phase_step", _spy_rl)

    # Stub decode_with_budget to avoid heavy generation and return a fixed think length
    def _stub_decode(tok, model, input_ids, **kw):
        return {"text": "<think>x</think><answer>ok</answer>", "think_tokens_used": 3}
    monkeypatch.setattr(rn, "decode_with_budget", _stub_decode)

    out = rn.onpolicy_sft_and_rl_smoke(steps=2, sample_every=1, budget_cap=8)
    batch = out.get("last_batch")
    assert batch is not None and isinstance(batch.get("think_tokens_used"), torch.Tensor)
    assert int(batch["think_tokens_used"][0].item()) > 0
    assert calls["n"] >= 1
    # RL stats contain expected keys
    assert out.get("rl_stats") and all(k in out["rl_stats"][0] for k in ("mu_mean", "mu_after", "reward_mean"))

