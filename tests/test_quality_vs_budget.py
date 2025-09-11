import json
import pytest

from train.eval_loop import quality_vs_budget_curve


def test_curve_shows_monotonic_budget(tmp_path, monkeypatch):
    # Stub decode_with_budget to return think_tokens_used proportional to budget
    import train.eval_loop as ev
    calls = {"seen": []}
    def _stub_decode(tokenizer, model, input_ids, *, think_budget, **kw):
        calls["seen"].append(int(think_budget))
        # correctness improves when budget >= threshold (e.g., 32)
        text = "ok" if int(think_budget) >= 32 else "no"
        return {"text": text, "think_tokens_used": int(think_budget)}
    monkeypatch.setattr(ev, "decode_with_budget", _stub_decode)

    # Dummy tokenizer/model; not used by the stub
    class Tok:
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            class R: pass
            import torch
            r = R(); r.input_ids = torch.tensor([[1,2,3]], dtype=torch.long)
            return r

    class Mod:
        pass

    prompts = [
        {"prompt": "Q1", "gold": "ok"},
        {"prompt": "Q2", "gold": "ok"},
    ]
    budgets = [16, 32, 64, 128]
    out_csv = tmp_path / "curve.csv"
    out = quality_vs_budget_curve(Mod(), budgets, tokenizer=Tok(), prompts=prompts, out_csv=str(out_csv))
    rows = out.get("curve")
    # One row per budget
    assert len(rows) == len(budgets)
    # mean_think_tokens increases monotonically with budget
    m = [r["mean_think_tokens"] for r in rows]
    assert all(m[i] < m[i+1] for i in range(len(m)-1))

