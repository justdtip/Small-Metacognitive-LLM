import json
from pathlib import Path

import pytest

import train.eval_loop as el


def test_monotone_think_tokens(tmp_path, monkeypatch):
    # Monkeypatch decode_with_budget so that think_tokens_used equals the budget (monotone in budget)
    def _stub_decode(tok, model, input_ids, *, think_budget, **kw):
        return {"text": "<think> x </think><answer> ok </answer>", "think_tokens_used": int(think_budget)}

    monkeypatch.setattr(el, "decode_with_budget", _stub_decode)

    # Tokenizer stub providing minimal interface
    class Tok:
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            class R: pass
            r = R(); r.input_ids = [1, 2, 3]
            return r

    tok = Tok()
    model = object()  # unused by stub
    budgets = [2, 4, 8, 16]
    out_csv = tmp_path / "curve.csv"
    prompts = [
        {"prompt": "Hello", "gold": "ok"},
        {"prompt": "World", "gold": "ok"},
    ]

    res = el.quality_vs_budget_curve(model, budgets, tokenizer=tok, prompts=prompts, out_csv=str(out_csv))
    assert "budgets" in res and "curve" in res
    assert out_csv.exists()

    # Parse CSV to extract mean_think_tokens
    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines and lines[0].startswith("budget,n,accuracy,f1,mean_think_tokens")
    vals = []
    for line in lines[1:]:
        parts = line.split(",")
        assert len(parts) == 5
        budget_val = int(parts[0])
        mean_think = float(parts[4])
        vals.append((budget_val, mean_think))

    # Ensure monotone increase in mean_think_tokens as budget grows
    for i in range(1, len(vals)):
        assert vals[i][1] >= vals[i-1][1]

