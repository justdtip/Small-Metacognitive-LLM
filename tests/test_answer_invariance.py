import pytest
from pathlib import Path

import train.eval_loop as el


def test_same_answer_different_hint(monkeypatch):
    # Stub decode_with_budget to emit different <strategy:...> tags but the same final numeric answer
    bodies = {
        "probe/experiment": "<think> reasoning variant A </think><answer> <strategy:probe/experiment> 42 </answer>",
        "working_backwards": "<think> different steps </think><answer> <strategy:working_backwards> 42 </answer>",
    }

    def _stub_decode(tok, model, input_ids, **kw):
        h = kw.get("style_tag") or ""
        return {"text": bodies.get(h, bodies["probe/experiment"]) }

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    res = el.check_answer_invariance_for_hints(tok, object(), "Prompt X", ["probe/experiment", "working_backwards"], decode_fn=_stub_decode)
    assert res.get("all_equal") is True
    ans = list((res.get("answers") or {}).values())
    assert all(a == "42" for a in ans)

