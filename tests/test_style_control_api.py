import pytest
import torch

from train.eval_loop import decode_with_budget, _extract_think_span
from tina.serve import _extract_answer


def test_style_changes_think_not_answer():
    class Tok:
        pad_token_id = 0
        eos_token_id = None
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            class R: pass
            r = R(); r.input_ids = torch.tensor([[1,2,3]], dtype=torch.long)
            return r
        def encode(self, s, add_special_tokens=False):
            # Map style hints to distinct ids
            if s == "<style:checklist>":
                return [101]
            if s == "<style:explainer>":
                return [102]
            return [5]
        def decode(self, ids, skip_special_tokens=True):
            # First call (THINK) returns text dependent on style id; second call (ANSWER) returns fixed answer
            if not hasattr(self, "_calls"):
                self._calls = 0
            self._calls += 1
            if self._calls == 1:
                if 101 in ids.tolist():
                    return "- step\n</think>\n"
                if 102 in ids.tolist():
                    return "Because ... </think>\n"
                return "</think>\n"
            return "<answer>OK</answer>"

    class Model:
        def __init__(self):
            self.device = torch.device('cpu')
        def generate(self, **kw):
            class O: pass
            o = O()
            inp = kw.get('input_ids')
            o.sequences = torch.cat([inp, torch.tensor([[7]], dtype=torch.long)], dim=1)
            return o

    tok = Tok(); model = Model()
    inp = tok("Q", return_tensors="pt").input_ids
    out_ck = decode_with_budget(tok, model, inp, think_budget=8, max_new_tokens=8, temperature=0.0, visible_cot=True, style_tag="checklist")
    out_ex = decode_with_budget(tok, model, inp, think_budget=8, max_new_tokens=8, temperature=0.0, visible_cot=True, style_tag="explainer")
    body_ck = out_ck["text"]; body_ex = out_ex["text"]
    # Extract answers must match
    assert _extract_answer(body_ck, include_think=False) == _extract_answer(body_ex, include_think=False)
    # Think-like prefix (before <answer>) differs in structure
    pre_ck = body_ck.split('<answer>')[0]
    pre_ex = body_ex.split('<answer>')[0]
    assert pre_ck != pre_ex
