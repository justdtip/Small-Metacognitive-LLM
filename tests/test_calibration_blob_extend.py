import json
from pathlib import Path

import pytest
import torch

from tina.serve import IntrospectiveEngine, EngineConfig


class TinyModel(torch.nn.Module):
    def forward(self, input_ids=None, attention_mask=None, use_cache=False, output_hidden_states=True, return_dict=True):
        # Minimal forward compatible with engine dry run
        class O: pass
        return O()


class DummyTok:
    def __init__(self):
        self.vocab = {"<pad>": 0}
        self.additional_special_tokens = []
        self.unique_no_split_tokens = []
        self._n = 1
        self.pad_token_id = 0
    def add_special_tokens(self, d):
        toks = d.get("additional_special_tokens", [])
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = self._n; self._n += 1
            if t not in self.additional_special_tokens:
                self.additional_special_tokens.append(t)
    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, None)
    def encode(self, text, add_special_tokens=False):
        for t in self.additional_special_tokens:
            text = text.replace(t, f" {t} ")
        ids=[]
        for p in text.split():
            if p not in self.vocab:
                self.vocab[p]=self._n; self._n+=1
            ids.append(self.vocab[p])
        return ids
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        class R: pass
        r = R(); r.input_ids = self.encode(text, add_special_tokens=add_special_tokens)
        return r


def test_thresholds_and_clip_applied(tmp_path, monkeypatch):
    # Calibration blob with conf_temp=1.5, deliberate threshold, and clip=96
    blob = {
        "conf_temp": 1.5,
        "plan_thresholds": {"deliberate": 0.5},
        "budget_posthoc_clip": 96,
    }
    cpath = tmp_path / "cal.json"
    cpath.write_text(json.dumps(blob), encoding="utf-8")

    # Build engine with dummy model/tokenizer
    model = TinyModel()
    tok = DummyTok()
    cfg = EngineConfig(visible_cot=False, budget_cap=256, calibration_path=str(cpath))
    eng = IntrospectiveEngine(model, tok, cfg, hidden_size=16, num_layers=0)

    # Monkeypatch metacog heads to return deterministic outputs
    def _fake_heads(self, B_max: int = 256):
        plan_logits = torch.tensor([[0.0, 5.0, 0.0, 0.0]], dtype=torch.float32)  # high for 'deliberate'
        budget = torch.tensor([[200.0]], dtype=torch.float32)
        confidence = torch.tensor([[0.9]], dtype=torch.float32)
        return {"plan_logits": plan_logits, "budget": budget, "confidence": confidence}

    from tina.metacog_heads import MetacogHeads
    monkeypatch.setattr(MetacogHeads, "forward", _fake_heads, raising=True)

    # Run budget estimation (no real generation)
    inp = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    b = eng._estimate_budget(inp)

    # Budget is clipped to 96
    assert int(b) <= 96
    assert eng.last_stats.get("think_budget") <= 96

    # Plan selection honors threshold ('deliberate')
    assert eng.last_stats.get("plan_label") in ("deliberate", 1)

    # Confidence temperature is applied/present
    assert eng.last_stats.get("conf_temp") == pytest.approx(1.5)

