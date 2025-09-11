import json
import pytest

from tina.serve import IntrospectiveEngine, EngineConfig


def test_persists_and_applies_thresholds(tmp_path):
    # Create a calibration blob
    blob = {
        "conf_temp": 1.5,
        "plan_thresholds": {"deliberate": 0.7},
        "budget_posthoc_clip": 96,
    }
    cal = tmp_path / "heads_calibration.json"
    cal.write_text(json.dumps(blob), encoding="utf-8")

    # Minimal tokenizer and model
    class DummyTok:
        def __init__(self):
            self.vocab = {}
            self.additional_special_tokens = []
            self.unique_no_split_tokens = []
            self._n = 1
        def add_special_tokens(self, d):
            toks = d.get("additional_special_tokens", [])
            for t in toks:
                if t not in self.vocab:
                    self.vocab[t] = self._n; self._n += 1
                if t not in self.additional_special_tokens:
                    self.additional_special_tokens.append(t)
        def convert_tokens_to_ids(self, t):
            return self.vocab.get(t, 0)
        def encode(self, s, add_special_tokens=False):
            return [self.convert_tokens_to_ids(s) or 1]

    class DummyModel:
        device = 'cpu'
        dtype = None
        def parameters(self):
            if False:
                yield None
            return iter(())
        def __call__(self, *a, **k):
            class O:
                pass
            o = O()
            # provide hidden_states list
            import torch
            o.hidden_states = [torch.zeros(1, 2, 4)]
            return o

    model = DummyModel()
    tok = DummyTok()
    eng = IntrospectiveEngine(model=model, tokenizer=tok, cfg=EngineConfig(calibration_path=str(cal)), hidden_size=16, num_layers=0)
    # Conf temp loaded
    assert eng.last_stats.get("parity_digest", {}).get("conf_temp") == 1.5 or eng.last_stats.get("conf_temp") == 1.5

    # Stub metacog to control outputs
    import torch
    class StubHeads:
        def __call__(self, B_max: int = 256):
            plan_logits = torch.tensor([[0.0, 3.0, 0.0, 0.0]])  # deliberate ~0.71
            budget = torch.tensor([[128.0]])
            conf = torch.tensor([[0.5]])
            return {"plan_logits": plan_logits, "budget": budget, "confidence": conf}
        def clear_cache(self):
            pass

    eng.metacog = StubHeads()
    import torch as _t
    inp = _t.ones((1, 4), dtype=_t.long)
    b = eng._estimate_budget(inp)
    # Budget may not exceed the post-hoc clip
    assert b <= 96
    # Plan label should be 'deliberate' under threshold rule
    assert eng.last_stats.get("plan_label") == "deliberate"

    # Change logits below threshold for deliberate
    class StubHeads2(StubHeads):
        def __call__(self, B_max: int = 256):
            plan_logits = torch.tensor([[0.0, 0.0, 2.0, 0.0]])  # deliberate low; class 2 high
            budget = torch.tensor([[64.0]])
            conf = torch.tensor([[0.5]])
            return {"plan_logits": plan_logits, "budget": budget, "confidence": conf}

    eng.metacog = StubHeads2()
    b2 = eng._estimate_budget(inp)
    # Now plan should not be 'deliberate'
    assert eng.last_stats.get("plan_label") != "deliberate"

