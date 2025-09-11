import pytest

from tina.serve import IntrospectiveEngine, EngineConfig


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


def test_slack_ratio_parity_digest():
    model = DummyModel()
    tok = DummyTok()
    eng = IntrospectiveEngine(model=model, tokenizer=tok, cfg=EngineConfig(), hidden_size=16, num_layers=0)
    assert "soft_cap_slack_ratio" in eng.last_stats


def test_altered_config_changes_slack(tmp_path, monkeypatch):
    import json, pathlib, types, torch
    from train import eval_loop as ev

    # Backup and override service_config.json
    cfg_path = pathlib.Path(__file__).resolve().parents[1] / 'config' / 'service_config.json'
    orig = cfg_path.read_text(encoding='utf-8')
    try:
        data = json.loads(orig)
        data['soft_cap_slack_ratio'] = 0.5
        cfg_path.write_text(json.dumps(data), encoding='utf-8')

        # Engine should pick up updated slack ratio
        model = DummyModel()
        tok = DummyTok()
        eng = IntrospectiveEngine(model=model, tokenizer=tok, cfg=EngineConfig(), hidden_size=16, num_layers=0)
        assert abs(float(eng.last_stats.get('soft_cap_slack_ratio', 0.0)) - 0.5) < 1e-9

        # Monkeypatch SlackStop in eval loop to capture the ratio used
        captured = {"ratio": None}

        class StubSlack:
            def __init__(self, *, base_len: int, budget: int, slack_ratio: float = 0.2):
                captured["ratio"] = float(slack_ratio)
            def __call__(self, input_ids, scores, **kwargs):
                return False

        monkeypatch.setattr(ev, "SlackStop", StubSlack)

        # Minimal tokenizer and model to invoke decode_with_budget
        class TT:
            pad_token_id = 0
            eos_token_id = None
            def __call__(self, text, add_special_tokens=False, return_tensors=None):
                class R: pass
                r = R(); r.input_ids = torch.tensor([[1]], dtype=torch.long)
                return r
            def encode(self, s, add_special_tokens=False):
                return [1]
            def decode(self, ids, skip_special_tokens=False):
                return "</think>"

        class MM:
            device = 'cpu'
            def generate(self, **k):
                class O:
                    pass
                o = O()
                import torch as _t
                base = k.get('input_ids'); L = base.shape[1] if hasattr(base, 'shape') else 1
                o.sequences = _t.ones((1, L + 1), dtype=_t.long)
                return o

        t2 = TT(); m2 = MM()
        import torch as _t
        inp = _t.tensor([[1, 2, 3]], dtype=_t.long)
        out = ev.decode_with_budget(t2, m2, inp, think_budget=8, max_new_tokens=4, temperature=0.0, visible_cot=False)
        # Slack ratio captured should equal the configured ratio
        assert abs(float(captured["ratio"] or 0.0) - 0.5) < 1e-9
    finally:
        cfg_path.write_text(orig, encoding='utf-8')
