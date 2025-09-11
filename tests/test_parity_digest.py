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
