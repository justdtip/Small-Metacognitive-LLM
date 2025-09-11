from tina.tokenizer_utils import ensure_reasoning_tokens


class DummyTokenizer:
    def __init__(self):
        self.vocab = {}
        self.additional_special_tokens = []
        self.unique_no_split_tokens = []
        self._next = 1
    def add_special_tokens(self, d):
        toks = d.get("additional_special_tokens", [])
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = self._next; self._next += 1
            if t not in self.additional_special_tokens:
                self.additional_special_tokens.append(t)
    def add_tokens(self, toks, special_tokens=False):
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = self._next; self._next += 1
    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, None)
    def encode(self, text, add_special_tokens=False):
        for t in self.additional_special_tokens:
            text = text.replace(t, f" {t} ")
        return [self.vocab.get(p, 0) for p in text.split() if p]
    def tokenize(self, text):
        for t in self.additional_special_tokens:
            text = text.replace(t, f" {t} ")
        return [p for p in text.split() if p]


def test_tag_round_trip_and_stop_sequences():
    tok = DummyTokenizer()
    ids = ensure_reasoning_tokens(tok)
    assert all(isinstance(v, int) for v in ids.values())
    import json, pathlib
    svc = json.loads((pathlib.Path(__file__).resolve().parents[1]/'config'/'service_config.json').read_text())
    assert '</answer>' in (svc.get('stop_sequences') or [])
    assert '</think>' in (svc.get('think_stop_sequences') or [])
