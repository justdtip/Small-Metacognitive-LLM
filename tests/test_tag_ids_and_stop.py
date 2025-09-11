from tina.tokenizer_utils import ensure_reasoning_tokens, SPECIAL_TOKENS


class DummyTokenizer:
    def __init__(self):
        self.vocab = {}
        self.additional_special_tokens = []
        self.unique_no_split_tokens = []
        self._next_id = 1
    def add_special_tokens(self, d):
        toks = d.get("additional_special_tokens", [])
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = self._next_id; self._next_id += 1
            if t not in self.additional_special_tokens:
                self.additional_special_tokens.append(t)
    def add_tokens(self, toks, special_tokens=False):
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = self._next_id; self._next_id += 1
    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, None)
    def encode(self, text, add_special_tokens=False):
        for t in self.additional_special_tokens:
            text = text.replace(t, f" {t} ")
        parts = [p for p in text.split() if p]
        ids = []
        for p in parts:
            if p in self.vocab:
                ids.append(self.vocab[p])
            else:
                self.vocab[p] = self._next_id; self._next_id += 1
                ids.append(self.vocab[p])
        return ids


def test_tag_round_trip_and_stop_sequences():
    tok = DummyTokenizer()
    ids = ensure_reasoning_tokens(tok)
    assert all(isinstance(v, int) for v in ids.values())
    # Each special tag must encode to exactly one token
    for t in SPECIAL_TOKENS:
        enc = tok.encode(t, add_special_tokens=False)
        assert isinstance(enc, list)
        assert len(enc) == 1, f"Tag {t} encodes into {enc}"
    # stop sequences must include </answer> and think stop must include </think>
    import json, pathlib
    svc = json.loads((pathlib.Path(__file__).resolve().parents[1]/'config'/'service_config.json').read_text())
    assert '</answer>' in (svc.get('stop_sequences') or [])
    assert '</think>' in (svc.get('think_stop_sequences') or [])
