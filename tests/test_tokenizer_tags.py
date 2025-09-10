from tina.tokenizer_utils import ensure_reasoning_tokens, segment_and_masks, SPECIAL_TOKENS

class DummyTokenizer:
    def __init__(self):
        self.vocab = {}
        self.additional_special_tokens = []
        self.unique_no_split_tokens = []
        self._next_id = 100
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
        # naive split: isolate special tokens first
        for t in self.additional_special_tokens:
            text = text.replace(t, f" {t} ")
        parts = [p for p in text.split() if p]
        ids = []
        for p in parts:
            if p in self.vocab:
                ids.append(self.vocab[p])
            else:
                # allocate new id for arbitrary token
                self.vocab[p] = self._next_id; self._next_id += 1
                ids.append(self.vocab[p])
        return ids
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        class R: pass
        r = R(); r.input_ids = self.encode(text, add_special_tokens=add_special_tokens)
        return r
    def decode(self, ids, skip_special_tokens=False):
        inv = {v:k for k,v in self.vocab.items()}
        return " ".join(inv.get(i, "?") for i in ids)

def test_tags_are_atomic():
    tok = DummyTokenizer()
    ids = ensure_reasoning_tokens(tok)
    for t in SPECIAL_TOKENS:
        enc = tok.encode(t, add_special_tokens=False)
        assert len(enc) == 1
        assert ids[t] == enc[0]

def test_segment_and_masks_spans():
    tok = DummyTokenizer()
    ensure_reasoning_tokens(tok)
    text = "<think> aaa bbb </think> <answer> ccc ddd </answer>"
    input_ids, attention_mask, loss_mask, think_mask, answer_mask = segment_and_masks(text, tok)
    # think spans should cover tokens between the think tags
    # answer spans should cover tokens between the answer tags
    # verify masks have non-zero entries
    assert sum(think_mask) == 2
    assert sum(answer_mask) == 2
    # loss defaults to answer mask
    assert loss_mask == answer_mask

