import pytest

from train.data import make_collate_fn
from transformers import AutoTokenizer
from pathlib import Path


class DummyTok:
    def __init__(self):
        self.vocab = {"<pad>": 0}
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
        return self.vocab.get(t, None)
    def encode(self, text, add_special_tokens=False):
        for t in self.additional_special_tokens:
            text = text.replace(t, f" {t} ")
        ids=[]
        for p in text.split():
            if p not in self.vocab:
                self.vocab[p] = self._n; self._n += 1
            ids.append(self.vocab[p])
        return ids
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        class R: pass
        r = R(); r.input_ids = self.encode(text, add_special_tokens=add_special_tokens)
        return r


def test_rejects_missing_tags():
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    collate = make_collate_fn(tok, strict=True)
    # Missing </answer>
    bad = [{"text": "<think> t </think> <answer>"}]
    with pytest.raises(ValueError) as ei:
        _ = collate(bad)
    assert "Malformed tags" in str(ei.value)


def test_passes_well_formed_sample():
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    collate = make_collate_fn(tok, strict=True)
    good = [{"text": "<think> consider bulbs </think> <answer> switch 2 and 3 then 1 </answer>"}]
    batch = collate(good)
    # sums > 0 for both masks
    import torch
    assert torch.sum(batch["think_mask"]).item() > 0
    assert torch.sum(batch["answer_mask"]).item() > 0
