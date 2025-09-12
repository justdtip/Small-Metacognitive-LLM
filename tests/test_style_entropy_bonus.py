import pytest
import torch

from train.runner import _train_step, TinyLM
from tina.tokenizer_utils import ensure_reasoning_tokens, segment_and_masks
from train.data import pad_and_stack


def _batch_with_style(style_id: int):
    class Tok:
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
            ids = []
            for p in text.split():
                if p not in self.vocab:
                    self.vocab[p] = self._n; self._n += 1
                ids.append(self.vocab[p])
            return ids
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            class R: pass
            r = R(); r.input_ids = self.encode(text, add_special_tokens=add_special_tokens)
            return r

    tok = Tok()
    ensure_reasoning_tokens(tok)
    text = "<think> a b c </think> <answer> ok </answer>"
    ids, attn, loss_mask, think_mask, answer_mask = segment_and_masks(text, tok)
    batch = pad_and_stack([(ids, attn, loss_mask, think_mask, answer_mask)], pad_id=tok.vocab.get("<pad>", 0))
    batch["style_id"] = torch.tensor([style_id], dtype=torch.long)
    return batch


def test_entropy_bonus_penalizes_single_style():
    torch.manual_seed(0)
    model = TinyLM(vocab_size=128, hidden=32)
    batch = _batch_with_style(2)
    out0 = _train_step(model, batch, step=0, total_steps=1, style_entropy_weight=0.0)
    out1 = _train_step(model, batch, step=0, total_steps=1, style_entropy_weight=0.1)
    # Total with entropy penalty should be greater for degenerate style distribution
    assert out1["losses"]["total"] > out0["losses"]["total"]
    assert out1["losses"].get("style_entropy", 0.0) >= 0.0
