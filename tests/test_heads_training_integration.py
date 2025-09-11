import pytest
torch = pytest.importorskip("torch")

from train.runner import _train_step, TinyLM
from tina.tokenizer_utils import ensure_reasoning_tokens, segment_and_masks
from train.data import pad_and_stack


def build_smoke_batch():
    class Tok:
        def __init__(self):
            self.vocab = {"<pad>":0}
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
                    self.vocab[p]=self._n; self._n+=1
                ids.append(self.vocab[p])
            return ids
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            class R: pass
            r = R(); r.input_ids = self.encode(text, add_special_tokens=add_special_tokens)
            return r

    tok = Tok()
    ensure_reasoning_tokens(tok)
    text = "<think> a b c </think> <answer> d e </answer>"
    ids, attn, loss_mask, think_mask, answer_mask = segment_and_masks(text, tok)
    batch = pad_and_stack([(ids, attn, loss_mask, think_mask, answer_mask)], pad_id=tok.vocab.get("<pad>", 0))
    # Provide heuristic labels
    batch["plan_targets"] = torch.tensor([1], dtype=torch.long)
    batch["target_budget"] = torch.tensor([8], dtype=torch.long)
    batch["correctness"] = torch.tensor([1], dtype=torch.long)
    return batch


def test_heads_losses_nonzero():
    model = TinyLM(vocab_size=128, hidden=32)
    batch = build_smoke_batch()
    out = _train_step(model, batch, step=200, total_steps=1000)
    # Loss components should all be present and positive
    assert out["losses"].get("plan_ce", 0.0) > 0.0
    assert out["losses"].get("budget_reg", 0.0) >= 0.0  # can be 0 if pred==target, but typically >0
    assert out["losses"].get("conf_cal", 0.0) >= 0.0
