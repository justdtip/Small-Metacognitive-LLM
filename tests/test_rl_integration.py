import pytest
torch = pytest.importorskip("torch")

from train.runner import rl_phase_step, TinyLM
from tina.tokenizer_utils import ensure_reasoning_tokens, segment_and_masks
from train.data import pad_and_stack


def _smoke_batch():
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
    text = "<think> a b c d </think> <answer> ok </answer>"
    ids, attn, loss_mask, think_mask, answer_mask = segment_and_masks(text, tok)
    batch = pad_and_stack([(ids, attn, loss_mask, think_mask, answer_mask)], pad_id=tok.vocab.get("<pad>", 0))
    # RL fields: observed think tokens and correctness
    batch["think_tokens_used"] = torch.tensor([int(sum(think_mask))], dtype=torch.long)
    batch["correctness"] = torch.tensor([1], dtype=torch.long)
    return batch


def test_rl_phase_step_updates_mu():
    model = TinyLM(vocab_size=128, hidden=32)
    batch = _smoke_batch()
    pol, stats0 = rl_phase_step(model, batch)
    pol2, stats1 = rl_phase_step(model, batch, policy=pol)
    # Expect μ to not increase (preference for concision under penalty)
    assert stats1["mu_after"] <= stats0["mu_after"] + 1e-3


def test_mu_reduction_depends_on_labels():
    torch.manual_seed(0)
    model_h = TinyLM(vocab_size=128, hidden=32)
    model_l = TinyLM(vocab_size=128, hidden=32)

    # Build batches with the same think length but different correctness labels
    def _batch_with_corr(val: int):
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
        text = "<think> a b c d </think> <answer> ok </answer>"
        ids, attn, loss_mask, think_mask, answer_mask = segment_and_masks(text, tok)
        batch = pad_and_stack([(ids, attn, loss_mask, think_mask, answer_mask)], pad_id=tok.vocab.get("<pad>", 0))
        batch["think_tokens_used"] = torch.tensor([int(sum(think_mask))], dtype=torch.long)
        batch["correctness"] = torch.tensor([int(val)], dtype=torch.long)
        return batch

    b_hi = _batch_with_corr(1)
    b_lo = _batch_with_corr(0)

    # High-correctness case
    pol_h, sh0 = rl_phase_step(model_h, b_hi)
    # Snapshot state after first update, then apply a second update
    sd_before = {k: v.clone() for k, v in pol_h.state_dict().items()}
    st_h1 = rl_phase_step(model_h, b_hi, policy=pol_h)[1]

    # Low-correctness case
    pol_l, sl0 = rl_phase_step(model_l, b_lo)
    st_l1 = rl_phase_step(model_l, b_lo, policy=pol_l)[1]

    # Compare μ after two steps
    mu_h = st_h1["mu_after"]
    mu_l = st_l1["mu_after"]
    assert mu_h <= mu_l + 1e-3  # budgets should be pushed lower when correctness is high

    # Ensure there is adaptation signal: μ decreased more (or to a lower value) in the high-correctness setting
    # than in the low-correctness setting.
    mu_h0 = sh0["mu_mean"]
    mu_l0 = sl0["mu_mean"]
    d_h = mu_h0 - mu_h
    d_l = mu_l0 - mu_l
    assert (d_h > d_l - 1e-3) or (mu_h < mu_l)
