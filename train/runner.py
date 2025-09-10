from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Dict, Any
import math

import torch
import torch.nn as nn

from tina.tokenizer_utils import ensure_reasoning_tokens, segment_and_masks
from train.data import pad_and_stack
from train.losses import compute_losses
from tina.serve import _extract_answer, StopOnTags
from train.reward import reward_fn, dry_run_mocked
import time, uuid, json


@dataclass
class SmokeCfg:
    vocab_size: int = 256
    hidden_size: int = 64
    taps: Sequence[int] = (1, 2)
    lr: float = 1e-3
    max_len: int = 128


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)
        y, _ = self.rnn(x)
        logits = self.lm_head(y)
        return logits


class DummyTok:
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.additional_special_tokens = []
        self.unique_no_split_tokens = []
        self._next = 5
        # seed a tiny base vocab
        for t in ["<pad>", "<unk>", "a", "b", "c"]:
            self.vocab[t] = len(self.vocab)

    def add_special_tokens(self, d):
        toks = d.get("additional_special_tokens", [])
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = self._next; self._next += 1
            if t not in self.additional_special_tokens:
                self.additional_special_tokens.append(t)

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, None)

    def encode(self, text: str, add_special_tokens: bool = False):
        for t in self.additional_special_tokens:
            text = text.replace(t, f" {t} ")
        parts = [p for p in text.split() if p]
        ids = []
        for p in parts:
            if p not in self.vocab:
                self.vocab[p] = self._next; self._next += 1
            ids.append(self.vocab[p])
        return ids

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        class R: pass
        r = R(); r.input_ids = self.encode(text, add_special_tokens=add_special_tokens)
        return r

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(i, "?") for i in ids)


def sft_one_step_smoke() -> Dict[str, Any]:
    """One-step SFT smoke: trains TinyLM to predict answer tokens masked by segment_and_masks.
    Returns a dict of metrics and the extracted answer from a synthetic sample.
    """
    tok = DummyTok()
    ensure_reasoning_tokens(tok)
    model = TinyLM(vocab_size=max(tok.vocab.values()) + 16, hidden=64)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Synthetic sample
    text = "<think> plan steps </think> <answer> final answer </answer>"
    ids, attn, loss_mask, think_mask, answer_mask = segment_and_masks(text, tok)

    batch = pad_and_stack([(ids, attn, loss_mask, think_mask, answer_mask)], pad_id=tok.vocab.get("<pad>", 0))
    input_ids = batch["input_ids"]
    # Teacher forcing: next token prediction; build labels by shifting left
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    # mask labels to only answer tokens
    loss_mask_t = batch["loss_mask"].bool()
    labels = torch.where(loss_mask_t, labels, torch.full_like(labels, -100))

    logits = model(input_ids)
    losses = compute_losses(logits, labels, gate_modules=None, weights={"answer_ce": 1.0})
    opt.zero_grad()
    losses["total"].backward()
    opt.step()

    # Eval extraction parity
    extracted = _extract_answer(text, include_think=False)
    return {"loss": float(losses["total"].item()), "extracted": extracted}


def main():
    t0 = time.time()
    rid = str(uuid.uuid4())
    out = sft_one_step_smoke()
    # RL dry-run to show reward sensitivity to think tokens
    rl = dry_run_mocked()
    log = {
        "ts": int(time.time()*1000),
        "request_id": rid,
        "loss": out["loss"],
        "extracted": out["extracted"],
        "rl_short": rl["short"],
        "rl_long": rl["long"],
        "elapsed_ms": int((time.time()-t0)*1000),
    }
    print(json.dumps(log))


if __name__ == "main":  # not executed by default
    main()
