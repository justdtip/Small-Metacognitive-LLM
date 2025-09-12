import pytest
import torch
from pathlib import Path

from train.data import make_collate_fn
from tina.serve import IntrospectiveEngine, EngineConfig


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)


def test_e2e_valid_full():
    tok = _tok()
    collate = make_collate_fn(tok, strict=True)
    ex = {"text": "<think> alpha <plan> do </plan> beta <exec> run </exec> gamma <eval> ok </eval> </think> <answer> fine </answer>"}
    batch = collate([ex])
    tm = batch["think_mask"].bool()
    pm = batch["plan_mask"].bool()
    em = batch["exec_mask"].bool()
    vm = batch["eval_mask"].bool()
    # Masks exist and are inside think
    assert pm.shape == tm.shape and em.shape == tm.shape and vm.shape == tm.shape
    assert torch.all(pm <= tm)
    assert torch.all(em <= tm)
    assert torch.all(vm <= tm)
    # Disjoint
    assert torch.sum(pm & em).item() == 0
    assert torch.sum(pm & vm).item() == 0
    assert torch.sum(em & vm).item() == 0


def test_e2e_only_exec():
    tok = _tok()
    collate = make_collate_fn(tok, strict=True)
    ex = {"text": "<think> xx <exec> run </exec> yy </think> <answer> ok </answer>"}
    batch = collate([ex])
    # Only exec has non-zero; plan/eval masks exist but are zero-filled
    assert batch["exec_mask"].sum().item() > 0
    assert batch["plan_mask"].sum().item() == 0
    assert batch["eval_mask"].sum().item() == 0


def test_fuzz_overlap_no_crash():
    tok = _tok()
    collate = make_collate_fn(tok, strict=True)
    texts = [
        "<think> <plan> a <exec> b </plan> c </exec> </think> <answer> ok </answer>",
        "<think> <exec> a <eval> b </exec> c </eval> </think> <answer> ok </answer>",
        "<think> <plan> a </exec> b </plan> </think> <answer> ok </answer>",
    ]
    batch = collate([{"text": t} for t in texts])
    # No crash; check masks are bounded to think and non-overlapping
    tm = batch["think_mask"].bool()
    for name in ["plan_mask", "exec_mask", "eval_mask"]:
        m = batch[name].bool()
        assert torch.all(m <= tm)
    # pairwise disjoint
    pm, em, vm = batch["plan_mask"].bool(), batch["exec_mask"].bool(), batch["eval_mask"].bool()
    assert torch.sum(pm & em).item() == 0
    assert torch.sum(pm & vm).item() == 0
    assert torch.sum(em & vm).item() == 0


def test_stream_unicode_strip():
    # Build a minimal engine; assemble_cot_output must strip decomp/think tags in hidden mode
    class Tiny(torch.nn.Module):
        def forward(self, *a, **k):
            return {}
    tok = _tok()
    eng = IntrospectiveEngine(model=Tiny(), tokenizer=tok, cfg=EngineConfig(visible_cot=False), hidden_size=16, num_layers=0)
    think_text = "<think> • plan <plan> étape 🧠 </plan> → <exec> exécuter ✅ </exec> ≈ <eval> vérifier áéîöü </eval> </think>"
    answer_text = "<answer> result with no tags </answer>"
    out = eng.assemble_cot_output(think_text, answer_text, visible_cot=False)
    # Ensure no decomposition or think tags remain
    for tag in ["<think>", "</think>", "<plan>", "</plan>", "<exec>", "</exec>", "<eval>", "</eval>"]:
        assert tag not in out
