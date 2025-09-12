import json
from pathlib import Path
import pytest

from train.data import make_collate_fn
from train.metrics import aggregate
from train.eval_loop import compute_eval_metrics
from transformers import AutoTokenizer


def test_infers_and_slices(tmp_path):
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    collate = make_collate_fn(tok, strict=True)

    recs = [
        {"text": "<think> a b </think><answer> ok </answer>", "style_tag": "checklist"},
        {"text": "<think> a b </think><answer> ok </answer>", "style_tag": "explainer"},
    ]
    batch = collate(recs)
    # style_id inferred and not -1
    import torch
    assert torch.all(batch["style_id"] != -1)
    # Aggregate slice metrics using style_tag
    eval_recs = [
        {"body": r["text"], "gold": "ok", "style_tag": r["style_tag"]} for r in recs
    ]
    out = aggregate(eval_recs, compute_eval_metrics, slice_keys=("style_tag",))
    assert "style_tag" in out.get("slices", {})
    assert "checklist" in out["slices"]["style_tag"]
    assert "explainer" in out["slices"]["style_tag"]
