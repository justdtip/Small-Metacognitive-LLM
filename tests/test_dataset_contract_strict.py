import pytest
from pathlib import Path

from train.data import make_collate_fn
from transformers import AutoTokenizer


def _tok():
    return AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)


def test_strict_mode_raises_with_index():
    tok = _tok()
    collate = make_collate_fn(tok, strict=True)
    # Missing </answer>
    bad = [
        {"text": "<think> plan </think> <answer> missing close"},
        {"text": "<think> ok </think> <answer> ok </answer>"},
    ]
    with pytest.raises(ValueError) as ei:
        _ = collate(bad)
    msg = str(ei.value)
    assert "Malformed tags in record" in msg
    assert "0" in msg  # record index present


def test_non_strict_wraps_minimally():
    tok = _tok()
    collate = make_collate_fn(tok, strict=False)
    # One malformed, one well-formed
    recs = [
        {"text": "No tags here at all"},
        {"text": "<think> a </think> <answer> ok </answer>"},
    ]
    batch = collate(recs)
    # Should proceed and produce masks
    import torch
    assert torch.sum(batch["answer_mask"]).item() > 0
    # think_mask should exist (may be zero if no think content was present)
    assert batch["think_mask"].shape == batch["answer_mask"].shape

