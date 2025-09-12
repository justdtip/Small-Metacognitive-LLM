from train.data import make_collate_fn
from transformers import AutoTokenizer
from pathlib import Path


def test_targets_follow_bins():
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    collate = make_collate_fn(tok, strict=True)
    recs = [
        {"text": "<think> a </think><answer> ok </answer>", "difficulty_bin": 0},
        {"text": "<think> a b </think><answer> ok </answer>", "difficulty_bin": 1},
        {"text": "<think> a b c </think><answer> ok </answer>", "difficulty_bin": 2},
    ]
    batch = collate(recs)
    tg = batch["target_budget"].tolist()
    assert tg[0] == 16 and tg[1] == 64 and tg[2] == 128
