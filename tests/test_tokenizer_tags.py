from tina.tokenizer_utils import ensure_reasoning_tokens, segment_and_masks, SPECIAL_TOKENS
from transformers import AutoTokenizer
from pathlib import Path

def test_tags_are_atomic():
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    ids = ensure_reasoning_tokens(tok)
    for t in SPECIAL_TOKENS:
        enc = tok.encode(t, add_special_tokens=False)
        assert isinstance(enc, list) and len(enc) == 1
        assert ids[t] == enc[0]

def test_segment_and_masks_spans():
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    ensure_reasoning_tokens(tok)
    text = "<think> aaa bbb </think> <answer> ccc ddd </answer>"
    input_ids, attention_mask, loss_mask, think_mask, answer_mask = segment_and_masks(text, tok)
    # non-zero spans between tags
    assert sum(think_mask) > 0
    assert sum(answer_mask) > 0
    # loss defaults to answer mask
    assert loss_mask == answer_mask
