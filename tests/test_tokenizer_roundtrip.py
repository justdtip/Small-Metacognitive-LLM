from tokenizer_utils import ensure_reasoning_tokens
from transformers import AutoTokenizer
from pathlib import Path


def test_reasoning_tokens_added_and_nosplit():
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    ids = ensure_reasoning_tokens(tok, model=None)
    for t in ["<think>","</think>","<answer>","</answer>"]:
        # must map to an id and encode to single token
        tid = tok.convert_tokens_to_ids(t)
        assert isinstance(tid, int) and tid is not None
        enc = tok.encode(t, add_special_tokens=False)
        assert isinstance(enc, list) and len(enc) == 1
        # If tokenizer exposes no-split list, tag must be present
        if hasattr(tok, 'unique_no_split_tokens'):
            assert t in tok.unique_no_split_tokens
