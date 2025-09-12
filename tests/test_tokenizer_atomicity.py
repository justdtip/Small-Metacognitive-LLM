import pytest
from pathlib import Path

from tina.tokenizer_utils import ensure_reasoning_tokens, REASONING_TOKENS


def test_single_token_enforced_with_fast_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    ids = ensure_reasoning_tokens(tok)
    # All tags should be single-token encodings
    for t in REASONING_TOKENS:
        enc = tok.encode(t, add_special_tokens=False)
        assert isinstance(enc, list)
        assert len(enc) == 1, f"Tag {t} must encode to a single token, got {enc}"
        # convert_tokens_to_ids should resolve the same id
        tid = tok.convert_tokens_to_ids(t)
        assert tid is not None
        assert enc[0] == tid


def test_single_token_violation_raises():
    # Construct a pathological tokenizer that never treats tags as atomic
    class BadTok:
        def __init__(self):
            self.additional_special_tokens = []
            self.unique_no_split_tokens = []
        def add_special_tokens(self, d):
            # ignore additions (simulate broken setup)
            pass
        def convert_tokens_to_ids(self, t):
            # fail to resolve ids for tags
            return None
        def encode(self, text, add_special_tokens=False):
            # split into many pseudo-ids
            return list(range(max(1, len(str(text)))))

    tok = BadTok()
    with pytest.raises(ValueError):
        ensure_reasoning_tokens(tok)

