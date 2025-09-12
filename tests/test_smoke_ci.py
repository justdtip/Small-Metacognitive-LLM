import pytest
from pathlib import Path


def test_imports_and_tokenizer_atomicity():
    # Basic imports
    import tina.tokenizer_utils as tu
    from train import eval_loop  # noqa: F401
    from train import runner     # noqa: F401

    # Atomicity check with local tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    ids = tu.ensure_reasoning_tokens(tok)
    for t in tu.REASONING_TOKENS:
        enc = tok.encode(t, add_special_tokens=False)
        assert isinstance(enc, list) and len(enc) == 1
