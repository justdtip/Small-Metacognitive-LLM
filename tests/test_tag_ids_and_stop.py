from tina.tokenizer_utils import ensure_reasoning_tokens, SPECIAL_TOKENS
from transformers import AutoTokenizer
from pathlib import Path


def test_tag_round_trip_and_stop_sequences():
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    ids = ensure_reasoning_tokens(tok)
    assert all(isinstance(v, int) for v in ids.values())
    # Each special tag must encode to exactly one token
    for t in SPECIAL_TOKENS:
        enc = tok.encode(t, add_special_tokens=False)
        assert isinstance(enc, list)
        assert len(enc) == 1, f"Tag {t} encodes into {enc}"
    # stop sequences must include </answer> and think stop must include </think>
    import json, pathlib
    svc = json.loads((pathlib.Path(__file__).resolve().parents[1]/'config'/'service_config.json').read_text())
    assert '</answer>' in (svc.get('stop_sequences') or [])
    assert '</think>' in (svc.get('think_stop_sequences') or [])
