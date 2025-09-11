import json
from pathlib import Path

from tina.tokenizer_utils import ensure_reasoning_tokens, SPECIAL_TOKENS
from tina.serve import StopOnTags as ServeStop
import train.eval_loop as ev
from transformers import AutoTokenizer


def test_tags_atomic_and_in_parity():
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    ensure_reasoning_tokens(tok)

    # Each special tag encodes into exactly one token id
    for t in SPECIAL_TOKENS:
        ids = tok.encode(t, add_special_tokens=False)
        assert isinstance(ids, list) and len(ids) == 1, f"tag {t} enc={ids}"

    # Load service config tags
    svc_path = Path(__file__).resolve().parents[1] / 'config' / 'service_config.json'
    svc = json.loads(svc_path.read_text(encoding='utf-8'))
    ans_tags = tuple(svc.get("stop_sequences") or ["</answer>"])
    think_tags = tuple(svc.get("think_stop_sequences") or ["</think>"])

    # Build StopOnTags for serve and eval and assert stop id parity
    serve_ans = ServeStop(tok, ans_tags)
    eval_ans = ev.StopOnTags(tok, ans_tags)
    assert serve_ans.stop_ids == eval_ans.stop_ids

    serve_think = ServeStop(tok, think_tags)
    eval_think = ev.StopOnTags(tok, think_tags)
    assert serve_think.stop_ids == eval_think.stop_ids
