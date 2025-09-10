#!/usr/bin/env python3
"""
Assert BOS/EOS ID harmony across tokenizer, base config, and generation config,
and verify reasoning tags are atomic (encode to exactly one token).

Usage:
  python3 tools/assert_config_ids.py --base-dir model/Base
Exits non-zero on mismatch; prints current IDs and tag encodings.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

TAGS = ["<think>", "</think>", "<answer>", "</answer>"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="model/Base", help="Path to base model dir with tokenizer/configs")
    args = ap.parse_args()
    base = Path(args.base_dir)
    cfg_path = base / "config.json"
    gen_path = base / "generation_config.json"
    tok_dir = base

    # Load tokenizer
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        sys.stderr.write("Transformers not installed. pip install transformers tokenizers\n")
        return 2

    try:
        tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True, local_files_only=True, trust_remote_code=True)
    except Exception as e:
        sys.stderr.write(f"Failed to load tokenizer from {tok_dir}: {e}\n")
        return 2

    # Load configs
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        with gen_path.open("r", encoding="utf-8") as f:
            gen = json.load(f)
    except Exception as e:
        sys.stderr.write(f"Failed to read configs: {e}\n")
        return 2

    bos_tok_id = getattr(tok, "bos_token_id", None)
    eos_tok_id = getattr(tok, "eos_token_id", None)
    ok = True

    # IDs from configs
    c_bos = cfg.get("bos_token_id")
    c_eos = cfg.get("eos_token_id")
    g_bos = gen.get("bos_token_id")
    g_eos = gen.get("eos_token_id")

    # Harmony checks
    if bos_tok_id != c_bos or bos_tok_id != g_bos:
        ok = False
        sys.stderr.write(f"bos_token_id mismatch: tokenizer={bos_tok_id} config.json={c_bos} generation_config.json={g_bos}\n")
    if eos_tok_id != c_eos or eos_tok_id != g_eos:
        ok = False
        sys.stderr.write(f"eos_token_id mismatch: tokenizer={eos_tok_id} config.json={c_eos} generation_config.json={g_eos}\n")

    # Tag atomicity
    for t in TAGS:
        try:
            ids = tok.encode(t, add_special_tokens=False)
        except Exception:
            ids = tok(t, add_special_tokens=False).input_ids
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) != 1:
            ok = False
            sys.stderr.write(f"Tag '{t}' encodes into {ids} (len={len(ids)}), expected 1 token.\n")

    if not ok:
        return 1

    print(json.dumps({
        "bos_token_id": bos_tok_id,
        "eos_token_id": eos_tok_id,
        "tags": {t: tok.encode(t, add_special_tokens=False) for t in TAGS},
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

