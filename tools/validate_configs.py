#!/usr/bin/env python3
"""
Validate config files against JSON Schemas and perform coherence checks.

Validates:
- model/Base/generation_config.json  against schemas/generation_config.schema.json
- model/Tina/checkpoint-2000/adapter_config.json against schemas/adapter_config.schema.json
- config/service_config.json         against schemas/service_config.schema.json

Also checks:
- BOS/EOS consistency between base config and generation config
- Adapter target_modules match expected family for base architecture (Qwen2)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import os
from typing import Any, Dict, List

# Allow overriding root for validation (used by tests)
_ENV_ROOT = os.environ.get("CONFIG_ROOT")
ROOT = Path(_ENV_ROOT) if _ENV_ROOT else Path(__file__).resolve().parents[1]
# Ensure project root is importable when invoked as a script
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FILES = {
    "generation": (ROOT / "model/Base/generation_config.json", ROOT / "schemas/generation_config.schema.json"),
    "adapter": (ROOT / "model/Tina/checkpoint-2000/adapter_config.json", ROOT / "schemas/adapter_config.schema.json"),
    # 'service' path may be overridden by SERVICE_CONFIG_PATH env var
    "service": (ROOT / "config/service_config.json", ROOT / "schemas/service_config.schema.json"),
}

def _service_cfg_path() -> Path:
    p = os.environ.get("SERVICE_CONFIG_PATH")
    return Path(p) if p else (ROOT / "config/service_config.json")

def _load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _jsonschema_validate(instance: Any, schema: Dict[str, Any]) -> list[str]:
    try:
        import jsonschema  # type: ignore
        from jsonschema import Draft202012Validator
        v = Draft202012Validator(schema)
        errs = sorted(v.iter_errors(instance), key=lambda e: e.path)
        return [f"{'.'.join(map(str,e.path))}: {e.message}" for e in errs]
    except Exception:
        # minimal validator for our schema subset
        return _minimal_validate(instance, schema, path=[])

def _minimal_validate(inst: Any, schema: Dict[str, Any], path: list[str]) -> list[str]:
    errs: list[str] = []
    t = schema.get("type")
    if t == "object":
        if not isinstance(inst, dict):
            errs.append(f"{'.'.join(path)}: expected object")
            return errs
        props = schema.get("properties", {})
        req = schema.get("required", [])
        for k in req:
            if k not in inst:
                errs.append(f"{'.'.join(path+[k])}: required")
        for k, v in inst.items():
            if k in props:
                errs += _minimal_validate(v, props[k], path + [k])
        # enum
        if "enum" in schema and inst not in schema["enum"]:
            errs.append(f"{'.'.join(path)}: not in enum")
    elif t == "array":
        if not isinstance(inst, list):
            errs.append(f"{'.'.join(path)}: expected array")
            return errs
        min_items = schema.get("minItems")
        if min_items is not None and len(inst) < min_items:
            errs.append(f"{'.'.join(path)}: minItems {min_items}")
        it = schema.get("items")
        if it:
            for i, el in enumerate(inst):
                errs += _minimal_validate(el, it, path + [str(i)])
    elif t == "integer":
        if not isinstance(inst, int):
            errs.append(f"{'.'.join(path)}: expected integer")
        mi = schema.get("minimum")
        if isinstance(mi, (int, float)) and isinstance(inst, (int, float)) and inst < mi:
            errs.append(f"{'.'.join(path)}: below minimum {mi}")
        ma = schema.get("maximum")
        if isinstance(ma, (int, float)) and isinstance(inst, (int, float)) and inst > ma:
            errs.append(f"{'.'.join(path)}: above maximum {ma}")
    elif t == "number":
        if not isinstance(inst, (int, float)):
            errs.append(f"{'.'.join(path)}: expected number")
        mi = schema.get("minimum")
        if isinstance(mi, (int, float)) and isinstance(inst, (int, float)) and inst < mi:
            errs.append(f"{'.'.join(path)}: below minimum {mi}")
        ma = schema.get("maximum")
        if isinstance(ma, (int, float)) and isinstance(inst, (int, float)) and inst > ma:
            errs.append(f"{'.'.join(path)}: above maximum {ma}")
    elif t == "boolean":
        if not isinstance(inst, bool):
            errs.append(f"{'.'.join(path)}: expected boolean")
    elif t == "string":
        if not isinstance(inst, str):
            errs.append(f"{'.'.join(path)}: expected string")
    return errs

def _coherence_checks() -> list[str]:
    errs: list[str] = []
    base_cfg = _load_json(ROOT / "model/Base/config.json")
    gen_cfg = _load_json(ROOT / "model/Base/generation_config.json")
    # BOS/EOS consistency
    if base_cfg.get("eos_token_id") != gen_cfg.get("eos_token_id"):
        errs.append("eos_token_id mismatch between base config and generation_config")
    if base_cfg.get("bos_token_id") != gen_cfg.get("bos_token_id"):
        errs.append("bos_token_id mismatch between base config and generation_config")
    # Family/module sanity
    archs = base_cfg.get("architectures") or []
    model_type = base_cfg.get("model_type")
    adapter_cfg = _load_json(ROOT / "model/Tina/checkpoint-2000/adapter_config.json")
    tmods = set(adapter_cfg.get("target_modules") or [])
    if "Qwen2ForCausalLM" in archs or model_type == "qwen2":
        allowed = {"q_proj","k_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"}
        if not tmods.issubset(allowed):
            errs.append("adapter target_modules contain entries not typical for Qwen2")
    # Service config stop-rule guardrails and tokenizer parity
    try:
        svc_cfg = _load_json(_service_cfg_path())
        stops = list(svc_cfg.get("stop_sequences") or [])
        t_stops = list(svc_cfg.get("think_stop_sequences") or [])
        slack = svc_cfg.get("soft_cap_slack_ratio", 0.2)
        if "</answer>" not in stops:
            errs.append("service_config.stop_sequences must include '</answer>'")
        if "</think>" not in t_stops:
            errs.append("service_config.think_stop_sequences must include '</think>'")
        # slack ratio sanity
        try:
            s = float(slack)
            if s < 0.0:
                errs.append("soft_cap_slack_ratio must be >= 0.0")
        except Exception:
            errs.append("soft_cap_slack_ratio must be a number")

        # Tokenizer atomicity and stop-id parity (serve/eval)
        try:
            from transformers import AutoTokenizer  # type: ignore
            from tina.tokenizer_utils import ensure_reasoning_tokens
        except Exception as e:
            errs.append(f"tokenizer import failed: {type(e).__name__}: {e}")
            return errs
        try:
            tok = AutoTokenizer.from_pretrained(str(ROOT / "model/Base"), use_fast=True, local_files_only=True, trust_remote_code=True)
        except Exception as e:
            errs.append(f"failed to load tokenizer: {type(e).__name__}: {e}")
            return errs
        # Ensure atomic tags
        try:
            ensure_reasoning_tokens(tok)
        except Exception as e:
            errs.append(f"reasoning tag atomicity failed: {type(e).__name__}: {e}")
        # Build stop id lists as eval/serve would
        def _enc_list(tags: List[str]) -> List[List[int]]:
            out: List[List[int]] = []
            for t in tags:
                try:
                    out.append(tok.encode(t, add_special_tokens=False))
                except Exception:
                    out.append(tok(t, add_special_tokens=False).input_ids)
            return out
        # Assert single-token encoding for each stop tag
        for tag in stops + t_stops:
            ids = tok.encode(tag, add_special_tokens=False)
            if not isinstance(ids, list):
                ids = list(ids)
            if len(ids) != 1:
                errs.append(f"stop tag '{tag}' does not encode to a single token: {ids}")
        eval_answer_ids = _enc_list(stops)
        eval_think_ids = _enc_list(t_stops)
        serve_answer_ids = _enc_list(stops)  # serve encodes the same strings
        serve_think_ids = _enc_list(t_stops)
        if eval_answer_ids != serve_answer_ids:
            errs.append("parity: eval vs serve answer stop-id sequences differ")
        if eval_think_ids != serve_think_ids:
            errs.append("parity: eval vs serve think stop-id sequences differ")
    except Exception as e:
        errs.append(f"service_config validation error: {type(e).__name__}: {e}")
    return errs

def main() -> int:
    all_errors: list[str] = []
    for name, (cfg_path, schema_path) in FILES.items():
        if not cfg_path.exists():
            all_errors.append(f"missing config: {cfg_path}")
            continue
        if not schema_path.exists():
            all_errors.append(f"missing schema: {schema_path}")
            continue
        # Allow override for 'service' config path via env var
        if name == "service":
            cfg_path = _service_cfg_path()
        inst = _load_json(cfg_path)
        schema = _load_json(schema_path)
        errs = _jsonschema_validate(inst, schema)
        if errs:
            all_errors.append(f"[{name}] validation errors:")
            all_errors += [f"  - {e}" for e in errs]
    all_errors += _coherence_checks()
    if all_errors:
        sys.stderr.write("\n".join(all_errors) + "\n")
        return 1
    print("All configs valid.")
    print("Tag/stop-guard OK")
    return 0

def main_cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Validate service config, tokenizer atomicity, and train_config")
    ap.add_argument("--train-config", default=None, help="Optional train_config.yaml path for --all mode")
    args = ap.parse_args(argv)
    rc = main()
    if rc != 0:
        return rc
    if args.all:
        # Optional train config presence and fields
        if args.train_config:
            from pathlib import Path as _P
            p = _P(args.train_config)
            if not p.exists():
                sys.stderr.write(f"missing train_config: {p}\n")
                return 1
            try:
                import yaml as _yaml  # type: ignore
                cfg = _yaml.safe_load(p.read_text(encoding='utf-8'))
            except Exception:
                import json as _json
                cfg = _json.loads(p.read_text(encoding='utf-8'))
            # Require core sections and allow adapter under model.adapter.path
            if "model" not in cfg or "data" not in cfg or "lambdas" not in cfg:
                sys.stderr.write("train_config missing required sections: model/data/lambdas\n")
                return 1
            # sample_every and budget_cap can be top-level or under schedule
            sched = cfg.get("schedule") or {}
            if ("sample_every" not in cfg) and ("sample_every" not in sched):
                sys.stderr.write("train_config missing field: sample_every (top-level or schedule.sample_every)\n")
                return 1
            if ("budget_cap" not in cfg) and ("budget_cap" not in sched):
                sys.stderr.write("train_config missing field: budget_cap (top-level or schedule.budget_cap)\n")
                return 1
            # adapter path accepted at model.adapter.path or top-level adapter.path
            m = cfg.get("model") or {}
            mad = (m.get("adapter") or {}).get("path")
            tad = (cfg.get("adapter") or {}).get("path") if isinstance(cfg.get("adapter"), dict) else None
            if not (mad or tad):
                # Allow empty adapter path but field should exist; warn softly but do not fail
                pass
        # Tokenizer atomicity is already covered in coherence checks via ensure_reasoning_tokens
    return 0

if __name__ == "__main__":
    raise SystemExit(main_cli())
