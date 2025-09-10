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
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "generation": (ROOT / "model/Base/generation_config.json", ROOT / "schemas/generation_config.schema.json"),
    "adapter": (ROOT / "model/Tina/checkpoint-2000/adapter_config.json", ROOT / "schemas/adapter_config.schema.json"),
    "service": (ROOT / "config/service_config.json", ROOT / "schemas/service_config.schema.json"),
}

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
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

