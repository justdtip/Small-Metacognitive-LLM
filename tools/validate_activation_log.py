#!/usr/bin/env python3
import sys, json
from pathlib import Path
from jsonschema import Draft202012Validator


def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/validate_activation_log.py <schema> <jsonl>")
        sys.exit(2)
    schema_path, jsonl_path = map(Path, sys.argv[1:3])
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    val = Draft202012Validator(schema)
    ok = True
    for i, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        errors = sorted(val.iter_errors(obj), key=lambda e: e.path)
        if errors:
            ok = False
            print(f"[FAIL] line {i}: {[e.message for e in errors]}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

