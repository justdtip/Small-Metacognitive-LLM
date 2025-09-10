#!/usr/bin/env python3
"""
Validate JSONL produced by tools/observe_infer.py against schemas/observe_infer.schema.json.

Usage:
  python tools/validate_observation.py --jsonl logs/obs.jsonl
"""
import argparse, json, sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas/observe_infer.schema.json"


def load_schema():
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    args = ap.parse_args()
    schema = load_schema()
    try:
        import jsonschema  # type: ignore
        from jsonschema import Draft202012Validator
        validator = Draft202012Validator(schema)
        errs = 0
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    sys.stderr.write(f"line {lineno}: invalid JSON: {e}\n")
                    errs += 1
                    continue
                for err in validator.iter_errors(rec):
                    sys.stderr.write(f"line {lineno}: {'.'.join(map(str, err.path))}: {err.message}\n")
                    errs += 1
        if errs:
            sys.stderr.write(f"Found {errs} validation error(s).\n")
            return 1
        print("OK: all records valid.")
        return 0
    except Exception:
        # Minimal structural check if jsonschema not available
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                rec = json.loads(line)
                for key in ("request_id", "ts_ms", "model", "device", "decode", "timing", "io", "text"):
                    if key not in rec:
                        sys.stderr.write(f"line {lineno}: missing key '{key}'\n")
                        return 1
        print("OK: basic checks passed (jsonschema not installed).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

