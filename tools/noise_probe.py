#!/usr/bin/env python3
"""
Sweep decode temperature and log mean think_tokens_used and accuracy.

Usage:
  python tools/noise_probe.py --temps 0.0,0.2,0.4,0.6 --records path.jsonl --out reports/noise_probe.csv

Notes:
  - This CLI operates on pre-generated evaluation records (JSONL), not live decoding.
  - Required fields per record:
      body: "<think>..</think><answer>..</answer>"
      gold or correct: reference string for EM scoring, or a correctness flag
      think_tokens_used (optional): numeric count; else computed as word-count in <think> span
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tina.serve import _extract_answer  # reuse canonical extraction


def _think_count_from_body(body: str) -> int:
    i = body.find("<think>")
    j = body.find("</think>", i + len("<think>")) if i != -1 else -1
    if i != -1 and j != -1:
        span = body[i + len("<think>"):j]
        return len([t for t in span.strip().split() if t])
    return 0


def load_records(path: Path):
    recs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs


def compute_metrics(records):
    n = 0
    acc_sum = 0.0
    think_sum = 0.0
    for r in records:
        body = r.get("body") or ""
        gold = (r.get("gold") or "").strip()
        ans = _extract_answer(body, include_think=False)
        if gold:
            acc_sum += 1.0 if ans.strip() == gold else 0.0
            n += 1
        elif "correct" in r:
            acc_sum += float(1.0 if r.get("correct") else 0.0)
            n += 1
        # think tokens
        if r.get("think_tokens_used") is not None:
            try:
                think_sum += float(r.get("think_tokens_used"))
            except Exception:
                think_sum += float(_think_count_from_body(body))
        else:
            think_sum += float(_think_count_from_body(body))
    mean_think = (think_sum / max(1, len(records))) if records else 0.0
    acc = (acc_sum / max(1, n)) if n > 0 else 0.0
    return mean_think, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", required=True, help="Comma-separated temperatures, e.g., 0.0,0.2,0.4")
    ap.add_argument("--records", required=True, help="Path to JSONL eval records")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    temps = []
    for t in (args.temps or "").split(","):
        t = t.strip()
        if not t:
            continue
        try:
            temps.append(float(t))
        except ValueError:
            pass
    records = load_records(Path(args.records))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # Determine if records carry explicit per-record temperatures
    has_temp = any("temperature" in r for r in records)
    rows = []
    for T in temps:
        sub = [r for r in records if (not has_temp) or (abs(float(r.get("temperature", 0.0)) - float(T)) < 1e-9)]
        if not sub:
            continue
        mean_think, acc = compute_metrics(sub)
        rows.append({"temperature": T, "mean_think_tokens": mean_think, "accuracy": acc})
    with outp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["temperature", "mean_think_tokens", "accuracy"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
