#!/usr/bin/env python3
"""
Aggregate evaluation metrics and produce quality-vs-budget curves.

Reads eval JSONL, uses train.metrics.aggregate + train.eval_loop.quality_vs_budget_curve,
and writes reports/{metrics,quality_vs_budget}.{json,csv} into the chosen output dir.

Usage:
  python tools/reports.py --records path/to/eval.jsonl --out-dir reports \
      --budgets 32,64,128,256

Notes:
  - Records should contain per-sample dicts with keys expected by compute_eval_metrics,
    e.g., body, gold/correct, optional plan/budget/conf fields, think_tokens_used, provenance.
  - If 'correct' is not present and 'gold' missing, accuracy will default to 0 for that sample.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.metrics import aggregate, to_csv_json
from train.eval_loop import compute_eval_metrics, quality_vs_budget_curve


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def parse_budgets(arg: str) -> list[int]:
    vals: list[int] = []
    for s in (arg or "").split(","):
        s = s.strip()
        if not s:
            continue
        try:
            vals.append(int(s))
        except ValueError:
            pass
    return vals or [32, 64, 128, 256]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="Path to eval JSONL records")
    ap.add_argument("--out-dir", required=True, help="Output directory for reports")
    ap.add_argument("--budgets", default="32,64,128,256", help="Comma-separated budget caps for the curve")
    args = ap.parse_args()

    recs = load_records(Path(args.records))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate metrics (overall + provenance/difficulty slices)
    agg = aggregate(recs, compute_eval_metrics)
    # Enforce non-empty slices when slice keys are present in records
    present_slice_keys = [
        k for k in ("plan_src", "budget_src", "difficulty_bin") if any(k in r for r in recs)
    ]
    if present_slice_keys:
        has_nonempty = any(bool(agg.get("slices", {}).get(k)) for k in present_slice_keys)
        if not has_nonempty:
            sys.stderr.write("error: expected non-empty slices in metrics but none were produced\n")
            return 1
    to_csv_json(agg, out_json=str(out_dir / "metrics.json"), out_csv=str(out_dir / "metrics.csv"))

    # Quality-vs-budget curve
    budgets = parse_budgets(args.budgets)
    curve = quality_vs_budget_curve(recs, budgets=budgets)
    # Enforce that the curve covers exactly the requested budgets
    if len(curve.get("curve", [])) != len(budgets):
        sys.stderr.write(
            f"error: quality_vs_budget curve length {len(curve.get('curve', []))} != requested budgets {len(budgets)}\n"
        )
        return 1
    # Save both JSON and CSV
    (out_dir / "quality_vs_budget.json").write_text(json.dumps(curve), encoding="utf-8")
    with (out_dir / "quality_vs_budget.csv").open("w", encoding="utf-8") as f:
        f.write("budget,accuracy\n")
        for row in curve.get("curve", []):
            f.write(f"{row.get('budget')},{row.get('accuracy')}\n")

    # Print L_opt (argmax) to stdout for convenience
    try:
        best = max(curve.get("curve", []), key=lambda r: r.get("accuracy", 0.0))
        print(json.dumps({"L_opt": best.get("budget"), "accuracy": best.get("accuracy")}))
    except ValueError:
        print(json.dumps({"L_opt": None, "accuracy": None}))


if __name__ == "__main__":
    main()
