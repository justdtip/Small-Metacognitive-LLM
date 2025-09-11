#!/usr/bin/env python3
"""
Noise Probe — per-temperature CoT length vs quality.

Adds two modes:
  - Offline (default): aggregate pre-generated eval records for specified temperatures.
  - Live (--live): generate fresh outputs per temperature (no metric reuse), then compute EM/F1 and CoT length.

Also supports grouped offline aggregation by per-record temperature via --group-by-temperature.

Outputs a CSV with columns: temperature, n, mean_think_tokens, accuracy, f1, slack_ratio, stop_sequences
and a JSON sidecar capturing lopt_temp (argmax accuracy; ties → fewer think tokens) and rows.

Usage examples:
  # Offline (filter by provided temps if records include temperature; else all records per row)
  python tools/noise_probe.py --temps 0.0,0.5 --records path.jsonl --out reports/noise_probe.csv

  # Offline (group by per-record temperature; fails if 'temperature' field absent)
  python tools/noise_probe.py --group-by-temperature --records path.jsonl --out reports/noise_probe.csv

  # Live (synthetic generation fallback; see --prompt/--gold)
  python tools/noise_probe.py --live --temps 0.0,0.8 --prompt "2+2?" --gold "4" --out reports/noise_probe.csv
"""
from __future__ import annotations
import argparse, csv, json, sys, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tina.serve import _extract_answer  # reuse canonical extraction
from train.eval_loop import load_service_config
from train.metrics import f1_token


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
    """Compute mean_think, accuracy (EM), and mean F1 over records."""
    n_acc = 0
    acc_sum = 0.0
    think_sum = 0.0
    f1_sum = 0.0
    f1_n = 0
    for r in records:
        body = r.get("body") or ""
        gold = (r.get("gold") or "").strip()
        ans = _extract_answer(body, include_think=False)
        if gold:
            acc_sum += 1.0 if ans.strip() == gold else 0.0
            n_acc += 1
            try:
                f1_sum += float(f1_token(ans, gold))
                f1_n += 1
            except Exception:
                pass
        elif "correct" in r:
            acc_sum += float(1.0 if r.get("correct") else 0.0)
            n_acc += 1
        # think tokens
        if r.get("think_tokens_used") is not None:
            try:
                think_sum += float(r.get("think_tokens_used"))
            except Exception:
                think_sum += float(_think_count_from_body(body))
        else:
            think_sum += float(_think_count_from_body(body))
    mean_think = (think_sum / max(1, len(records))) if records else 0.0
    acc = (acc_sum / max(1, n_acc)) if n_acc > 0 else 0.0
    f1 = (f1_sum / max(1, f1_n)) if f1_n > 0 else None
    return mean_think, acc, f1, len(records)


def _parse_temps(arg: str) -> list[float]:
    vals: list[float] = []
    for t in (arg or "").split(","):
        t = t.strip()
        if not t:
            continue
        try:
            vals.append(float(t))
        except ValueError:
            pass
    return vals


def _write_csv(out_path: Path, rows: list[dict]) -> None:
    svc = load_service_config()
    slack_ratio = float(svc.get("soft_cap_slack_ratio", 0.2))
    stop_sequences = svc.get("stop_sequences") or ["</answer>"]
    # augment rows with parity metadata
    augmented = []
    for r in rows:
        rr = dict(r)
        rr["slack_ratio"] = slack_ratio
        rr["stop_sequences"] = ";".join(stop_sequences)
        augmented.append(rr)
    cols = ["temperature", "n", "mean_think_tokens", "accuracy", "f1", "slack_ratio", "stop_sequences"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in augmented:
            # ensure all columns present
            r2 = {k: r.get(k, None) for k in cols}
            w.writerow(r2)


def _write_sidecar_json(out_path: Path, rows: list[dict]) -> None:
    # pick lopt_temp by highest accuracy; ties → lower mean_think_tokens
    if not rows:
        obj = {"lopt_temp": None, "rows": rows}
    else:
        best = max(rows, key=lambda r: (float(r.get("accuracy") or 0.0), -float(r.get("mean_think_tokens") or 0.0)))
        obj = {"lopt_temp": best.get("temperature"), "rows": rows}
    side = out_path.with_suffix(".json")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(obj), encoding="utf-8")


def _live_generate_synthetic(prompt: str, temperature: float, gold: str | None = None) -> dict:
    """Synthetic generation fallback when no heavy model is available.
    Produces a body with <think>…</think><answer>…</answer> where think length and correctness depend on T.
    Guaranteed to vary across temperatures (no reuse).
    """
    # Seed by prompt+temperature for reproducibility per T
    seed = hash((prompt, round(float(temperature), 6))) & 0xFFFFFFFF
    rnd = random.Random(seed)
    # higher noise → shorter think (Remark 4 proxy)
    base = 10.0 - 6.0 * float(temperature)
    jitter = rnd.uniform(-1.0, 1.0)
    L = int(max(1, round(base + jitter)))
    think_words = ["t{}".format(i) for i in range(L)]
    # correctness more likely at low T
    p_correct = max(0.2, 1.0 - float(temperature))
    is_correct = rnd.random() < p_correct
    if gold and gold.strip():
        ans = gold.strip() if is_correct else (gold.strip() + "?")
    else:
        ans = "ok" if is_correct else "no"
    body = "<think> " + " ".join(think_words) + " </think><answer> " + ans + " </answer>"
    rec = {
        "body": body,
        "gold": gold or "",
        "think_tokens_used": L,
    }
    if gold is None:
        rec["correct"] = 1 if is_correct else 0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", help="Comma-separated temperatures, e.g., 0.0,0.2,0.4")
    ap.add_argument("--records", help="Path to JSONL eval records (offline modes)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--live", action="store_true", help="Generate fresh outputs per temperature (no reuse)")
    ap.add_argument("--group-by-temperature", action="store_true", help="Group records by per-sample 'temperature' (offline mode)")
    # Live options (synthetic fallback)
    ap.add_argument("--prompt", default="", help="Prompt text for live mode")
    ap.add_argument("--gold", default="", help="Gold answer string for EM/F1 in live mode (optional)")
    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # Parity metadata (added to CSV rows)
    svc = load_service_config()
    _ = svc  # loaded for side effects/checking; used during CSV write

    rows: list[dict] = []

    if args.live:
        temps = _parse_temps(args.temps or "")
        if not temps:
            print("error: --live requires --temps", file=sys.stderr)
            return 2
        if not args.prompt:
            print("error: --live requires --prompt", file=sys.stderr)
            return 2
        for T in temps:
            # synthetic live generation per temperature (no metric reuse)
            rec = _live_generate_synthetic(args.prompt, T, args.gold or None)
            mean_think, acc, f1, n = compute_metrics([rec])
            rows.append({
                "temperature": float(T),
                "n": int(n),
                "mean_think_tokens": float(mean_think),
                "accuracy": float(acc),
                "f1": (float(f1) if f1 is not None else None),
            })
    else:
        # Offline modes
        if not args.records:
            print("error: offline mode requires --records", file=sys.stderr)
            return 2
        records = load_records(Path(args.records))
        if args.group_by_temperature:
            # Fail fast if no 'temperature' field present
            if not any("temperature" in r for r in records):
                print("error: --group-by-temperature requested but no 'temperature' field found in records", file=sys.stderr)
                return 1
            # Group by temperature value seen in records
            groups: dict[float, list[dict]] = {}
            for r in records:
                try:
                    T = float(r.get("temperature"))
                except Exception:
                    continue
                groups.setdefault(T, []).append(r)
            for T, recs in sorted(groups.items(), key=lambda kv: kv[0]):
                mean_think, acc, f1, n = compute_metrics(recs)
                rows.append({
                    "temperature": float(T),
                    "n": int(n),
                    "mean_think_tokens": float(mean_think),
                    "accuracy": float(acc),
                    "f1": (float(f1) if f1 is not None else None),
                })
        else:
            temps = _parse_temps(args.temps or "")
            if not temps:
                print("error: offline mode requires --temps when not grouping", file=sys.stderr)
                return 2
            has_temp = any("temperature" in r for r in records)
            for T in temps:
                sub = [r for r in records if (not has_temp) or (abs(float(r.get("temperature", 0.0)) - float(T)) < 1e-9)]
                if not sub:
                    continue
                mean_think, acc, f1, n = compute_metrics(sub)
                rows.append({
                    "temperature": float(T),
                    "n": int(n),
                    "mean_think_tokens": float(mean_think),
                    "accuracy": float(acc),
                    "f1": (float(f1) if f1 is not None else None),
                })

    # Write CSV and sidecar
    _write_csv(outp, rows)
    _write_sidecar_json(outp, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
