import json
import subprocess
import sys
from pathlib import Path


def test_reports_cli(tmp_path):
    # Create tiny eval JSONL
    recs = [
        {"body": "<think> a b </think><answer>ok</answer>", "gold": "ok", "plan_src": "heuristic", "budget_src": "heuristic", "think_tokens_used": 2},
        {"body": "<think> x y z </think><answer>no</answer>", "gold": "ok", "plan_src": "heuristic", "budget_src": "heuristic", "think_tokens_used": 3},
    ]
    records_path = tmp_path / "eval.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "reports"
    p = subprocess.run([sys.executable, str(Path("tools/reports.py")), "--records", str(records_path), "--out-dir", str(out_dir), "--budgets", "16,32"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr + p.stdout

    # Check files exist
    m_json = out_dir / "metrics.json"
    m_csv = out_dir / "metrics.csv"
    q_json = out_dir / "quality_vs_budget.json"
    q_csv = out_dir / "quality_vs_budget.csv"
    for fp in (m_json, m_csv, q_json, q_csv):
        assert fp.exists(), f"missing {fp}"

    # CSV should have 'section' column for metrics export
    header = (m_csv.read_text(encoding="utf-8").splitlines() or [""])[0]
    assert "section" in header, header


def test_slices_nonempty_and_curve_length(tmp_path):
    # Prepare records with two difficulty bins and plan_src values; differing outcomes
    recs = [
        {"body": "<think> t </think><answer> ok </answer>", "gold": "ok", "difficulty_bin": 0, "plan_src": "gold", "think_tokens_used": 1},
        {"body": "<think> t t </think><answer> no </answer>", "gold": "ok", "difficulty_bin": 1, "plan_src": "gold", "think_tokens_used": 2},
        {"body": "<think> t t t </think><answer> ok </answer>", "gold": "ok", "difficulty_bin": 1, "plan_src": "teacher", "think_tokens_used": 3},
    ]
    records_path = tmp_path / "eval2.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    out_dir = tmp_path / "reports2"
    p = subprocess.run([sys.executable, str(Path("tools/reports.py")), "--records", str(records_path), "--out-dir", str(out_dir), "--budgets", "8,16"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr + p.stdout

    # Load outputs
    m_json = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    q_json = json.loads((out_dir / "quality_vs_budget.json").read_text(encoding="utf-8"))

    # Slices should be non-empty
    assert m_json.get("slices"), "slices empty"
    # Curve length must equal number of budgets
    curve = q_json.get("curve") or []
    assert len(curve) == 2
    # At least one metric difference across slices or budgets
    # Compare difficulty slices if present
    diffs = False
    if "difficulty_bin" in (m_json.get("slices") or {}):
        s = m_json["slices"]["difficulty_bin"]
        vals = [v.get("accuracy") for _, v in s.items() if isinstance(v, dict)]
        vals = [x for x in vals if x is not None]
        diffs = (len(set(vals)) > 1)
    # Or check budget accuracies differ
    if not diffs and curve:
        accs = [row.get("accuracy") for row in curve]
        diffs = (len(set(accs)) > 1)
    assert diffs, "no differences across slices or budgets"
