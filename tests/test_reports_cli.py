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

