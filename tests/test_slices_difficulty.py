from train.metrics import aggregate, to_csv_json
from train.eval_loop import compute_eval_metrics

def test_slices_difficulty(tmp_path):
    recs = [
        {"body": "<think> a b </think><answer>ok</answer>", "gold": "ok", "difficulty_bin": 0, "plan_src": "heuristic", "budget_src": "heuristic"},
        {"body": "<think> x y z </think><answer>no</answer>", "gold": "ok", "difficulty_bin": 1, "plan_src": "heuristic", "budget_src": "heuristic"},
    ]
    agg = aggregate(recs, compute_eval_metrics)
    out_csv = tmp_path / "metrics.csv"
    to_csv_json(agg, out_json=str(tmp_path/"metrics.json"), out_csv=str(out_csv))
    lines = out_csv.read_text(encoding="utf-8").splitlines()
    assert lines, "empty CSV"
    headers = lines[0].split(",")
    assert "section" in headers
    sec_idx = headers.index("section")
    sections = [l.split(",")[sec_idx] for l in lines[1:] if l.strip()]
    assert any(s.startswith("slice:difficulty_bin=") for s in sections)
