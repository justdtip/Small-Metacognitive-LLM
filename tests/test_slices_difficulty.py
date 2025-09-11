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


def test_slices_consistency_and_difference():
    records = [
        {"correctness": 1, "difficulty_bin": 0, "plan_src": "gold"},
        {"correctness": 0, "difficulty_bin": 1, "plan_src": "gold"},
        {"correctness": 1, "difficulty_bin": 1, "plan_src": "gold"},
    ]
    result = aggregate(records, compute_eval_metrics, slice_keys=("difficulty_bin",))
    # Slice accuracies
    s0 = result["slices"]["difficulty_bin"]["0"]["accuracy"]
    s1 = result["slices"]["difficulty_bin"]["1"]["accuracy"]
    assert s0 != s1
    # Weighted average equals overall
    n0 = sum(1 for r in records if r.get("difficulty_bin") == 0)
    n1 = sum(1 for r in records if r.get("difficulty_bin") == 1)
    overall = result["overall"]["accuracy"]
    wa = (s0 * n0 + s1 * n1) / (n0 + n1)
    assert abs(overall - wa) < 1e-8
