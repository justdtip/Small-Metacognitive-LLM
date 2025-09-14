import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(tmp_path: Path, n: int, supervised: bool = False) -> Path:
    body_ok = "<think> plan step </think><answer> ok </answer>"
    body_no = "<think> foo bar </think><answer> no </answer>"
    recs = []
    for i in range(n):
        r = {"text": body_ok if (i % 2 == 0) else body_no}
        if supervised:
            r["plan_class"] = int(i % 3)
            r["target_budget"] = int(8 + (i % 5))
            r["correct"] = int(i % 2 == 0)
        recs.append(r)
    p = tmp_path / "toy.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return p


def _run_one_cycle(data_path: Path):
    p = subprocess.run(
        [
            sys.executable,
            str(Path("train/one_cycle.py")),
            "--data",
            str(data_path),
            "--batch-size",
            "8",
            "--allow-missing-base",
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr + p.stdout
    # Script prints exactly one JSON line
    line = p.stdout.strip().splitlines()[-1]
    return json.loads(line)


def test_json_log_completeness(tmp_path):
    data = _write_jsonl(tmp_path, n=8, supervised=False)
    rec = _run_one_cycle(data)
    # Required keys present
    for k in [
        "loss_answer_ce", "loss_plan_ce", "loss_budget_reg", "loss_conf_cal", "loss_gate_reg", "loss_aux_loss",
        "budget_pred_mean", "think_tokens_used", "parity_digest", "stop_sequences", "slack_ratio", "leakage_detected",
    ]:
        assert k in rec
    # Parity fields sanity
    pd = rec.get("parity_digest") or {}
    assert isinstance(pd.get("visible_cot"), bool) or isinstance(pd.get("visible_cot_default"), (bool, type(None)))
    assert "</answer>" in (rec.get("stop_sequences") or [])
    assert rec.get("leakage_detected") is False


def test_nonzero_aux_losses(tmp_path):
    data = _write_jsonl(tmp_path, n=12, supervised=True)
    rec = _run_one_cycle(data)
    assert (rec.get("loss_plan_ce", 0.0) > 0.0) or (rec.get("loss_budget_reg", 0.0) > 0.0) or (rec.get("loss_conf_cal", 0.0) > 0.0)
