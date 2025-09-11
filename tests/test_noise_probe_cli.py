import json
import subprocess
import sys
from pathlib import Path


def test_noise_probe_cli(tmp_path):
    # Prepare a tiny records JSONL
    recs = [
        {"body": "<think> a b </think><answer>ok</answer>", "gold": "ok", "think_tokens_used": 2},
        {"body": "<think> x </think><answer>no</answer>", "gold": "ok", "think_tokens_used": 1},
    ]
    records_path = tmp_path / "recs.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    out_csv = tmp_path / "probe.csv"
    p = subprocess.run([sys.executable, str(Path("tools/noise_probe.py")), "--temps", "0.0,0.5", "--records", str(records_path), "--out", str(out_csv)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr + p.stdout
    assert out_csv.exists()
    lines = out_csv.read_text(encoding="utf-8").splitlines()
    assert lines, "empty CSV"
    header = lines[0].strip().split(",")
    assert "temperature" in header and "mean_think_tokens" in header and "accuracy" in header

