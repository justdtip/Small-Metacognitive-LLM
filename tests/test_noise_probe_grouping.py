import json
import subprocess
import sys
from pathlib import Path


def test_noise_probe_cli_groups_by_temperature(tmp_path):
    # Create records with embedded per-record temperatures
    recs = [
        {"temperature": 0.0, "think_tokens_used": 10, "correct": 1},
        {"temperature": 0.0, "think_tokens_used": 12, "correct": 0},
        {"temperature": 0.2, "think_tokens_used": 8,  "correct": 1},
    ]
    jsonl = tmp_path / "recs.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    out = tmp_path / "out.csv"

    p = subprocess.run([sys.executable, str(Path("tools/noise_probe.py")),
                        "--temps", "0.0,0.2",
                        "--records", str(jsonl),
                        "--out", str(out)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr + p.stdout
    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 3  # header + two rows
    header = lines[0].split(",")
    assert header == ["temperature", "mean_think_tokens", "accuracy"]
    # Parse rows
    rows = [dict(zip(header, r.split(","))) for r in lines[1:] if r.strip()]
    temps = sorted(float(r["temperature"]) for r in rows)
    assert temps == [0.0, 0.2]
    # Check means against inputs: 0.0 -> (10+12)/2=11, acc=(1+0)/2=0.5; 0.2 -> 8, acc=1.0
    row0 = next(r for r in rows if abs(float(r["temperature"]) - 0.0) < 1e-9)
    row2 = next(r for r in rows if abs(float(r["temperature"]) - 0.2) < 1e-9)
    assert abs(float(row0["mean_think_tokens"]) - 11.0) < 1e-6
    assert abs(float(row0["accuracy"]) - 0.5) < 1e-6
    assert abs(float(row2["mean_think_tokens"]) - 8.0) < 1e-6
    assert abs(float(row2["accuracy"]) - 1.0) < 1e-6

