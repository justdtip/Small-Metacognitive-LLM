import json
import subprocess
import sys
from pathlib import Path


def test_noise_probe_cli_offline_headers(tmp_path):
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
    # New columns are present
    for col in ("temperature", "n", "mean_think_tokens", "accuracy", "slack_ratio", "stop_sequences"):
        assert col in header


def test_noise_probe_cli_live_two_temps_different_rows(tmp_path):
    out_csv = tmp_path / "probe_live.csv"
    p = subprocess.run([sys.executable, str(Path("tools/noise_probe.py")),
                        "--live", "--temps", "0.0,0.8",
                        "--prompt", "2+2?", "--gold", "4",
                        "--out", str(out_csv)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr + p.stdout
    lines = out_csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 3  # header + at least 2 rows
    # Parse two rows
    header = lines[0].strip().split(",")
    rows = [dict(zip(header, l.split(","))) for l in lines[1:3]]
    # Temperatures differ
    assert rows[0]["temperature"] != rows[1]["temperature"]
    # At least one metric differs (mean_think_tokens or accuracy)
    mt0, mt1 = rows[0]["mean_think_tokens"], rows[1]["mean_think_tokens"]
    acc0, acc1 = rows[0]["accuracy"], rows[1]["accuracy"]
    assert (mt0 != mt1) or (acc0 != acc1)


def test_noise_probe_cli_offline_grouping_requires_field(tmp_path):
    # Records without 'temperature' field
    recs = [
        {"body": "<think> a </think><answer>ok</answer>", "gold": "ok", "think_tokens_used": 1},
        {"body": "<think> x y </think><answer>ok</answer>", "gold": "ok", "think_tokens_used": 2},
    ]
    records_path = tmp_path / "recs2.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    out_csv = tmp_path / "probe_group.csv"
    p = subprocess.run([sys.executable, str(Path("tools/noise_probe.py")),
                        "--group-by-temperature", "--records", str(records_path), "--out", str(out_csv)],
                       capture_output=True, text=True)
    assert p.returncode != 0
    assert "no 'temperature' field" in (p.stderr or "")
