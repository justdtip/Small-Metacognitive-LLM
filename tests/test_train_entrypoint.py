import json
from pathlib import Path
import subprocess, sys, os


def test_end_to_end_config_load(tmp_path):
    # Minimal service config
    svc = {
        "host": "127.0.0.1",
        "port": 8000,
        "input_tokens_cap": 4096,
        "max_new_tokens_cap": 1024,
        "rate_limit_per_min": 60,
        "visible_cot_default": False,
        "stop_sequences": ["</answer>"],
        "think_stop_sequences": ["</think>"],
        "soft_cap_slack_ratio": 0.2,
    }
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "service_config.json").write_text(json.dumps(svc), encoding="utf-8")

    # Tiny dataset
    data = tmp_path / "data.jsonl"
    data.write_text("\n".join([
        json.dumps({"text": "<think> a b </think> <answer> ok </answer>"}),
        json.dumps({"text": "<think> x y </think> <answer> ok </answer>"}),
    ]), encoding="utf-8")

    # Train config (YAML-compatible JSON)
    cfg = {
        "model": {"base": str(tmp_path / "model" / "Base")},
        "adapter": {"path": str(tmp_path / "model" / "Tina" / "checkpoint-2000")},
        "data": {"jsonl": str(data)},
        "sample_every": 1,
        "budget_cap": 8,
        "lambdas": {"plan": 0.5, "budget": 0.1, "conf": 0.1, "gate": 1e-4, "rewrite": 0.0},
    }
    cfg_path = tmp_path / "train_config.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    env = dict(os.environ)
    env.update({
        "CONFIG_ROOT": str(tmp_path),
        "SERVICE_CONFIG_PATH": str(tmp_path / "config" / "service_config.json"),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZER_PARALLELISM": "false",
    })
    # Ensure repo root is importable for subprocess
    env["PYTHONPATH"] = str(Path.cwd())
    p = subprocess.run([sys.executable, str(Path("train/runner.py")), "--train-config", str(cfg_path), "--steps", "2"], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr + p.stdout
    # Find manifest line
    lines = [l for l in p.stdout.strip().splitlines() if '"manifest"' in l]
    assert lines, p.stdout
    rec = json.loads(lines[-1])
    man = rec.get("manifest") or {}
    # Verify presence of sample_every, budget_cap, lambdas, and parity info
    assert man.get("sample_every") == 1
    assert man.get("budget_cap") == 8
    assert isinstance((man.get("lambdas") or {}).get("plan"), float)
    par = man.get("parity") or {}
    assert "</answer>" in (par.get("stop_sequences") or [])
    assert par.get("soft_cap_slack_ratio") == 0.2
