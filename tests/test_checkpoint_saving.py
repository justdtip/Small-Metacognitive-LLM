import json
from pathlib import Path
import subprocess, sys, os


def test_periodic_checkpoint_saving(tmp_path):
    # Minimal service config for decode parity (not used when sample_every=0 but required by loader)
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
        json.dumps({"text": "<think> a b </think> <answer> ok </answer>"}) for _ in range(5)
    ]), encoding="utf-8")

    # Train config (YAML-compatible JSON) with checkpointing
    ckpts = tmp_path / "ckpts"
    cfg = {
        "model": {"base": str(tmp_path / "model" / "Base")},  # non-existent to trigger TinyLM fallback
        "data": {"jsonl": str(data)},
        "save_interval": 2,
        "save_dir": str(ckpts),
        # Avoid on-policy decode in test
        "sample_every": 0,
        "budget_cap": 8,
        "lambdas": {"plan": 0.0, "budget": 0.0, "conf": 0.0, "gate": 0.0, "rewrite": 0.0},
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
    env["PYTHONPATH"] = str(Path.cwd())

    p = subprocess.run([sys.executable, str(Path("train/runner.py")), "--train-config", str(cfg_path), "--steps", "5"], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr + p.stdout

    # Check that periodic checkpoints are created
    c2 = ckpts / "checkpoint-2"
    c4 = ckpts / "checkpoint-4"
    assert c2.exists() and c2.is_dir(), f"missing {c2} — stdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"
    assert c4.exists() and c4.is_dir(), f"missing {c4} — stdout:\n{p.stdout}\n\nstderr:\n{p.stderr}"

