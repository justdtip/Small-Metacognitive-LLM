import json
import subprocess
import sys
from pathlib import Path


def _run_check(tmp_root: Path, svc: dict) -> subprocess.CompletedProcess:
    # stage a minimal config tree under tmp_root/config
    cfg_dir = tmp_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "service_config.json").write_text(json.dumps(svc), encoding="utf-8")
    # Run the eval_loop validator against this root
    return subprocess.run(
        [sys.executable, str(Path("train/eval_loop.py")), "--check-config", "--root", str(tmp_root)],
        capture_output=True,
        text=True,
    )


def test_stop_ids_and_slack_ratio_ok(tmp_path):
    svc = {
        "host": "127.0.0.1",
        "port": 8000,
        "input_tokens_cap": 4096,
        "max_new_tokens_cap": 1024,
        "rate_limit_per_min": 60,
        "visible_cot_default": False,
        "stop_sequences": ["</answer>"],
        "think_stop_sequences": ["</think>"],
        "soft_cap_slack_ratio": 0.15,
    }
    p = _run_check(tmp_path, svc)
    assert p.returncode == 0, p.stderr + p.stdout


def test_stop_ids_missing_tags_fail(tmp_path):
    # Missing </answer> should fail
    bad = {
        "host": "127.0.0.1",
        "port": 8000,
        "input_tokens_cap": 4096,
        "max_new_tokens_cap": 1024,
        "rate_limit_per_min": 60,
        "visible_cot_default": False,
        "stop_sequences": [],
        "think_stop_sequences": ["</think>"],
        "soft_cap_slack_ratio": 0.2,
    }
    p = _run_check(tmp_path, bad)
    assert p.returncode != 0
