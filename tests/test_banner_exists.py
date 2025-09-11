from pathlib import Path


def test_runner_contains_banner():
    p = Path('train/runner.py')
    text = p.read_text(encoding='utf-8')
    # Must include key phrases
    for kw in ("hidden CoT", "on-policy", "budget", "acceptance_criteria"):
        assert kw in text, f"missing banner keyword: {kw}"

