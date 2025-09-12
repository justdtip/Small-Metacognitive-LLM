import json
import types
import pytest
import torch

from pathlib import Path
from train.eval_loop import decode_with_budget


class _FakeOut:
    def __init__(self, seq):
        self.sequences = seq


class _FakeModel:
    def __init__(self, extra: int):
        self._extra = int(extra)
        # minimal param to provide device
        self.weight = torch.nn.Parameter(torch.zeros(1))
    def parameters(self):
        return iter([self.weight])
    def generate(self, input_ids=None, attention_mask=None, **kw):
        # Return prompt + fixed number of new tokens
        B, T = input_ids.shape
        add = torch.arange(1, self._extra + 1, dtype=torch.long).unsqueeze(0)
        seq = torch.cat([input_ids, add], dim=1)
        return _FakeOut(seq)


def test_applies_config_tags_and_slack_ratio(tmp_path, monkeypatch):
    # Service config override with custom tags and slack ratio
    svc = {
        "host": "127.0.0.1",
        "port": 8000,
        "input_tokens_cap": 4096,
        "max_new_tokens_cap": 1024,
        "rate_limit_per_min": 60,
        "visible_cot_default": False,
        "stop_sequences": ["</ANS>"] ,
        "think_stop_sequences": ["</THK>"],
        "soft_cap_slack_ratio": 0.10,
    }
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "service_config.json").write_text(json.dumps(svc), encoding="utf-8")

    # Spy on StopOnTags/SlackStop construction
    import train.eval_loop as el
    seen = {"think_tags": None, "ans_tags": None, "ratio": None}

    class _StopSpy:
        def __init__(self, tok, tags, max_new=None):
            # record tags
            if seen["think_tags"] is None:
                seen["think_tags"] = tuple(tags)
            else:
                seen["ans_tags"] = tuple(tags)
        def __call__(self, *a, **k):
            return False

    class _SlackSpy:
        def __init__(self, *, base_len, budget, slack_ratio):
            seen["ratio"] = float(slack_ratio)
        def __call__(self, *a, **k):
            return False

    monkeypatch.setattr(el, "StopOnTags", _StopSpy)
    monkeypatch.setattr(el, "SlackStop", _SlackSpy)

    # Load real tokenizer to exercise encoding
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(Path('model/Base')), use_fast=True, local_files_only=True)
    # Build prompt and fake model sized to match slack ratio limit (budget=10 → extra=budget*(1+ratio))
    prompt_ids = tok("Hello", add_special_tokens=False, return_tensors="pt").input_ids
    budget = 10
    extra = int(budget * (1.0 + svc["soft_cap_slack_ratio"]))
    model = _FakeModel(extra=extra)

    # Ensure eval loop reads the temp config
    monkeypatch.setenv("CONFIG_ROOT", str(tmp_path))
    monkeypatch.setenv("SERVICE_CONFIG_PATH", str(cfg_dir / "service_config.json"))

    out = decode_with_budget(tok, model, prompt_ids, think_budget=budget, max_new_tokens=extra + 5, temperature=0.0, visible_cot=True)
    # think_tokens_used should reflect configured slack ratio
    assert int(out.get("think_tokens_used")) == extra
    # StopOnTags received exactly the configured tags
    assert seen["think_tags"] == tuple(svc["think_stop_sequences"]) and seen["ans_tags"] == tuple(svc["stop_sequences"]) 
    # SlackStop used configured ratio
    assert abs(seen["ratio"] - svc["soft_cap_slack_ratio"]) < 1e-6

