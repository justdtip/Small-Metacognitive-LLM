import json
import pytest

from train.eval_loop import decode_with_budget


def test_uses_service_config_tags_and_slack(tmp_path, monkeypatch):
    # Write a custom service_config with non-default tags and slack ratio
    cfg_path = tmp_path / "service_config.json"
    cfg = {
        "visible_cot_default": False,
        "stop_sequences": ["</XANS>"],
        "think_stop_sequences": ["</XTHINK>"],
        "soft_cap_slack_ratio": 0.0,
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # Monkeypatch loader to read from tmp config
    import train.eval_loop as ev
    def _load_svc(_root=None):
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(ev, "load_service_config", _load_svc)

    # Dummy tokenizer mapping the custom tags
    class Tok:
        pad_token_id = 0
        eos_token_id = None
        def encode(self, s, add_special_tokens=False):
            if s == "</XTHINK>":
                return [7]
            if s == "</XANS>":
                return [9]
            return [1]
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            class R: pass
            import torch
            r = R(); r.input_ids = torch.tensor([[1,2,3]], dtype=torch.long)
            return r
        def decode(self, ids, skip_special_tokens=False):
            return ""  # not used in assertions

    # Dummy model that respects StoppingCriteria
    class Model:
        def __init__(self):
            import torch
            self._device = torch.device('cpu')
        @property
        def device(self):
            return self._device
        def generate(self, *, input_ids, attention_mask, max_new_tokens, stopping_criteria, **kw):
            import torch
            seq = input_ids.clone()
            # Append tokens until criteria fires
            for _ in range(max_new_tokens):
                # append a dummy token id 5
                seq = torch.cat([seq, torch.tensor([[5]], dtype=torch.long)], dim=1)
                stop = False
                for crit in stopping_criteria:
                    if crit(seq, None):
                        stop = True
                        break
                if stop:
                    break
            class O:
                pass
            o = O()
            o.sequences = seq
            return o

    tok = Tok(); model = Model()
    import torch
    inp = torch.tensor([[1,2,3]], dtype=torch.long)
    out0 = decode_with_budget(tok, model, inp, think_budget=8, max_new_tokens=32, temperature=0.0, visible_cot=False)
    used0 = out0.get("think_tokens_used")

    # Increase slack ratio and ensure more tokens are allowed
    cfg["soft_cap_slack_ratio"] = 0.5
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    out1 = decode_with_budget(tok, model, inp, think_budget=8, max_new_tokens=32, temperature=0.0, visible_cot=False)
    used1 = out1.get("think_tokens_used")

    assert used1 > used0  # larger slack should allow more think tokens
