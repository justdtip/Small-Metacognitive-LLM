import pytest

from tina.serve import IntrospectiveEngine, EngineConfig


def test_hidden_mode_strips_think(monkeypatch):
    import torch

    class DummyTok:
        pad_token_id = 0
        eos_token_id = None
        def __init__(self):
            self.vocab = {}
            self.additional_special_tokens = []
            self.unique_no_split_tokens = []
            self._n = 1
        def add_special_tokens(self, d):
            toks = d.get("additional_special_tokens", [])
            for t in toks:
                if t not in self.vocab:
                    self.vocab[t] = self._n; self._n += 1
                if t not in self.additional_special_tokens:
                    self.additional_special_tokens.append(t)
        def convert_tokens_to_ids(self, t):
            return self.vocab.get(t, 0)
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"):
            return torch.tensor([[1,2,3]], dtype=torch.long)
        def __call__(self, s, add_special_tokens=False, return_tensors=None):
            class R: pass
            r = R(); r.input_ids = torch.tensor([[4]], dtype=torch.long)
            return r
        def encode(self, s, add_special_tokens=False):
            return [5]
        def decode(self, ids, skip_special_tokens=False):
            # First call (THINK) returns closing tag, second includes leakage inside answer span
            if not hasattr(self, "_dec_calls"):
                self._dec_calls = 0
            self._dec_calls += 1
            if self._dec_calls == 1:
                return "</think>\n"
            return "<answer> final <think>leak</think> </answer>"

    class DummyModel:
        device = 'cpu'
        dtype = None
        def parameters(self):
            if False:
                yield None
            return iter(())
        def __call__(self, *a, **k):
            class O:
                pass
            import torch
            o = O()
            o.hidden_states = [torch.zeros(1, 2, 4)]
            return o
        def generate(self, **k):
            class O:
                pass
            import torch
            out = O()
            seq = k.get('input_ids')
            out.sequences = torch.cat([seq, torch.tensor([[7]], dtype=torch.long)], dim=1)
            return out

    eng = IntrospectiveEngine(model=DummyModel(), tokenizer=DummyTok(), cfg=EngineConfig(visible_cot=False), hidden_size=8, num_layers=0)
    text = eng.generate_cot([{"role": "user", "content": "Hi"}], max_new_tokens=4, temperature=0.0, top_p=1.0, repetition_penalty=1.0, ignore_eos=False, stream=False)
    # No reasoning tags should remain
    assert "<think>" not in text and "</think>" not in text
    assert "<answer>" not in text and "</answer>" not in text
    # Leakage counter should be set
    assert eng.last_stats.get("leakage") == 1
