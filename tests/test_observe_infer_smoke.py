import types, json
import torch
import tools.observe_infer as oi


class DummyTok:
    def __init__(self):
        self.v = {"<think>": 100, "</think>": 101, "<answer>": 102, "</answer>": 103}
    def add_special_tokens(self, *a, **k):
        return 0
    def encode(self, s, add_special_tokens=False):
        return [self.v.get(s, 1)]
    def __call__(self, t, return_tensors="pt", add_special_tokens=True):
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
    def convert_tokens_to_ids(self, t):
        return self.v.get(t, 1)
    def decode(self, ids, skip_special_tokens=False):
        return "<think>x</think><answer>y</answer>"


class DummyGenOut:
    def __init__(self, seqs):
        self.sequences = seqs
        self.scores = [torch.randn(1, 200)]
        # hidden_states: tuple of per-step tuples; use one step with 4 layer tensors
        self.hidden_states = (tuple(torch.randn(1, seqs.shape[1], 8) for _ in range(4)),)
        # attentions: tuple of per-step tuples; last has 4 layer attn tensors
        self.attentions = (tuple(torch.rand(1, 2, seqs.shape[1], seqs.shape[1]) for _ in range(4)),)


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=8)
    def to(self, *a, **k):
        return self
    def generate(self, input_ids, **k):
        return DummyGenOut(torch.cat([input_ids, torch.tensor([[103]])], dim=1))
    def forward(self, *a, **k):
        class O:
            pass
        o = O()
        o.hidden_states = [torch.randn(1, 4, 8) for _ in range(4)]
        return o


def test_script_main_monkeypatched(monkeypatch, tmp_path):
    # Monkeypatch model loader to dummy; use real tokenizer from local Base
    monkeypatch.setattr(oi, "AutoModelForCausalLM", types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyModel()))
    # Run main via argv simulation
    import sys
    jsonl = tmp_path / "obs.jsonl"
    argv = ["x", "--model", "model/Base", "--prompt", "hello", "--jsonl", str(jsonl)]
    monkeypatch.setattr(sys, "argv", argv)
    oi.main()
    data = json.loads(jsonl.read_text().splitlines()[0])
    assert "activations" in data and "sections" in data and "text" in data
