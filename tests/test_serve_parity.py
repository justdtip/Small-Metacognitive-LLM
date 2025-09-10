import pytest
from tina.serve import StopOnTags

def test_hidden_cot_removed_and_stops_on_answer():
    # Extraction check
    from tina.serve import _extract_answer
    think = "<think>reasoning</think>"
    answer = "<answer>final</answer>"
    out = _extract_answer(think + answer, include_think=False)
    assert out == "final"
    assert "<think>" not in out and "</think>" not in out
    assert "<answer>" not in out and "</answer>" not in out

    # StoppingCriteria check using a dummy tokenizer
    class DummyTok:
        def encode(self, s, add_special_tokens=False):
            return [1,2] if s == "</answer>" else [3]
    tok = DummyTok()
    sc = StopOnTags(tok, ("</answer>",))
    torch = pytest.importorskip("torch")
    # sequence ending without stop ids
    ids = torch.tensor([[9,9,9]], dtype=torch.long)
    assert sc(ids, None) is False
    # sequence ending with stop ids [1,2]
    ids2 = torch.tensor([[5, 1, 2]], dtype=torch.long)
    assert sc(ids2, None) is True

