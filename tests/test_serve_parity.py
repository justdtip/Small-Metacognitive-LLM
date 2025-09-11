import pytest
from tina.serve import StopOnTags, SlackStop

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


def test_slack_stop_soft_cap_and_close_tag_precedence():
    class DummyTok:
        def encode(self, s, add_special_tokens=False):
            # map </think> to [7,8]
            return [7, 8] if s == "</think>" else [9]

    tok = DummyTok()
    torch = pytest.importorskip("torch")

    base_len = 5
    budget = 10
    slack = 0.2  # allow up to floor(10*1.2)=12 before stopping
    soft = SlackStop(base_len=base_len, budget=budget, slack_ratio=slack)
    stop_tag = StopOnTags(tok, ("</think>",))

    # Under soft cap: should not stop
    ids = torch.tensor([[0] * (base_len + 12)], dtype=torch.long)
    assert soft(ids, None) is False

    # Exceed soft cap by one: should stop
    ids_exceed = torch.tensor([[0] * (base_len + 13)], dtype=torch.long)
    assert soft(ids_exceed, None) is True

    # If closing tag appears, StopOnTags should stop regardless of slack
    seq = [0] * (base_len + 10) + [7, 8]
    ids_close = torch.tensor([seq], dtype=torch.long)
    assert stop_tag(ids_close, None) is True
