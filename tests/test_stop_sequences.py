from tina.tokenizer_utils import STOP_SEQUENCES

def test_stop_sequences_contains_close_answer():
    assert "</answer>" in STOP_SEQUENCES

