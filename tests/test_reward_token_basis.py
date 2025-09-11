import pytest

from train.reward import reward_fn


class RLETok:
    """Tokenizer that collapses consecutive identical non-space chars into a single token (run-length encoding)."""
    def encode(self, text, add_special_tokens=False):
        toks = []
        prev = None
        for c in (ch for ch in text if ch.strip()):
            if c != prev:
                toks.append(ord(c))
                prev = c
        return toks


def test_penalizes_tokens_not_words():
    tok = RLETok()
    # Many-token think span vs few-token think span (same character repeated vs alternating characters)
    many = {"body": "<think> a b c d e f g h i j </think><answer> ok </answer>", "correct": 1.0}
    few = {"body": "<think> aaaaaaaaaa </think><answer> ok </answer>", "correct": 1.0}
    # With tokenizer (character-level), 'many' has many more tokens → higher penalty at same budget
    r_many = reward_fn(many, budget_cap=8, alpha=0.01, format_bonus=0.0, tokenizer=tok)
    r_few = reward_fn(few, budget_cap=8, alpha=0.01, format_bonus=0.0, tokenizer=tok)
    assert r_many < r_few
