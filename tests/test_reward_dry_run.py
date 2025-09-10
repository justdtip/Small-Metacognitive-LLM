from train.reward import reward_fn

def test_reward_reacts_to_think_length():
    base = {"body": "<think> one two </think> <answer> ok </answer>", "correct": 1.0}
    long = {"body": "<think> " + ("x " * 200) + "</think> <answer> ok </answer>", "correct": 1.0}
    r1 = reward_fn(base, budget_cap=64, alpha=0.01, format_bonus=0.5)
    r2 = reward_fn(long, budget_cap=64, alpha=0.01, format_bonus=0.5)
    assert -1.0 <= r1 <= 2.0 and -1.0 <= r2 <= 2.0
    assert r2 < r1  # longer think yields lower reward due to penalty

