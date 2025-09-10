from __future__ import annotations
from typing import Optional, Callable, Dict, Any

def budget_aware_reward(think_len: int, answer_quality: float, budget_cap: int, alpha: float = 0.1) -> float:
    """
    Simple reward: quality minus penalty for exceeding budget.
    answer_quality in [0,1]; penalty grows linearly beyond cap.
    """
    penalty = 0.0
    if think_len > budget_cap:
        penalty = alpha * (think_len - budget_cap)
    return float(max(0.0, answer_quality - penalty))

def _extract_span(body: str, open_t: str, close_t: str) -> str:
    i = body.find(open_t)
    j = body.find(close_t, i + len(open_t)) if i != -1 else -1
    if i != -1 and j != -1:
        return body[i + len(open_t):j]
    return ""

def _format_ok(body: str) -> bool:
    # Must contain exactly one <answer>...</answer>; hidden think is allowed in body but will be ignored.
    return (body.count("<answer>") >= 1 and body.count("</answer>") >= 1)

def reward_fn(sample: Dict[str, Any], *, budget_cap: int = 256, alpha: float = 0.01,
              format_bonus: float = 0.5, grader: Optional[Callable[[Dict[str, Any]], float]] = None,
              tokenizer=None) -> float:
    """
    Compute budget-aware preference/RL reward.
    reward = correctness (0..1) + format_bonus*is_well_formed - alpha*max(0, think_len - budget_cap)
    Clamped to [-1, +2]. If 'grader' provided, it should return correctness in [0,1].
    The sample dict should include at least {'body': '<think>..</think><answer>..</answer>'}.
    """
    body = sample.get("body") or ""
    # determine think length in tokens or words
    think_text = _extract_span(body, "<think>", "</think>")
    if tokenizer is not None:
        try:
            think_len = len(tokenizer.encode(think_text, add_special_tokens=False))
        except Exception:
            think_len = len((tokenizer(think_text, add_special_tokens=False).input_ids))
    else:
        think_len = len([w for w in think_text.strip().split() if w])

    # correctness from grader or sample
    if grader is not None:
        correctness = float(max(0.0, min(1.0, grader(sample))))
    else:
        correctness = float(sample.get("correct", 0.0))
        if not (0.0 <= correctness <= 1.0):
            correctness = 0.0

    fmt_ok = 1.0 if _format_ok(body) else 0.0
    penalty = alpha * max(0, think_len - int(budget_cap))
    reward = correctness + format_bonus * fmt_ok - penalty
    # clamp to [-1, +2]
    if reward < -1.0:
        reward = -1.0
    if reward > 2.0:
        reward = 2.0
    return float(reward)

def dry_run_mocked() -> Dict[str, float]:
    """Offline dry-run demonstrating reward sensitivity to think length.
    Returns a dict with rewards for short/long think spans.
    """
    base = {
        "body": "<think> step1 step2 </think> <answer> ok </answer>",
        "correct": 1.0,
    }
    long = {
        "body": "<think> " + ("step " * 500) + "</think> <answer> ok </answer>",
        "correct": 1.0,
    }
    r_short = reward_fn(base, budget_cap=64, alpha=0.01, format_bonus=0.5)
    r_long = reward_fn(long, budget_cap=64, alpha=0.01, format_bonus=0.5)
    return {"short": r_short, "long": r_long}
