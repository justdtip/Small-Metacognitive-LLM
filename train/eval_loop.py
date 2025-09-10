from __future__ import annotations
from typing import List
from tina.serve import _extract_answer, StopOnTags

def eval_extract_answers(texts: List[str]) -> List[str]:
    """Eval-only helper: run extraction parity on provided bodies (think+answer strings)."""
    return [_extract_answer(t, include_think=False) for t in texts]

