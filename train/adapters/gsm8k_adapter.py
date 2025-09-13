from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional
from train.data_schema import ReasoningSample


class GSM8KAdapter:
    name = "gsm8k"

    def __init__(self, rows: Iterable[Dict[str, Any]]):
        self.rows = list(rows)

    def __iter__(self) -> Iterator[ReasoningSample]:
        for i, r in enumerate(self.rows):
            q = r.get("question") or r.get("prompt") or r.get("input") or ""
            a = r.get("answer") or r.get("label") or r.get("target") or ""
            cot = r.get("rationale") or r.get("chain_of_thought") or r.get("cot") or r.get("solution")
            # Try to extract final numeric answer pattern '#### 42'
            num = _parse_numeric(a)
            verifier = {"type": "numeric", "value": num} if num is not None else {"type": "regex", "pattern": str(a).strip()}
            yield ReasoningSample(
                id=str(r.get("id", i)),
                source=self.name,
                domain="math",
                subdomain="arithmetic",
                difficulty=str(r.get("level") or r.get("difficulty") or ""),
                prompt=str(q),
                target=str(a).strip(),
                rationale=str(cot) if cot is not None else None,
                verifier=verifier,
                meta={"tags": ["school_math", "word_problem"]},
            )


def _parse_numeric(text: Any) -> Optional[float]:
    try:
        import re
        s = str(text)
        # common GSM8K final answer marker '#### <num>'
        m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1))
        # fallback: plain numeric
        m = re.search(r"([-+]?\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1))
    except Exception:
        return None
    return None

