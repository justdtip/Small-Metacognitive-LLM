from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional
from train.data_schema import ReasoningSample


class MATHAdapter:
    name = "math"

    def __init__(self, rows: Iterable[Dict[str, Any]]):
        self.rows = list(rows)

    def __iter__(self) -> Iterator[ReasoningSample]:
        for i, r in enumerate(self.rows):
            q = r.get("problem") or r.get("question") or r.get("prompt") or ""
            a = r.get("solution") or r.get("answer") or r.get("target") or ""
            cot = r.get("rationale") or r.get("derivation") or r.get("proof") or None
            num = _parse_numeric(a)
            verifier = {"type": "numeric", "value": num} if num is not None else {"type": "regex", "pattern": str(a).strip()}
            yield ReasoningSample(
                id=str(r.get("id", i)),
                source=self.name,
                domain="math",
                subdomain=str(r.get("subject") or ""),
                difficulty=str(r.get("level") or r.get("difficulty") or ""),
                prompt=str(q),
                target=str(a).strip(),
                rationale=str(cot) if cot is not None else None,
                verifier=verifier,
                meta={"tags": ["olympiad" if r.get("level") == "hard" else ""]},
            )


def _parse_numeric(text: Any) -> Optional[float]:
    try:
        import re
        s = str(text)
        m = re.search(r"([-+]?\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1))
    except Exception:
        return None
    return None

