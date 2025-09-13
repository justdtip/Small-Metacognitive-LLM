from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator
from train.data_schema import ReasoningSample


class GlaiveAdapter:
    name = "glaive"

    def __init__(self, rows: Iterable[Dict[str, Any]]):
        self.rows = list(rows)

    def __iter__(self) -> Iterator[ReasoningSample]:
        for i, r in enumerate(self.rows):
            prompt = r.get("prompt") or r.get("question") or r.get("input") or r.get("text") or ""
            answer = r.get("answer") or r.get("target") or r.get("output") or ""
            cot = r.get("rationale") or r.get("explanation") or None
            yield ReasoningSample(
                id=str(r.get("id", i)),
                source=self.name,
                domain=str(r.get("domain") or "mixed"),
                subdomain=str(r.get("subdomain") or ""),
                difficulty=str(r.get("difficulty") or ""),
                prompt=str(prompt),
                target=str(answer),
                rationale=cot,
                verifier={"type": "regex", "pattern": str(answer).strip()},
                meta={"tags": list(r.get("tags") or [])},
            )

