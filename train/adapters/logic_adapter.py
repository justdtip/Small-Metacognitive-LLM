from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator
from train.data_schema import ReasoningSample


class LogicAdapter:
    name = "logic"

    def __init__(self, rows: Iterable[Dict[str, Any]]):
        self.rows = list(rows)

    def __iter__(self) -> Iterator[ReasoningSample]:
        for i, r in enumerate(self.rows):
            prompt = r.get("prompt") or r.get("question") or r.get("text") or ""
            target = r.get("answer") or r.get("target") or ""
            rationale = r.get("rationale") or r.get("explanation") or None
            # default verifier: exact regex on normalized answer
            verifier = {"type": "regex", "pattern": str(target).strip()}
            yield ReasoningSample(
                id=str(r.get("id", i)),
                source=self.name,
                domain="logic",
                subdomain=str(r.get("type") or ""),
                difficulty=str(r.get("difficulty") or ""),
                prompt=str(prompt),
                target=str(target),
                rationale=rationale,
                verifier=verifier,
                meta={"tags": list(r.get("tags") or [])},
            )

