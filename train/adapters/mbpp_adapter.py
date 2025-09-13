from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator
from train.data_schema import ReasoningSample


class MBPPAdapter:
    name = "mbpp"

    def __init__(self, rows: Iterable[Dict[str, Any]]):
        self.rows = list(rows)

    def __iter__(self) -> Iterator[ReasoningSample]:
        for i, r in enumerate(self.rows):
            prompt = r.get("text") or r.get("prompt") or r.get("question") or r.get("task") or ""
            code = r.get("code_solution") or r.get("code") or r.get("target") or ""
            tests = r.get("test_list") or r.get("tests") or []
            verifier = {
                "type": "pytest",
                "snippet": str("\n".join([str(t) for t in tests])) if tests else None,
            }
            yield ReasoningSample(
                id=str(r.get("id", i)),
                source=self.name,
                domain="code",
                subdomain="mbpp",
                difficulty=str(r.get("difficulty") or ""),
                prompt=str(prompt),
                target=str(code),
                rationale=r.get("explanation") or None,
                verifier=verifier,
                meta={"tags": ["python", "unit_test"]},
            )

