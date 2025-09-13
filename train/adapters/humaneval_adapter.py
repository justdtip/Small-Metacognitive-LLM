from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator
from train.data_schema import ReasoningSample


class HumanEvalAdapter:
    name = "humaneval"

    def __init__(self, rows: Iterable[Dict[str, Any]]):
        self.rows = list(rows)

    def __iter__(self) -> Iterator[ReasoningSample]:
        for i, r in enumerate(self.rows):
            prompt = r.get("prompt") or r.get("question") or r.get("text") or ""
            code = r.get("canonical_solution") or r.get("solution") or r.get("target") or ""
            tests = r.get("tests") or r.get("unit_tests") or r.get("check") or ""
            verifier = {
                "type": "pytest",
                "snippet": str(tests) if tests else None,
            }
            yield ReasoningSample(
                id=str(r.get("task_id", r.get("id", i))),
                source=self.name,
                domain="code",
                subdomain="humaneval",
                difficulty=str(r.get("difficulty") or ""),
                prompt=str(prompt),
                target=str(code),
                rationale=r.get("explanation") or None,
                verifier=verifier,
                meta={"tags": ["python", "unit_test"]},
            )

