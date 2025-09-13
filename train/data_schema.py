from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReasoningSample:
    """
    Unified sample schema for reasoning datasets.

    Fields
    - id: unique identifier (string)
    - source: dataset/source name (e.g., 'gsm8k', 'math', 'mbpp')
    - domain: high-level domain label (e.g., 'math', 'logic', 'code')
    - subdomain: fine-grained category (e.g., 'arithmetic', 'algebra')
    - difficulty: optional difficulty label/score
    - prompt: input prompt/question
    - target: canonical short answer (string)
    - rationale: optional chain-of-thought/explanation text
    - verifier: dict describing how to verify target (e.g., numeric, regex, pytest snippet)
    - meta: miscellaneous metadata
        - est_steps: estimated reasoning steps (int)
        - est_tokens_in: estimated input tokens (int)
        - est_tokens_out: estimated output tokens (int)
        - length_bin: 'short' | 'mid' | 'long'
        - tags: arbitrary string tags
    """

    id: str
    source: str
    domain: str
    subdomain: Optional[str]
    difficulty: Optional[str]
    prompt: str
    target: str
    rationale: Optional[str]
    verifier: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=lambda: {
        "est_steps": None,
        "est_tokens_in": None,
        "est_tokens_out": None,
        "length_bin": None,
        "tags": [],
    })


def ensure_reasoning_sample(obj: Any) -> ReasoningSample:
    if isinstance(obj, ReasoningSample):
        return obj
    if isinstance(obj, dict):
        return ReasoningSample(
            id=str(obj.get("id") or ""),
            source=str(obj.get("source") or ""),
            domain=str(obj.get("domain") or "unknown"),
            subdomain=obj.get("subdomain"),
            difficulty=obj.get("difficulty"),
            prompt=str(obj.get("prompt") or ""),
            target=str(obj.get("target") or ""),
            rationale=obj.get("rationale"),
            verifier=dict(obj.get("verifier") or {}),
            meta=dict(obj.get("meta") or {}),
        )
    raise TypeError(f"Cannot coerce type {type(obj)} to ReasoningSample")

