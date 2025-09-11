from __future__ import annotations
from typing import List, Dict, Any, Iterable, Callable, Optional
from pathlib import Path
import json
import torch
from tina.serve import _extract_answer, StopOnTags
from train.metrics import temperature_fit, ece_binary


def eval_extract_answers(texts: List[str]) -> List[str]:
    """Eval-only helper: run extraction parity on provided bodies (think+answer strings)."""
    return [_extract_answer(t, include_think=False) for t in texts]


def fit_confidence_temperature_and_save(conf_logits: torch.Tensor, labels: torch.Tensor, out_path: str) -> Dict[str, Any]:
    """
    Fit a scalar confidence temperature on held-out logits and write a calibration blob.
    - conf_logits: (N,) or (N,1) raw logits from the confidence head
    - labels: (N,) binary labels (0/1) indicating correctness
    - out_path: file path to write JSON {"conf_temp": T, "ece_before": e0, "ece_after": e1}
    Returns a dict with the same keys.
    """
    x = conf_logits.detach().float().view(-1)
    y = labels.detach().float().view(-1)
    # ECE before calibration
    p0 = torch.sigmoid(x)
    e0 = ece_binary(p0, y)
    # Fit temperature and evaluate
    T = temperature_fit(x, y)
    p1 = torch.sigmoid(x / max(T, 1e-6))
    e1 = ece_binary(p1, y)
    blob = {"conf_temp": float(T), "ece_before": float(e0), "ece_after": float(e1)}
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(blob), encoding="utf-8")
    return blob


def _extract_think_span(body: str) -> str:
    i = body.find("<think>")
    j = body.find("</think>", i + len("<think>")) if i != -1 else -1
    if i != -1 and j != -1:
        return body[i + len("<think>"):j]
    return ""


def _word_count(s: str) -> int:
    return len([w for w in s.strip().split() if w])


def compute_eval_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute evaluation metrics from a list of sample records.
    Each record should contain:
      - 'body': '<think>..</think><answer>..</answer>'
      - 'gold': reference answer string (optional)
      - optional: 'plan_true','plan_pred','budget_pred','budget_target','conf_prob','correct'
    Returns a dict with accuracy, think token stats, plan acc, budget error, leakage rate, and ECE if conf present.
    """
    n = len(records) or 1
    corrects: List[int] = []
    leakage = 0
    think_counts: List[int] = []
    plan_hits = 0
    plan_total = 0
    b_err_abs = []
    b_err_signed = []
    conf_probs: List[float] = []
    conf_labels: List[int] = []

    for r in records:
        body = r.get("body") or ""
        gold = (r.get("gold") or "").strip()
        ans = _extract_answer(body, include_think=False)
        if gold:
            corrects.append(1 if ans.strip() == gold else 0)
        elif "correct" in r:
            corrects.append(int(r.get("correct") or 0))
        else:
            corrects.append(0)

        # leakage: answer extraction should remove tags
        if "<think>" in ans or "</think>" in ans or "<answer>" in ans or "</answer>" in ans:
            leakage += 1

        think_counts.append(_word_count(_extract_think_span(body)))

        if r.get("plan_true") is not None and r.get("plan_pred") is not None:
            plan_total += 1
            plan_hits += int(r["plan_true"] == r["plan_pred"])

        if r.get("budget_pred") is not None and r.get("budget_target") is not None:
            bp = float(r["budget_pred"]) ; bt = float(r["budget_target"]) 
            b_err_signed.append(bp - bt)
            b_err_abs.append(abs(bp - bt))

        if r.get("conf_prob") is not None and (gold or ("correct" in r)):
            conf_probs.append(float(r["conf_prob"]))
            conf_labels.append(int(corrects[-1]))

    acc = sum(corrects) / max(1, len(corrects))
    plan_acc = (plan_hits / plan_total) if plan_total > 0 else None
    budget_mae = (sum(b_err_abs) / len(b_err_abs)) if b_err_abs else None
    budget_me = (sum(b_err_signed) / len(b_err_signed)) if b_err_signed else None
    leakage_rate = leakage / n
    ece = (ece_binary(torch.tensor(conf_probs), torch.tensor(conf_labels)) if conf_probs else None)

    return {
        "accuracy": acc,
        "think_tokens_mean": (sum(think_counts) / len(think_counts)) if think_counts else 0.0,
        "plan_accuracy": plan_acc,
        "budget_mae": budget_mae,
        "budget_me": budget_me,
        "leakage_rate": leakage_rate,
        "ece": float(ece) if ece is not None else None,
    }


def quality_vs_budget_curve(
    records: List[Dict[str, Any]],
    budgets: Iterable[int],
    *,
    quality_fn: Optional[Callable[[Dict[str, Any], int], float]] = None,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sweep decoding control (budget caps) and compute quality vs budget curve.
    - records: list of samples; if quality_fn is None, uses sample['correct'] else calls quality_fn(sample, B)
    - budgets: iterable of budget caps (ints)
    - out_path: optional JSON/CSV path to save curve
    Returns {'curve': [{'budget': B, 'accuracy': acc}], 'budgets': [...]}.
    """
    curve = []
    budgets_list = list(int(b) for b in budgets)
    for B in budgets_list:
        vals = []
        for r in records:
            if quality_fn is None:
                vals.append(float(r.get("correct", 0)))
            else:
                vals.append(float(quality_fn(r, B)))
        acc = sum(vals) / max(1, len(vals))
        curve.append({"budget": B, "accuracy": acc})

    out = {"budgets": budgets_list, "curve": curve}
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            p.write_text(json.dumps(out), encoding="utf-8")
        else:
            # CSV header then rows
            rows = ["budget,accuracy\n"] + [f"{c['budget']},{c['accuracy']}\n" for c in curve]
            p.write_text("".join(rows), encoding="utf-8")
    return out
