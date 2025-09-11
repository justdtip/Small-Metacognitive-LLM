from __future__ import annotations
from typing import List, Dict, Any, Iterable, Callable, Optional
from pathlib import Path
import json
import torch
from tina.serve import _extract_answer, StopOnTags, SlackStop
from pathlib import Path
import json as _json
from train.metrics import temperature_fit, ece_binary, f1_token


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
    think_counts: List[float] = []
    plan_hits = 0
    plan_total = 0
    b_err_abs = []
    b_err_signed = []
    conf_probs: List[float] = []
    conf_labels: List[int] = []
    gate_covs: List[float] = []
    think_scores: List[float] = []
    f1s: List[float] = []
    for r in records:
        body = r.get("body") or ""
        gold = (r.get("gold") or "").strip()
        ans = _extract_answer(body, include_think=False)
        if gold:
            corrects.append(1 if ans.strip() == gold else 0)
            try:
                f1s.append(float(f1_token(ans, gold)))
            except Exception:
                pass
        elif "correct" in r:
            corrects.append(int(r.get("correct") or 0))
        else:
            corrects.append(0)

        # leakage: answer extraction should remove tags
        if "<think>" in ans or "</think>" in ans or "<answer>" in ans or "</answer>" in ans:
            leakage += 1

        # Prefer explicit token count if provided; else proxy via words in think span
        if r.get("think_tokens_used") is not None:
            try:
                think_counts.append(float(r.get("think_tokens_used")))
            except Exception:
                think_counts.append(float(_word_count(_extract_think_span(body))))
        else:
            think_counts.append(float(_word_count(_extract_think_span(body))))

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

        if "gate_coverage" in r and r.get("gate_coverage") is not None:
            try:
                gate_covs.append(float(r.get("gate_coverage")))
            except Exception:
                pass

        # Optional: record provided think_score directly
        if r.get("think_score") is not None:
            try:
                ts = float(r.get("think_score"))
                if 0.0 <= ts <= 1.0:
                    think_scores.append(ts)
            except Exception:
                pass
        # Or compute a rubric score if a grader is provided in the record
        if r.get("body") is not None and r.get("think_grader") is not None:
            try:
                from train.metrics import think_rubric_score
                s = think_rubric_score(r.get("body"), r.get("think_grader"))
                if s is not None:
                    think_scores.append(s)
            except Exception:
                pass

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
        "gate_coverage_mean": (sum(gate_covs) / len(gate_covs)) if gate_covs else None,
        "f1": (sum(f1s) / len(f1s)) if f1s else None,
        "think_score_mean": (sum(think_scores) / len(think_scores)) if think_scores else None,
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


def load_service_config(root: Optional[str] = None) -> Dict[str, Any]:
    """
    Load service_config.json to centralize parity behavior (visibility, stop sequences, calibration path).
    Returns a dict with keys: visible_cot_default, stop_sequences, think_stop_sequences, confidence_calibration_path.
    """
    try:
        base = Path(root) if root else Path(__file__).resolve().parents[1]
        p = base / "config" / "service_config.json"
        if not p.exists():
            return {
                "visible_cot_default": False,
                "stop_sequences": ["</answer>"],
                "think_stop_sequences": ["</think>"],
                "confidence_calibration_path": "",
                "soft_cap_slack_ratio": 0.2,
            }
        d = _json.loads(p.read_text(encoding="utf-8"))
        return {
            "visible_cot_default": bool(d.get("visible_cot_default", False)),
            "stop_sequences": list(d.get("stop_sequences") or ["</answer>"]),
            "think_stop_sequences": list(d.get("think_stop_sequences") or ["</think>"]),
            "confidence_calibration_path": str(d.get("confidence_calibration_path") or ""),
            "soft_cap_slack_ratio": float(d.get("soft_cap_slack_ratio", 0.2)),
        }
    except Exception:
        return {
            "visible_cot_default": False,
            "stop_sequences": ["</answer>"],
            "think_stop_sequences": ["</think>"],
            "confidence_calibration_path": "",
            "soft_cap_slack_ratio": 0.2,
        }



def decode_with_budget(
    tokenizer,
    model,
    input_ids,
    *,
    think_budget: int,
    slack_ratio: float = 0.2,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    ignore_eos: bool = False,
    visible_cot: bool = False,
) -> Dict[str, Any]:
    """
    Generate using a soft-cap budget for the <think> segment and StopOnTags for </think> and </answer>.
    Mirrors tina/serve.py stopping criteria to keep train/eval/serve parity.
    Returns {'text': str, 'think_tokens_used': int} where 'text' respects visible_cot.
    """
    import torch
    from transformers import StoppingCriteriaList

    device = next(model.parameters()).device if hasattr(model, "parameters") else getattr(model, "device", "cpu")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    eos_id = None if ignore_eos else getattr(tokenizer, "eos_token_id", None)

    # THINK with soft cap + closing tag
    svc = load_service_config()
    stop_think = StopOnTags(tokenizer, ("</think>",), max_new=None)
    ratio = float(svc.get("soft_cap_slack_ratio", slack_ratio))
    soft_cap = SlackStop(base_len=input_ids.shape[1], budget=int(think_budget), slack_ratio=ratio)

    with torch.no_grad():
        out1 = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([stop_think, soft_cap]),
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )
    seqs1 = getattr(out1, "sequences", out1)
    gen1 = seqs1[0, input_ids.shape[1]:]
    text1 = tokenizer.decode(gen1, skip_special_tokens=True)

    # Count think tokens used (before </think> if present)
    try:
        close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        g = gen1.tolist()
        used = len(g)
        k = len(close_ids)
        if k > 0:
            for i in range(max(0, len(g) - k), -1, -1):
                if g[i:i + k] == close_ids:
                    used = i
                    break
    except Exception:
        used = int(gen1.shape[0])

    if "</think>" not in text1:
        text1 = text1 + "</think>\n"

    # ANSWER until </answer>
    ans_tok = tokenizer("<answer>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full_prompt_ids = torch.cat([seqs1, ans_tok], dim=1)
    full_attn = torch.ones_like(full_prompt_ids)
    stop_answer = StopOnTags(tokenizer, ("</answer>",), max_new=max_new_tokens)
    with torch.no_grad():
        out2 = model.generate(
            input_ids=full_prompt_ids,
            attention_mask=full_attn,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=StoppingCriteriaList([stop_answer]),
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
        )
    seqs2 = getattr(out2, "sequences", out2)
    gen2 = seqs2[0, full_prompt_ids.shape[1]:]
    text2 = tokenizer.decode(gen2, skip_special_tokens=True)

    body = (text1 or "") + (text2 or "")
    if visible_cot:
        return {"text": body.strip(), "think_tokens_used": int(used)}
    return {"text": _extract_answer(body, include_think=False), "think_tokens_used": int(used)}
