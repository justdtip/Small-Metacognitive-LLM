from __future__ import annotations
from typing import List, Dict, Any, Iterable, Callable, Optional
from pathlib import Path
import json
import sys
import torch
from pathlib import Path
import json as _json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tina.serve import _extract_answer, StopOnTags, SlackStop, _count_think_tokens
from train.metrics import temperature_fit, ece_binary, f1_token
import os as _os


def eval_extract_answers(texts: List[str]) -> List[str]:
    """Eval-only helper: run extraction parity on provided bodies (think+answer strings)."""
    return [_extract_answer(t, include_think=False) for t in texts]


def fit_confidence_temperature_and_save(
    conf_logits: torch.Tensor,
    labels: torch.Tensor,
    out_path: str,
    *,
    plan_thresholds: Dict[str, float] | None = None,
    budget_posthoc_clip: int | None = None,
) -> Dict[str, Any]:
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
    if isinstance(plan_thresholds, dict) and plan_thresholds:
        # sanitize to plain float mapping
        blob["plan_thresholds"] = {str(k): float(v) for k, v in plan_thresholds.items() if v is not None}
    if isinstance(budget_posthoc_clip, (int, float)) and budget_posthoc_clip:
        blob["budget_posthoc_clip"] = int(budget_posthoc_clip)
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
    # Decomposition telemetry accumulators
    plan_tokens_all: List[float] = []
    exec_tokens_all: List[float] = []
    eval_tokens_all: List[float] = []
    plan_frac_all: List[float] = []
    exec_frac_all: List[float] = []
    eval_frac_all: List[float] = []
    plan_present = 0
    exec_present = 0
    eval_present = 0

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
        elif "correctness" in r:
            # accept 'correctness' synonym used in training paths
            try:
                corrects.append(int(r.get("correctness") or 0))
            except Exception:
                corrects.append(0)
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

        # Optional decomposition metrics if masks or counts provided on the record
        # Accept either full-length boolean masks or precomputed counts along with a denominator
        try:
            # Derive per-record think denominator for fractions if masks are present
            denom = None
            tm = r.get("think_mask")
            if tm is not None:
                try:
                    import torch as _t
                    if isinstance(tm, _t.Tensor):
                        denom = float(tm.to(dtype=_t.float32).sum().item())
                    else:
                        denom = float(sum(1 for v in tm if int(v) != 0))
                except Exception:
                    denom = None
            # plan
            pt = None
            if r.get("plan_tokens") is not None:
                pt = float(r.get("plan_tokens") or 0.0)
            elif r.get("plan_mask") is not None:
                pm = r.get("plan_mask")
                try:
                    import torch as _t
                    if isinstance(pm, _t.Tensor):
                        pt = float(pm.to(dtype=_t.float32).sum().item())
                    else:
                        pt = float(sum(1 for v in pm if int(v) != 0))
                except Exception:
                    pt = None
            if pt is not None and pt > 0:
                plan_present += 1
                plan_tokens_all.append(pt)
                if denom and denom > 0:
                    plan_frac_all.append(float(pt / denom))
            # exec
            et = None
            if r.get("exec_tokens") is not None:
                et = float(r.get("exec_tokens") or 0.0)
            elif r.get("exec_mask") is not None:
                em = r.get("exec_mask")
                try:
                    import torch as _t
                    if isinstance(em, _t.Tensor):
                        et = float(em.to(dtype=_t.float32).sum().item())
                    else:
                        et = float(sum(1 for v in em if int(v) != 0))
                except Exception:
                    et = None
            if et is not None and et > 0:
                exec_present += 1
                exec_tokens_all.append(et)
                if denom and denom > 0:
                    exec_frac_all.append(float(et / denom))
            # eval
            vt = None
            if r.get("eval_tokens") is not None:
                vt = float(r.get("eval_tokens") or 0.0)
            elif r.get("eval_mask") is not None:
                vm = r.get("eval_mask")
                try:
                    import torch as _t
                    if isinstance(vm, _t.Tensor):
                        vt = float(vm.to(dtype=_t.float32).sum().item())
                    else:
                        vt = float(sum(1 for v in vm if int(v) != 0))
                except Exception:
                    vt = None
            if vt is not None and vt > 0:
                eval_present += 1
                eval_tokens_all.append(vt)
                if denom and denom > 0:
                    eval_frac_all.append(float(vt / denom))
        except Exception:
            pass

        if r.get("plan_true") is not None and r.get("plan_pred") is not None:
            plan_total += 1
            plan_hits += int(r["plan_true"] == r["plan_pred"])

        if r.get("budget_pred") is not None and r.get("budget_target") is not None:
            bp = float(r["budget_pred"]) ; bt = float(r["budget_target"]) 
            b_err_signed.append(bp - bt)
            b_err_abs.append(abs(bp - bt))

        if r.get("conf_prob") is not None and (gold or ("correct" in r) or ("correctness" in r)):
            conf_probs.append(float(r["conf_prob"]))
            conf_labels.append(int(corrects[-1]))

        # Collect adapter gate coverage from either legacy or new key
        gv = r.get("gate_coverage")
        if gv is None:
            gv = r.get("gate_coverage_mean")
        if gv is not None:
            try:
                gate_covs.append(float(gv))
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

        # Evidence coverage if evidence_keys provided
        if isinstance(r.get("evidence_keys"), list) and r.get("evidence_keys"):
            th_span = _extract_think_span(body)
            keys = [str(k) for k in r.get("evidence_keys")]
            lo = th_span.lower()
            num = 0
            for k in keys:
                try:
                    if str(k).lower() in lo:
                        num += 1
                except Exception:
                    pass
            if len(keys) > 0:
                think_scores.append(float(num / len(keys)))

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
        # Decomposition telemetry aggregates
        "mean_plan_fraction": (sum(plan_frac_all) / len(plan_frac_all)) if plan_frac_all else None,
        "mean_exec_fraction": (sum(exec_frac_all) / len(exec_frac_all)) if exec_frac_all else None,
        "mean_eval_fraction": (sum(eval_frac_all) / len(eval_frac_all)) if eval_frac_all else None,
        "presence_rate_plan": (plan_present / max(1, len(records))) if records else None,
        "presence_rate_exec": (exec_present / max(1, len(records))) if records else None,
        "presence_rate_eval": (eval_present / max(1, len(records))) if records else None,
    }


def quality_vs_budget_curve(
    records_or_model: Any,
    budgets: Iterable[int],
    *,
    quality_fn: Optional[Callable[[Dict[str, Any], int], float]] = None,
    out_path: Optional[str] = None,
    # Decode-sweep mode (optional): if provided, runs live decoding per budget
    tokenizer: Any = None,
    prompts: Optional[Iterable[Any]] = None,
    out_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sweep decoding control (budget caps) and compute quality vs budget curve.
    - records: list of samples; if quality_fn is None, uses sample['correct'] else calls quality_fn(sample, B)
    - budgets: iterable of budget caps (ints)
    - out_path: optional JSON/CSV path to save curve
    Returns {'curve': [{'budget': B, 'accuracy': acc}], 'budgets': [...]}.
    """
    budgets_list = list(int(b) for b in budgets)
    # Dispatch: records-mode vs decode-sweep mode
    if isinstance(records_or_model, list) and (not records_or_model or isinstance(records_or_model[0], dict)):
        records = records_or_model
        curve = []
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
                rows = ["budget,accuracy\n"] + [f"{c['budget']},{c['accuracy']}\n" for c in curve]
                p.write_text("".join(rows), encoding="utf-8")
        return out

    # Decode-sweep mode
    model = records_or_model
    if tokenizer is None or prompts is None:
        raise ValueError("Decode sweep requires 'tokenizer' and 'prompts'.")
    rows = []
    for B in budgets_list:
        recs: List[Dict[str, Any]] = []
        for pr in prompts:
            if isinstance(pr, dict):
                text = pr.get("prompt") or pr.get("text") or ""
                gold = pr.get("gold")
            else:
                text = str(pr)
                gold = None
            # Build input_ids for this prompt
            try:
                enc = tokenizer(text, add_special_tokens=False, return_tensors="pt")
                input_ids = getattr(enc, "input_ids", None) if not isinstance(enc, dict) else enc.get("input_ids")
            except Exception:
                input_ids = None
            if input_ids is None:
                # Fallback minimal input
                import torch as _t
                input_ids = _t.tensor([[1, 2, 3]], dtype=_t.long)
            d = decode_with_budget(tokenizer, model, input_ids, think_budget=B, max_new_tokens=256, temperature=0.0, visible_cot=False)
            rec = {"body": d.get("text") or "", "think_tokens_used": int(d.get("think_tokens_used") or 0)}
            if gold is not None:
                rec["gold"] = str(gold)
            recs.append(rec)
        m = compute_eval_metrics(recs)
        rows.append({
            "budget": B,
            "n": len(recs),
            "accuracy": float(m.get("accuracy") or 0.0),
            "f1": float(m.get("f1")) if (m.get("f1") is not None) else None,
            "mean_think_tokens": float(m.get("think_tokens_mean") or 0.0),
            "mean_plan_fraction": m.get("mean_plan_fraction"),
            "mean_exec_fraction": m.get("mean_exec_fraction"),
            "mean_eval_fraction": m.get("mean_eval_fraction"),
            "presence_rate_plan": m.get("presence_rate_plan"),
            "presence_rate_exec": m.get("presence_rate_exec"),
            "presence_rate_eval": m.get("presence_rate_eval"),
        })
    # Optional CSV
    if out_csv:
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="") as f:
            f.write("budget,n,accuracy,f1,mean_think_tokens,mean_plan_fraction,mean_exec_fraction,mean_eval_fraction,presence_rate_plan,presence_rate_exec,presence_rate_eval\n")
            for r in rows:
                f.write(
                    f"{r['budget']},{r['n']},{r['accuracy']},{'' if r['f1'] is None else r['f1']},{r['mean_think_tokens']}"
                    f",{'' if r.get('mean_plan_fraction') is None else r['mean_plan_fraction']}"
                    f",{'' if r.get('mean_exec_fraction') is None else r['mean_exec_fraction']}"
                    f",{'' if r.get('mean_eval_fraction') is None else r['mean_eval_fraction']}"
                    f",{'' if r.get('presence_rate_plan') is None else r['presence_rate_plan']}"
                    f",{'' if r.get('presence_rate_exec') is None else r['presence_rate_exec']}"
                    f",{'' if r.get('presence_rate_eval') is None else r['presence_rate_eval']}\n"
                )
    return {"budgets": budgets_list, "curve": rows}


def load_service_config(root: Optional[str] = None) -> Dict[str, Any]:
    """
    Load service_config.json to centralize parity behavior (visibility, stop sequences, calibration path).
    Returns a dict with keys: visible_cot_default, stop_sequences, think_stop_sequences, confidence_calibration_path.
    """
    # Env overrides for tests and tooling
    env_path = _os.environ.get("SERVICE_CONFIG_PATH")
    if env_path:
        p = Path(env_path)
    else:
        base_env = _os.environ.get("CONFIG_ROOT")
        base = Path(base_env) if base_env else (Path(root) if root else Path(__file__).resolve().parents[1])
        p = base / "config" / "service_config.json"
    if not p.exists():
        raise ValueError(
            f"service_config.json not found at {p}. Example: { '{' } 'visible_cot_default': false, 'stop_sequences': ['</answer>'], "
            "'think_stop_sequences': ['</think>'], 'soft_cap_slack_ratio': 0.2 }"
        )
    try:
        d = _json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to parse service_config.json at {p}: {type(e).__name__}: {e}")
    stops = d.get("stop_sequences")
    t_stops = d.get("think_stop_sequences")
    if not stops or not isinstance(stops, list):
        raise ValueError("service_config.stop_sequences missing or empty; see docs.")
    if not t_stops or not isinstance(t_stops, list):
        raise ValueError("service_config.think_stop_sequences missing or empty; see docs.")
    return {
        "visible_cot_default": bool(d.get("visible_cot_default", False)),
        "stop_sequences": list(stops),
        "think_stop_sequences": list(t_stops),
        "confidence_calibration_path": str(d.get("confidence_calibration_path") or ""),
        "soft_cap_slack_ratio": float(d.get("soft_cap_slack_ratio")),
    }



def decode_with_budget(
    tokenizer,
    model,
    input_ids,
    *,
    think_budget: int,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
    ignore_eos: bool = False,
    visible_cot: bool = False,
    style_tag: Optional[str] = None,
    verbose: bool = False,
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
    # Optional style hint before THINK
    if style_tag:
        try:
            hint = tokenizer(f"<style:{style_tag}>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            input_ids = torch.cat([input_ids, hint], dim=1)
            attention_mask = torch.ones_like(input_ids)
        except Exception:
            pass
    eos_id = None if ignore_eos else getattr(tokenizer, "eos_token_id", None)

    # THINK with soft cap + closing tag from service config
    svc = load_service_config()
    think_tags = tuple(svc.get("think_stop_sequences"))
    ans_tags = tuple(svc.get("stop_sequences"))
    # Validate tags encode to 1 token and optionally log ids
    def _enc_single(tag: str) -> list[int]:
        try:
            ids = tokenizer.encode(tag, add_special_tokens=False)
        except Exception:
            ids = tokenizer(tag, add_special_tokens=False).input_ids
        if not isinstance(ids, list):
            ids = list(ids)
        if len(ids) != 1:
            raise ValueError(f"Stop tag '{tag}' does not encode to a single token: {ids}")
        return ids
    t_ids = [_enc_single(t) for t in think_tags]
    a_ids = [_enc_single(t) for t in ans_tags]
    if verbose:
        import sys as _sys
        _sys.stderr.write(f"think_stop={list(think_tags)} ids={t_ids}\nanswer_stop={list(ans_tags)} ids={a_ids}\n")

    stop_think = StopOnTags(tokenizer, think_tags, max_new=None)
    ratio = float(svc.get("soft_cap_slack_ratio"))
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

    # Count think tokens used via helper
    try:
        close_ids_list = []
        for tag in think_tags:
            try:
                close_ids_list.append(tokenizer.encode(str(tag), add_special_tokens=False))
            except Exception:
                close_ids_list.append(tokenizer(str(tag), add_special_tokens=False).input_ids)
        used = _count_think_tokens(seqs1[0], close_ids_list, base_len=input_ids.shape[1], cap=int(think_budget), slack_ratio=ratio)
    except Exception:
        used = int(gen1.shape[0])

    if not any(tag in text1 for tag in think_tags):
        # Append the configured closing tag to keep downstream assembly stable
        close_tag = think_tags[0] if think_tags else "</think>"
        text1 = text1 + close_tag + "\n"

    # ANSWER until </answer>
    ans_tok = tokenizer("<answer>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    full_prompt_ids = torch.cat([seqs1, ans_tok], dim=1)
    full_attn = torch.ones_like(full_prompt_ids)
    stop_answer = StopOnTags(tokenizer, ans_tags, max_new=max_new_tokens)
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
    ans = _extract_answer(body, include_think=False)
    # Sanitize any decomposition or think tags if they leaked into answer
    try:
        for tag in ("<think>", "</think>", "<plan>", "</plan>", "<exec>", "</exec>", "<eval>", "</eval>"):
            ans = ans.replace(tag, "")
        import re as _re
        ans = _re.sub(r"<strategy:[^>]+>", "", ans).strip()
    except Exception:
        pass
    return {"text": ans, "think_tokens_used": int(used)}


def check_answer_invariance_for_hints(
    tokenizer,
    model,
    prompt: str,
    hints: Iterable[str],
    *,
    think_budget: int = 16,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    decode_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Decode with multiple strategy hints and verify that extracted answers are invariant.
    - Inject each hint as a control tag (test-time stubs may embed <strategy:...> in outputs).
    - normalize answers by stripping any '<strategy:...>' control tokens before comparison.
    Returns {'answers': {hint: answer}, 'all_equal': bool}.
    """
    import re as _re
    decode = decode_fn or decode_with_budget
    answers: Dict[str, str] = {}
    # Build base prompt ids once
    try:
        enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        base_ids = getattr(enc, "input_ids", None) if not isinstance(enc, dict) else enc.get("input_ids")
    except Exception:
        base_ids = None
    for h in hints:
        # Visible mode to capture full body for normalization
        out = decode(tokenizer, model, base_ids if base_ids is not None else [], think_budget=think_budget,
                     max_new_tokens=max_new_tokens, temperature=temperature, visible_cot=True, style_tag=h)
        body = out.get("text") or ""
        ans = _extract_answer(body, include_think=False)
        ans = _re.sub(r"<strategy:[^>]+>", "", ans).strip()
        answers[str(h)] = ans
    vals = list(answers.values())
    all_equal = all(v == vals[0] for v in vals)
    return {"answers": answers, "all_equal": all_equal}


def _check_config_cli(root: Optional[str] = None, service_config_path: Optional[str] = None) -> int:
    """
    Run the same validation performed by tools/validate_configs.py, optionally against a provided root or config path.
    If a custom service_config is provided, it will be copied into ROOT/config for validation scope.
    Returns process exit code (0 OK, 1 errors printed to stderr).
    """
    import sys
    from pathlib import Path as _P
    base = _P(root) if root else _P(__file__).resolve().parents[1]
    # Allow validator to read a custom service_config path without relocating model files
    # Delegate to validator main() with CONFIG_ROOT override
    try:
        import os as _os
        if service_config_path:
            _os.environ["SERVICE_CONFIG_PATH"] = str(service_config_path)
        else:
            # If a root is provided and contains config/service_config.json, use it
            cand = base / "config" / "service_config.json"
            if cand.exists():
                _os.environ["SERVICE_CONFIG_PATH"] = str(cand)
        # Lazy import to honor CONFIG_ROOT at import time
        from tools import validate_configs as _vc  # type: ignore
        rc = int(_vc.main())
        # Also print stop tags and ids to stderr for human inspection
        try:
            svc = load_service_config()
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(base / 'model' / 'Base'), use_fast=True, local_files_only=True, trust_remote_code=True)
            def _enc(tag: str):
                try:
                    ids = tok.encode(tag, add_special_tokens=False)
                except Exception:
                    ids = tok(tag, add_special_tokens=False).input_ids
                return ids if isinstance(ids, list) else list(ids)
            import sys as _sys
            _sys.stderr.write(f"think_stop={svc.get('think_stop_sequences')} ids={[ _enc(t) for t in svc.get('think_stop_sequences') ]}\n")
            _sys.stderr.write(f"answer_stop={svc.get('stop_sequences')} ids={[ _enc(t) for t in svc.get('stop_sequences') ]}\n")
        except Exception:
            pass
        return rc
    except SystemExit as se:
        return int(getattr(se, "code", 1) or 0)
    except Exception as e:
        import sys as _sys
        _sys.stderr.write(f"validation error: {type(e).__name__}: {e}\n")
        return 1


def _parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-config", action="store_true", help="Validate config/schema and serve/eval parity")
    ap.add_argument("--root", default=None, help="Optional project root for service config lookup")
    ap.add_argument("--service-config", default=None, help="Optional explicit path to a service_config.json to validate")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if getattr(args, "check_config", False):
        return _check_config_cli(args.root, args.service_config)
    # No standalone CLI for eval loop otherwise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

def inject_noise_markers_into_think(body: str, markers: Optional[List[str]] = None) -> str:
    markers = markers or ["Step 1:", "===="]
    i = body.find("<think>")
    j = body.find("</think>", i + len("<think>")) if i != -1 else -1
    if i != -1 and j != -1:
        pre = body[:i + len("<think>")]
        mid = body[i + len("<think>"):j]
        post = body[j:]
        injected = " ".join(markers) + "\n" + mid
        return pre + injected + post
    return body

def noise_probe(records: List[Dict[str, Any]], markers: Optional[List[str]] = None) -> Dict[str, float]:
    base = compute_eval_metrics(records)
    rec2: List[Dict[str, Any]] = []
    for r in records:
        b = (r.get('body') or '')
        b2 = inject_noise_markers_into_think(b, markers)
        rr = dict(r)
        rr['body'] = b2
        rec2.append(rr)
    alt = compute_eval_metrics(rec2)
    return {
        'delta_accuracy': float((alt.get('accuracy') or 0.0) - (base.get('accuracy') or 0.0)),
        'delta_budget': float((alt.get('think_tokens_mean') or 0.0) - (base.get('think_tokens_mean') or 0.0)),
    }
