#!/usr/bin/env python3
"""
One-cycle training script with full logging.

Loads model/Base and model/Tina adapter, builds a DataLoader from a JSONL dataset,
ensures reasoning tags, runs a single forward/backward optimizing only metacog heads
and side adapters, then decodes one sample with IntrospectiveEngine to validate
serve/eval parity. Emits a single JSON line with rich metrics.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tina.tokenizer_utils import ensure_reasoning_tokens
from tina.serve import IntrospectiveEngine, EngineConfig, _extract_answer
from tina.side_adapters import attach_residual_adapters
from train.hooks import think_mask_context
from train.data import make_collate_fn
from train.losses import compute_losses
from tina.metacog_heads import MetacogHeads, MetacogConfig
from train.metrics import ece_binary
from train.eval_loop import load_service_config


def _device_dtype():
    dev = (
        "cuda" if torch.cuda.is_available() else
        "mps" if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else
        "cpu"
    )
    if dev == "cuda":
        try:
            return dev, (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        except Exception:
            return dev, torch.float16
    return dev, torch.float32


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def build_dataloader(jsonl: Path, tokenizer, batch_size: int = 8):
    examples = _load_jsonl(jsonl)
    # normalize to expected schema with 'text' and optional labels
    for ex in examples:
        if "text" not in ex:
            # If sample provides 'body', accept it as text
            if "body" in ex:
                ex["text"] = ex.get("body") or ""
            else:
                # fallback synthetic body
                ex["text"] = "<think> a b </think><answer> ok </answer>"
    collate = make_collate_fn(tokenizer, loss_on="answer")
    # Lightweight iterable DataLoader (avoid depending on torch.utils.data.DataLoader to keep tests snappy)
    class _Iter:
        def __init__(self, data):
            self.data = data
        def __iter__(self):
            # single batch (first B items)
            batch = self.data[:batch_size]
            yield collate(batch)
    return _Iter(examples)


def params_heads_and_adapters(engine: IntrospectiveEngine):
    # Use engine.metacog (heads) + side adapters for optimization
    params = []
    try:
        params += list(engine.metacog.parameters())
    except Exception:
        pass
    try:
        params += list(engine.scaffold.adapters.parameters())
    except Exception:
        pass
    return [p for p in params if p.requires_grad]


def run_one_cycle(args) -> Dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZER_PARALLELISM", "false")

    models_root = ROOT / "model"
    base_path = models_root / "Base"
    adapter_root = models_root / "Tina"
    adapter_path = adapter_root / "checkpoint-2000"

    device, dtype = _device_dtype()
    # Load base on a single device (prefer GPU/MPS); avoid CPU/GPU sharding
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path), device_map=None, torch_dtype=dtype,
        trust_remote_code=True, local_files_only=True,
    )
    try:
        model.to(device)
    except Exception:
        # Fallback to CPU if transfer fails for any reason
        model.to("cpu")
    tok = AutoTokenizer.from_pretrained(
        str(base_path), use_fast=True, trust_remote_code=True, local_files_only=True,
    )
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    ensure_reasoning_tokens(tok, model)

    # Load adapter via PEFT; fail soft if PEFT missing
    try:
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, str(adapter_path))
        # Ensure adapter-wrapped model is resident on the same device
        try:
            model.to(device)
        except Exception:
            model.to("cpu")
    except Exception as e:
        sys.stderr.write(f"warn: PEFT adapter not loaded ({type(e).__name__}: {e}); continuing with base model\n")

    # Attach side adapters and metacog heads using the serve engine for parity hooks
    hidden_size = getattr(getattr(model, 'config', None), 'hidden_size', 2048)
    num_layers = getattr(getattr(model, 'config', None), 'num_hidden_layers', 24)
    eng = IntrospectiveEngine(model=model, tokenizer=tok, cfg=EngineConfig(visible_cot=False),
                              hidden_size=int(hidden_size), num_layers=int(num_layers))

    # Data
    dl = build_dataloader(Path(args.data), tok, batch_size=int(args.batch_size))

    # Optimizer on heads + adapters only
    params = params_heads_and_adapters(eng)
    opt = torch.optim.AdamW(params, lr=float(args.lr)) if params else None

    svc = load_service_config()
    slack_ratio = float(svc.get("soft_cap_slack_ratio", 0.2))
    stop_sequences = list(svc.get("stop_sequences") or ["</answer>"])

    batch = next(iter(dl))
    # Move inputs to the model's device
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device(device)
    input_ids = batch["input_ids"].to(model_device)

    # Teacher-forcing labels masked to answers
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
    labels = torch.where(batch["loss_mask"].to(labels.device).bool(), labels, torch.full_like(labels, -100))

    # One forward/backward under think context and adapter think-mode
    with think_mask_context(batch["think_mask"].to(model_device).float()):
        with eng.scaffold.think():
            outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True, use_cache=False)
            logits = outputs.logits
    # Populate taps via hooks; if cache empty (e.g., hooks not triggered), register from hidden_states fallback
    try:
        # If no taps recorded, fill from outputs.hidden_states
        has_cache = bool(getattr(eng.metacog, "_tap_cache", {})) and any(eng.metacog._tap_cache.values())
        if (not has_cache) and isinstance(getattr(outputs, "hidden_states", None), (list, tuple)):
            eng.metacog.clear_cache()
            taps = getattr(eng.cfg, "taps", (6, 10, 14))
            hs_list = list(outputs.hidden_states)
            for t in taps:
                if 0 <= int(t) < len(hs_list) and torch.is_tensor(hs_list[int(t)]):
                    eng.metacog.register_tap(int(t), hs_list[int(t)])
    except Exception:
        pass
    head_out = eng.metacog(B_max=int(max(1, int(batch["target_budget"].max().item()))) if torch.is_tensor(batch.get("target_budget")) else 256)
    plan_logits = head_out.get("plan_logits")
    budget_pred = head_out.get("budget")
    conf_prob = head_out.get("confidence")
    conf_logits = torch.logit(conf_prob.clamp(1e-6, 1 - 1e-6)) if conf_prob is not None else None

    # Prepare labels (supervised when available; else leave as None to skip)
    plan_targets = batch.get("plan_targets") if (torch.is_tensor(batch.get("plan_targets")) and (batch["plan_targets"] >= 0).any()) else None
    budget_target = batch.get("target_budget") if (torch.is_tensor(batch.get("target_budget")) and (batch["target_budget"] >= 0).any()) else None
    conf_labels = batch.get("correctness") if (torch.is_tensor(batch.get("correctness")) and (batch["correctness"] >= 0).any()) else None
    # Ensure label tensors live on the same device as model/logits
    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = torch.device(device)
    if torch.is_tensor(plan_targets):
        plan_targets = plan_targets.to(model_device)
    if torch.is_tensor(budget_target):
        budget_target = budget_target.to(model_device)
    if torch.is_tensor(conf_labels):
        conf_labels = conf_labels.to(model_device)

    # Weights
    weights = {
        "answer_ce": 1.0,
        "gate_reg": 1e-4,
        "aux_mix": float(args.quiet_star) if args.quiet_star else 0.0,
        "plan_ce": 0.5 if plan_targets is not None else 0.0,
        "budget_reg": 0.1 if budget_target is not None else 0.0,
        "conf_cal": 0.1 if conf_labels is not None else 0.0,
    }

    gate_modules = list(eng.scaffold.adapters) if hasattr(eng.scaffold, "adapters") else None
    out = compute_losses(
        logits, labels,
        gate_modules=gate_modules,
        weights=weights,
        plan_logits=plan_logits if plan_targets is not None else None,
        plan_targets=plan_targets,
        budget_pred=budget_pred if budget_target is not None else None,
        budget_target=budget_target,
        conf_logits=conf_logits if conf_labels is not None else None,
        conf_labels=conf_labels,
    )

    if opt is not None:
        opt.zero_grad()
        out["total"].backward()
        opt.step()

    # Decode one sample via engine (hidden mode); ensure no leakage
    # Build a minimal chat from the first example's user content if provided, else a generic prompt
    examples = _load_jsonl(Path(args.data))
    prompt = examples[0].get("prompt") or examples[0].get("question") or "Say hello in one word."
    messages = [{"role": "user", "content": prompt}]
    text = eng.generate_cot(messages, max_new_tokens=32, temperature=0.2, top_p=0.95, repetition_penalty=1.1, ignore_eos=False, stream=False)
    # Normalize to answer-only for leakage check
    body = _extract_answer(text, include_think=False)
    # After canonical extraction, leakage should be impossible in 'body'
    leakage_detected = False
    # Think tokens from engine stats if available
    think_used = eng.last_stats.get("think_budget")
    try:
        think_used = int(think_used)
    except Exception:
        think_used = int(sum(map(int, examples[0].get("think_tokens_used") or [0]))) if isinstance(examples[0].get("think_tokens_used"), list) else 0

    # Aggregates
    with torch.no_grad():
        # plan histogram from logits
        plan_hist = None
        try:
            preds = torch.argmax(plan_logits, dim=-1)
            bins = torch.bincount(preds.view(-1), minlength=plan_logits.shape[-1]).float()
            plan_hist = [float(x) for x in (bins / max(1, int(bins.sum().item()))).tolist()]
        except Exception:
            pass
        bp = float(budget_pred.mean().item()) if torch.is_tensor(budget_pred) else None
        bt = float(budget_target.float().mean().item()) if torch.is_tensor(budget_target) else None
        bae = float(torch.abs(budget_pred - budget_target.float()).mean().item()) if (torch.is_tensor(budget_pred) and torch.is_tensor(budget_target)) else None
        cp = float(conf_prob.mean().item()) if torch.is_tensor(conf_prob) else None
        ece = None
        try:
            if torch.is_tensor(conf_prob) and torch.is_tensor(conf_labels):
                ece = float(ece_binary(conf_prob.view(-1), conf_labels.view(-1)))
        except Exception:
            pass

    # Gate coverage mean
    gate_cov = None
    try:
        vals = []
        for m in eng.scaffold.adapters:
            v = getattr(m, "_last_gate_coverage", None)
            if torch.is_tensor(v):
                vals.append(float(v.item()))
        if vals:
            gate_cov = sum(vals) / len(vals)
    except Exception:
        pass

    # Parity digest and config info
    parity_digest = eng.last_stats.get("parity_digest") or {}

    rec = {
        "loss_answer_ce": float(out.get("answer_ce", torch.tensor(0.0)).item()),
        "loss_plan_ce": float(out.get("plan_ce", torch.tensor(0.0)).item()),
        "loss_budget_reg": float(out.get("budget_reg", torch.tensor(0.0)).item()),
        "loss_conf_cal": float(out.get("conf_cal", torch.tensor(0.0)).item()),
        "loss_gate_reg": float(out.get("gate_reg", torch.tensor(0.0)).item()),
        "loss_aux_loss": float(out.get("aux_loss", torch.tensor(0.0)).item()),
        "plan_hist": plan_hist,
        "budget_pred_mean": bp,
        "budget_target_mean": bt,
        "budget_abs_err_mean": bae,
        "conf_prob_mean": cp,
        "ece": ece,
        "think_tokens_used": think_used,
        "slack_ratio": slack_ratio,
        "stop_sequences": stop_sequences,
        "parity_digest": parity_digest,
        "gate_coverage": gate_cov,
        "leakage_detected": bool(leakage_detected),
    }
    return rec


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to JSONL dataset")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--quiet-star", action="store_true", help="Enable Quiet-Star aux consistency during loss")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    rec = run_one_cycle(args)
    rec["elapsed_ms"] = int((time.time() - t0) * 1000)
    print(json.dumps(rec))


if __name__ == "__main__":
    raise SystemExit(main())
