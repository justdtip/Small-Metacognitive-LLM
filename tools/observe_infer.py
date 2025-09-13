#!/usr/bin/env python3
"""
Activation logging during inference (no training required).

Emits one JSON line with:
  - request_id, timing, device, decode settings
  - token counts (prompt/gen), section counts (think/answer)
  - metacog heads outputs (if available)
  - per-layer hidden-state L2 norms (last token)
  - attention entropies (last token) if returned
  - side-adapter gate activity (if exposed)
  - answer text (hidden CoT removed by default)

Usage examples:
  python tools/observe_infer.py \
    --models-root model --base Base --adapter Tina --subfolder checkpoint-2000 \
    --prompt "Factor 12345" --visible-cot false --jsonl-out logs/obs.jsonl --debug-per-layer --calibration artifacts/metacog_calibration.json
"""
import argparse, json, time, uuid, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import json as _json

from tina.tokenizer_utils import ensure_reasoning_tokens
try:
    from tina.serve import _extract_answer  # reuse canonical extraction
except Exception:
    def _extract_answer(body: str, include_think: bool = False) -> str:
        def _slice(s, open_t, close_t):
            i = s.find(open_t)
            j = s.find(close_t, i + len(open_t)) if i != -1 else -1
            return s[i + len(open_t):j] if (i != -1 and j != -1) else s
        if include_think:
            return body.strip()
        ans = _slice(body, "<answer>", "</answer>")
        return ans.strip()

from tina.serve import IntrospectiveEngine, EngineConfig


class StopOnTags(StoppingCriteria):
    def __init__(self, tokenizer, stop_strs=("</answer>",)):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strs]
    def __call__(self, input_ids, scores, **kwargs):
        for sid in self.stop_ids:
            if len(sid) <= input_ids.size(1) and input_ids[0, -len(sid):].tolist() == sid:
                return True
        return False


def softmax_logprob_for_token(logits_row: torch.Tensor, token_id: int) -> float | None:
    if token_id is None:
        return None
    lr = logits_row.to(torch.float32)
    mx = lr.max()
    ex = torch.exp(lr - mx)
    p = (ex[token_id] / ex.sum()).clamp_min(1e-45)
    return float(torch.log(p).item())


def layer_last_token_norms(hidden_states: list[torch.Tensor]) -> list[float]:
    norms = []
    for h in hidden_states:
        v = h[:, -1, :]
        norms.append(float(torch.linalg.vector_norm(v).item()))
    return norms


def attn_entropy_last_token(attn_step: list[torch.Tensor]) -> list[float]:
    # attn_step: per-layer tensors [B, H, T, T]
    ents = []
    for la in attn_step:
        a = la[0, :, -1, :].clamp_min(1e-12)
        a = a / a.sum(dim=-1, keepdim=True)
        e = -(a * a.log()).sum(dim=-1).mean().item()
        ents.append(float(e))
    return ents


def collect_gate_stats(model) -> dict:
    stats, count, ssum = {}, 0, 0.0
    for name, mod in model.named_modules():
        ga = getattr(mod, "_last_gate_activity", None)
        if torch.is_tensor(ga):
            m = float(ga.mean().item())
            stats[name] = m
            ssum += m
            count += 1
    return {
        "num_gated_modules": count or None,
        "mean_gate_activity": (ssum / count) if count else None,
        "by_module": stats or None,
    }


def count_sections(tokenizer, text: str) -> dict:
    tids = {t: tokenizer.convert_tokens_to_ids(t) for t in ["<think>","</think>","<answer>","</answer>"]}
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
    def spans(open_id, close_id):
        S = []
        i = 0
        while i < len(ids):
            if ids[i] == open_id:
                j = i + 1
                while j < len(ids) and ids[j] != close_id:
                    j += 1
                S.append((i, j))
                i = j + 1
            else:
                i += 1
        return S
    th = spans(tids["<think>"], tids["</think>"])
    an = spans(tids["<answer>"], tids["</answer>"])
    th_len = sum(max(0, e - s - 1) for s, e in th)
    an_len = sum(max(0, e - s - 1) for s, e in an)
    return {"think_tokens": th_len, "answer_tokens": an_len}


def _resolve_base_from_adapter(adapter_dir: Path) -> Path | None:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
        base = cfg.get("base_model_name_or_path")
        if not base:
            return None
        # If base is a relative name (e.g., "Base"), resolve under the model root
        # adapter_dir like model/Tina/checkpoint-2000 → model/<base>
        root = adapter_dir.parents[1] if len(adapter_dir.parents) >= 2 else adapter_dir.parent
        candidate = (root / base)
        return candidate if candidate.exists() else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-root", default="model")
    ap.add_argument("--base", default="Base")
    ap.add_argument("--adapter", default="Tina")
    ap.add_argument("--subfolder", default="checkpoint-2000")
    ap.add_argument("--calibration", default="", help="Optional calibration JSON path")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--visible-cot", type=lambda s: s.lower()=="true", default=False)
    ap.add_argument("--max-new", dest="max_new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--jsonl-out", default="", help="Append JSON line to this file (else print).")
    ap.add_argument("--debug-per-layer", action="store_true", help="Include per-layer diagnostics (alpha, per_layer_plan)")
    ap.add_argument("--log-raw-prompt", action="store_true", help="Include raw prompt (redacted otherwise).")
    args = ap.parse_args()

    rid = str(uuid.uuid4())
    device = ("cuda" if (args.device=="auto" and torch.cuda.is_available()) else
              "mps"  if (args.device=="auto" and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()) else
              args.device if args.device!="auto" else "cpu")

    models_root = Path(args.models_root)
    base_path = models_root / args.base
    adapter_path = models_root / args.adapter / args.subfolder
    tok = AutoTokenizer.from_pretrained(str(base_path), use_fast=True, trust_remote_code=True, local_files_only=True)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(base_path), torch_dtype=dtype, trust_remote_code=True, local_files_only=True)
    model.to(device)
    ensure_reasoning_tokens(tok, model)
    hidden = int(getattr(model.config, 'hidden_size', 2048))
    layers = int(getattr(model.config, 'num_hidden_layers', 24))
    eng = IntrospectiveEngine(
        model=model,
        tokenizer=tok,
        cfg=EngineConfig(
            visible_cot=args.visible_cot,
            budget_cap=256,
            calibration_path=(args.calibration or None),
            linked_all_layers=True,
            proj_dim=128,
            agg='attn',
            dump_per_layer=bool(args.debug_per_layer),
        ),
        hidden_size=hidden,
        num_layers=layers,
    )

    messages = [{"role": "user", "content": args.prompt}]
    t0 = time.time()
    out_text = eng.generate_cot(messages, max_new_tokens=args.max_new, temperature=args.temperature, top_p=args.top_p, repetition_penalty=1.1, ignore_eos=False, stream=False)
    t1 = time.time()
    # Gather stats
    full_text = out_text
    output_text = _extract_answer(full_text, include_think=args.visible_cot)
    sections = count_sections(tok, full_text)

    # Per-step logprobs of chosen tokens (if available)
    logprobs = []
    try:
        in_len = input_ids.shape[1]
        chosen = gen.sequences[0, in_len:]
        for k, logits in enumerate(getattr(gen, "scores", []) or []):
            tok_id = int(chosen[k].item()) if k < chosen.shape[0] else None
            logprobs.append(softmax_logprob_for_token(logits[0], tok_id))
    except Exception:
        logprobs = None

    gate_stats = collect_gate_stats(eng.scaffold)
    stats = getattr(eng, "last_stats", {}) or {}

    model_id = f"{base_path}::{adapter_path}" if adapter_path.exists() else str(base_path)
    rec = {
        "request_id": rid,
        "ts_ms": int(time.time()*1000),
        "model": model_id,
        "device": device,
        "decode": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new},
        "timing": {
            "latency_s": round(t1 - t0, 4),
            "tokens_per_sec": None
        },
        "io": {
            "input_tokens": None,
            "output_tokens": None,
            "visible_cot": args.visible_cot,
            "prompt_preview": args.prompt if args.log_raw_prompt else (args.prompt[:64] + ("..." if len(args.prompt) > 64 else ""))
        },
        "sections": sections,
        "heads": {
            "plan": stats.get("plan_label") or stats.get("plan"),
            "confidence": stats.get("confidence"),
            "budget": stats.get("think_budget"),
            "alpha_summary": stats.get("alpha_summary"),
            "plan_agreement": stats.get("plan_agreement"),
        },
        "activations": {
            "layer_last_token_norms": None,
            "attn_entropies_last_token": None
        },
        "gates": gate_stats,
        "logprobs": None,
        "text": {
            "full": full_text if args.log_raw_prompt or args.visible_cot else None,
            "answer": output_text
        }
    }

    line = json.dumps(rec, ensure_ascii=False)
    out_path = args.jsonl_out
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    else:
        print(line)


if __name__ == "__main__":
    main()
