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

Usage:
  python tools/observe_infer.py --prompt "Factor 12345" --visible-cot true --jsonl logs/obs.jsonl
"""
import argparse, json, time, uuid, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

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

try:
    from tina.metacog_heads import MetacogHeads
except Exception:
    MetacogHeads = None  # optional


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--visible-cot", type=lambda s: s.lower()=="true", default=False)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--jsonl", default="", help="Append JSON line to this file (else print).")
    ap.add_argument("--log-raw-prompt", action="store_true", help="Include raw prompt (redacted otherwise).")
    args = ap.parse_args()

    rid = str(uuid.uuid4())
    device = ("cuda" if (args.device=="auto" and torch.cuda.is_available()) else
              "mps"  if (args.device=="auto" and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()) else
              args.device if args.device!="auto" else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)

    # Ensure tags exist and resize embeddings if added
    ensure_reasoning_tokens(tok, model)

    seed = args.prompt.strip()
    seed = (seed + "\n<think>\n") if args.visible_cot else (seed + "\n<think>\n</think><answer>")
    input_ids = tok(seed, return_tensors="pt").input_ids.to(model.device)

    stop = StoppingCriteriaList([StopOnTags(tok, stop_strs=("</answer>",))])

    t0 = time.time()
    gen = model.generate(
        input_ids=input_ids,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        output_scores=True,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_attentions=True,
        stopping_criteria=stop
    )
    t1 = time.time()

    full_text = tok.decode(gen.sequences[0], skip_special_tokens=False)
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

    # Hidden states for last step (fallback: post-forward)
    last_hs = None
    try:
        if gen.hidden_states and isinstance(gen.hidden_states, tuple):
            step_hs = gen.hidden_states[-1]  # tuple per layer
            if isinstance(step_hs, tuple):
                last_hs = [h.detach().to("cpu") for h in step_hs]
    except Exception:
        last_hs = None
    if last_hs is None:
        with torch.no_grad():
            out = model(gen.sequences, output_hidden_states=True)
        last_hs = [h.detach().to("cpu") for h in out.hidden_states]

    norms = layer_last_token_norms(last_hs)

    # Attention entropies (optional)
    attn_entropies = None
    try:
        if getattr(gen, "attentions", None):
            attn_entropies = attn_entropy_last_token(gen.attentions[-1])
    except Exception:
        attn_entropies = None

    # Heads (optional)
    heads_out = None
    if MetacogHeads is not None:
        try:
            heads = MetacogHeads(hidden_size=model.config.hidden_size, taps=(6,10,14), plan_k=3)
            plan_logits, budget_cap, conf_logit = heads(last_hs)
            heads_out = {
                "plan_logits": [float(x) for x in plan_logits[0].detach().cpu().tolist()],
                "budget_cap": float(budget_cap[0,0].detach().cpu().item()),
                "confidence_logit": float(conf_logit[0,0].detach().cpu().item()),
                "confidence_sigmoid": float(torch.sigmoid(conf_logit)[0,0].detach().cpu().item()),
            }
        except Exception as e:
            heads_out = {"error": f"metacog_heads_failed: {e.__class__.__name__}"}

    gate_stats = collect_gate_stats(model)

    rec = {
        "request_id": rid,
        "ts_ms": int(time.time()*1000),
        "model": args.model,
        "device": device,
        "decode": {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens},
        "timing": {
            "latency_s": round(t1 - t0, 4),
            "tokens_per_sec": round((gen.sequences.shape[1] - input_ids.shape[1]) / max(1e-6, (t1 - t0)), 2)
        },
        "io": {
            "input_tokens": int(input_ids.shape[1]),
            "output_tokens": int(gen.sequences.shape[1] - input_ids.shape[1]),
            "visible_cot": args.visible_cot,
            "prompt_preview": args.prompt if args.log_raw_prompt else (args.prompt[:64] + ("..." if len(args.prompt) > 64 else ""))
        },
        "sections": sections,
        "heads": heads_out,
        "activations": {
            "layer_last_token_norms": norms,
            "attn_entropies_last_token": attn_entropies
        },
        "gates": gate_stats,
        "logprobs": logprobs,
        "text": {
            "full": full_text if args.log_raw_prompt or args.visible_cot else None,
            "answer": output_text
        }
    }

    line = json.dumps(rec, ensure_ascii=False)
    if args.jsonl:
        Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.jsonl, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    else:
        print(line)


if __name__ == "__main__":
    main()
