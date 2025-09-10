#!/usr/bin/env python3
"""
tina_chat_introspective.py — additive introspection + CoT budget serving

Does NOT modify your base or existing Tina adapter. All extras are side modules with gates=0 by default.
"""

import argparse, os, sys, time, json, datetime
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import set_peft_model_state_dict

from tina.tokenizer_utils import ensure_reasoning_tokens
from tina.serve import IntrospectiveEngine, EngineConfig

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZER_PARALLELISM", "false")

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

def pick_dtype(name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]

def load_base_and_adapter(models_root, base, adapter, subfolder, dtype):
    base_path = Path(models_root) / base
    adapter_path = Path(models_root) / adapter / subfolder
    if not base_path.exists():
        sys.exit(f"Missing base dir: {base_path}")
    if not adapter_path.exists():
        sys.exit(f"Missing adapter dir: {adapter_path}")

    model = AutoModelForCausalLM.from_pretrained(
        str(base_path), device_map="auto", torch_dtype=dtype, trust_remote_code=True, local_files_only=True
    )
    tok = AutoTokenizer.from_pretrained(str(base_path), use_fast=True, trust_remote_code=True, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Load Tina adapter (existing)
    import inspect as _inspect
    from peft import LoraConfig, TaskType
    with (adapter_path / "adapter_config.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # sanitize
    sig = _inspect.signature(LoraConfig.__init__)
    allowed = {k for k in sig.parameters if k not in {"self", "**kwargs"}}
    clean = {k: v for k, v in raw.items() if k in allowed}
    t = clean.get("task_type", raw.get("task_type", "CAUSAL_LM"))
    if isinstance(t, str):
        clean["task_type"] = getattr(TaskType, t, TaskType.CAUSAL_LM)
    elif not isinstance(t, TaskType):
        clean["task_type"] = TaskType.CAUSAL_LM
    clean.setdefault("r", 32)
    clean.setdefault("lora_alpha", 128)
    clean.setdefault("lora_dropout", 0.05)
    clean.setdefault("bias", "none")
    clean.setdefault("inference_mode", True)

    model = get_peft_model(model, LoraConfig(**clean))
    # weights
    w_st = adapter_path / "adapter_model.safetensors"
    w_pt = adapter_path / "adapter_model.bin"
    if w_st.exists():
        if not safe_load_file:
            sys.exit("Install safetensors or provide adapter_model.bin")
        sd = safe_load_file(str(w_st), device="cpu")
    elif w_pt.exists():
        sd = torch.load(str(w_pt), map_location="cpu")
    else:
        sys.exit(f"Missing adapter weights in {adapter_path}")
    set_peft_model_state_dict(model, sd)
    model.eval()
    return model, tok

def infer_hidden_and_layers(base_config_path: Path):
    with (base_config_path / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return int(cfg["hidden_size"]), int(cfg["num_hidden_layers"])

def parse_args():
    p = argparse.ArgumentParser()
    # accept hyphen/underscore forms for convenience
    p.add_argument("--models-root", "--models_root", dest="models_root", default="/workspace/model")
    p.add_argument("--base", default="Base")
    p.add_argument("--adapter", default="Tina")
    p.add_argument("--subfolder", default="checkpoint-2000")
    p.add_argument("--system", default="You are Tina: concise, helpful, and safe.")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", "--repetition_penalty", dest="repetition_penalty", type=float, default=1.1)
    p.add_argument("--ignore-eos", "--ignore_eos", dest="ignore_eos", action="store_true")
    p.add_argument("--no-stream", "--no_stream", dest="no_stream", action="store_true")
    # logging
    p.add_argument("--log-jsonl", "--log_jsonl", dest="log_jsonl", default="",
                   help="Path to JSONL log of turns (prompts, outputs, token counts, timings).")
    p.add_argument("--log-transcript", "--log_transcript", dest="log_transcript", default="",
                   help="Path to plain-text transcript log.")
    p.add_argument("--log-raw", "--log_raw", dest="log_raw", action="store_true",
                   help="Do not redact logs; include raw text and prompt.")
    p.add_argument("--redact-logs", dest="redact_logs", action="store_true", help="Redact PII in logs (default).")
    p.add_argument("--no-redact-logs", dest="redact_logs", action="store_false", help="Disable PII redaction.")
    p.set_defaults(redact_logs=True)
    p.add_argument("--log-max-bytes", dest="log_max_bytes", type=int, default=10_000_000,
                   help="Rotate log files when exceeding this size (bytes). Default 10MB.")
    p.add_argument("--log-backups", dest="log_backups", type=int, default=3,
                   help="Number of rotated backups to keep. Default 3.")
    # reproducibility
    p.add_argument("--seed", type=int, default=-1, help="Seed for RNGs. If -1 and --no-seed not set, uses 42.")
    p.add_argument("--no-seed", dest="no_seed", action="store_true", help="Do not set seeds for this run.")
    # introspection flags
    p.add_argument("--visible-cot", action="store_true", help="Show <think> content in output")
    p.add_argument("--budget-cap", type=int, default=256)
    p.add_argument("--no-dynamic-budget", action="store_true")
    return p.parse_args()

def iso_now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

def _roll_file(path: Path, max_bytes: int, backups: int):
    try:
        if not path.exists() or max_bytes <= 0:
            return
        if path.stat().st_size < max_bytes:
            return
        # rotate: file -> .1, .1->.2, ...
        for i in range(backups, 0, -1):
            src = path.with_suffix(path.suffix + f".{i}") if i > 0 else path
            dst = path.with_suffix(path.suffix + f".{i+1}")
            if i == backups and dst.exists():
                try: dst.unlink()
                except Exception: pass
            if (i == backups) or (not src.exists()):
                continue
            try:
                src.rename(dst)
            except Exception:
                pass
        # move current to .1
        try:
            path.rename(path.with_suffix(path.suffix + ".1"))
        except Exception:
            pass
    except Exception:
        pass

def write_jsonl(path: str, obj: dict, *, rotate: tuple[int,int] | None = None):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if rotate:
        _roll_file(p, rotate[0], rotate[1])
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_transcript(path: str, text: str, *, rotate: tuple[int,int] | None = None):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if rotate:
        _roll_file(p, rotate[0], rotate[1])
    with p.open("a", encoding="utf-8") as f:
        f.write(text)

def redact_text(text: str) -> str:
    """Light PII redaction for logs: emails and digit runs."""
    try:
        import re
        if not text:
            return text
        # mask emails
        text = re.sub(r"[\w\.-]+@[\w\.-]+", "[redacted-email]", text)
        # mask digit sequences (>=4 digits)
        text = re.sub(r"\d{4,}", lambda m: "X" * len(m.group(0)), text)
        return text
    except Exception:
        return text

def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
        attn = torch.ones_like(encoded)
        return encoded, attn, None
    except Exception:
        # Fallback simple role tags
        parts = []
        for m in messages:
            role = m.get("role", "user")
            tag = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}[role]
            content = m.get("content") or ""
            parts.append(f"{tag}\n{content}\n")
        parts.append("<|assistant|>\n")
        prompt_text = "".join(parts)
        enc = tokenizer(prompt_text, return_tensors="pt")
        return enc.input_ids, enc.attention_mask, prompt_text

def main():
    args = parse_args()
    dtype = pick_dtype(args.dtype)

    # Seeding
    if not args.no_seed:
        seed = args.seed if args.seed is not None and args.seed >= 0 else 42
        try:
            import random, numpy as np
            random.seed(seed)
            np.random.seed(seed)
        except Exception:
            pass
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Best-effort determinism; may impact perf
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

    model, tok = load_base_and_adapter(args.models_root, args.base, args.adapter, args.subfolder, dtype)
    hidden, layers = infer_hidden_and_layers(Path(args.models_root) / args.base)

    engine = IntrospectiveEngine(
        model=model,
        tokenizer=tok,
        cfg=EngineConfig(
            visible_cot=args.visible_cot,
            max_think_tokens=args.budget_cap,
            use_dynamic_budget=not args.no_dynamic_budget,
            budget_cap=args.budget_cap,
            side_rank=8,
            adapter_layers=None,   # all
            taps=[6, 10, 14],
        ),
        hidden_size=hidden,
        num_layers=layers,
    )

    # Prepare loggers
    loggers = {"jsonl": args.log_jsonl or "", "transcript": args.log_transcript or ""}

    messages = [{"role": "system", "content": args.system}]
    print("Tina (introspective) ready. Type your message, /help for commands.\n")

    while True:
        try:
            text = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n/exit")
            break

        if text in {"/exit", "exit", "quit", ":q"}:
            break
        if text == "/help":
            print("Commands: /reset | /system <text> | /visible | /hidden | /save <path> | /exit")
            continue
        if text == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("↺ conversation reset.")
            continue
        if text.startswith("/system "):
            args.system = text[len("/system "):].strip() or args.system
            messages = [{"role": "system", "content": args.system}]
            print("✓ system updated and conversation reset.")
            continue
        if text.startswith("/save "):
            out_path = Path(text[len("/save "):].strip() or "conversation.jsonl")
            with out_path.open("w", encoding="utf-8") as f:
                for m in messages:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            print(f"✓ saved to {out_path}")
            continue
        if text == "/visible":
            engine.cfg.visible_cot = True; print("✓ visible CoT")
            continue
        if text == "/hidden":
            engine.cfg.visible_cot = False; print("✓ hidden CoT")
            continue

        messages.append({"role": "user", "content": text})
        t0 = time.time()

        # Build prompt once to log input token count
        enc_ids, enc_attn, prompt_text = apply_chat_template(tok, messages, add_generation_prompt=True)
        input_ids = enc_ids.to(model.device)
        input_tok = int(input_ids.shape[1])

        out = engine.generate_cot(
            messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            ignore_eos=args.ignore_eos,
            stream=not args.no_stream,
        )
        messages.append({"role": "assistant", "content": out})
        if args.no_stream:
            print(out)
        # Logging similar to local tool
        ts = iso_now()
        assistant_text = out
        out_tok = None
        try:
            out_tok = len(tok(assistant_text, add_special_tokens=False).input_ids)
        except Exception:
            pass
        # Compose log record (redacted unless --log-raw)
        system_text = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        do_redact = bool(args.redact_logs) and not bool(args.log_raw)
        user_log = redact_text(text) if do_redact else text
        asst_log = redact_text(assistant_text) if do_redact else assistant_text
        sys_log = redact_text(system_text) if do_redact else system_text
        stats = getattr(engine, "last_stats", {}) or {}
        rec = {
            "ts": ts,
            "system": sys_log,
            "user": user_log,
            "assistant": asst_log,
            "input_tokens": input_tok,
            "output_tokens": out_tok,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "dtype": str(getattr(model, "dtype", None)),
            "ignore_eos": args.ignore_eos,
            "stop_reason": "length",
            "elapsed_sec": round(time.time()-t0, 3),
            "think_budget": stats.get("think_budget"),
            "plan": stats.get("plan"),
            "confidence": stats.get("confidence"),
        }
        if args.log_raw:
            rec["prompt_text"] = prompt_text
        # Add request correlation + throughput
        try:
            import uuid as _uuid
            rid = str(_uuid.uuid4())
        except Exception:
            rid = ""
        if out_tok and (time.time()-t0) > 0:
            rec["tokens_per_sec"] = round(out_tok / max(1e-6, (time.time()-t0)), 3)
        rec["request_id"] = rid
        rec["visible_cot"] = bool(engine.cfg.visible_cot)
        write_jsonl(loggers["jsonl"], rec, rotate=(args.log_max_bytes, args.log_backups))

        transcript_line = (
            f"[{ts}] SYSTEM: {sys_log}\n"
            f"[{ts}] USER: {user_log}\n"
            f"[{ts}] ASSISTANT: {asst_log}\n\n"
        )
        write_transcript(loggers["transcript"], transcript_line, rotate=(args.log_max_bytes, args.log_backups))

        print(f"[took {time.time()-t0:.2f}s]")

if __name__ == "__main__":
    main()
