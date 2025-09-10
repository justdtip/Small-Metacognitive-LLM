#!/usr/bin/env python3
"""
tina_chat_local.py — offline back-and-forth chat (Base + Tina adapter), with logging

Key features:
- Local load from /workspace/model/{Base,Tina}/...
- PEFT config sanitizer (avoids 'eva_config' etc. errors)
- Streaming + JSONL logging of prompts & outputs (+ token counts & timings)
- Accepts hyphen_or_underscore CLI variants
- Optional --ignore-eos to avoid EOS early stop

Example:
python tina_chat_local.py \
  --models-root /workspace/model \
  --base Base \
  --adapter Tina \
  --subfolder checkpoint-2000 \
  --system "You are a logic machine. You solve logic puzzles." \
  --max-new-tokens 8192 \
  --log-jsonl /workspace/tina_chat.log.jsonl \
  --log-transcript /workspace/tina_chat.transcript.txt
"""

import argparse, json, os, sys, time, inspect, datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import get_peft_model, LoraConfig, TaskType
from peft.utils import set_peft_model_state_dict

# Fully offline by default
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZER_PARALLELISM", "false")

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # accept hyphen/underscore forms for convenience
    p.add_argument("--models-root", "--models_root", dest="models_root", default="/workspace/model")
    p.add_argument("--base", default="Base")
    p.add_argument("--adapter", default="Tina")
    p.add_argument("--subfolder", default="checkpoint-2000")
    p.add_argument("--system", default="You are Tina: concise, helpful, and safe.")
    p.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", "--repetition_penalty", dest="repetition_penalty", type=float, default=1.1)
    p.add_argument("--no-stream", "--no_stream", dest="no_stream", action="store_true")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--ignore-eos", "--ignore_eos", dest="ignore_eos", action="store_true",
                   help="Do not set eos_token_id so generation won't stop early on EOS.")
    # Logging
    p.add_argument("--log-jsonl", "--log_jsonl", dest="log_jsonl", default="",
                   help="Path to JSONL log of turns (prompts, outputs, token counts, timings).")
    p.add_argument("--log-transcript", "--log_transcript", dest="log_transcript", default="",
                   help="Path to plain-text transcript log.")
    return p.parse_args()


# ------------------------- Utilities ---------------------------
def pick_dtype(name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def assert_exists(path: Path, what: str):
    if not path.exists():
        sys.exit(f"✗ Missing {what}: {path}")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sanitize_lora_kwargs(raw_cfg: dict) -> dict:
    """Keep only kwargs that installed peft.LoraConfig accepts. Normalize task_type."""
    import inspect as _inspect
    sig = _inspect.signature(LoraConfig.__init__)
    allowed = {k for k in sig.parameters if k not in {"self", "**kwargs"}}
    clean = {k: v for k, v in raw_cfg.items() if k in allowed}

    t = clean.get("task_type", raw_cfg.get("task_type", "CAUSAL_LM"))
    if isinstance(t, str):
        clean["task_type"] = getattr(TaskType, t, TaskType.CAUSAL_LM)
    elif not isinstance(t, TaskType):
        clean["task_type"] = TaskType.CAUSAL_LM

    clean.setdefault("r", 32)
    clean.setdefault("lora_alpha", 128)
    clean.setdefault("lora_dropout", 0.05)
    clean.setdefault("bias", "none")
    clean.setdefault("inference_mode", True)
    return clean


def iso_now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def write_jsonl(path: str, obj: dict):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_transcript(path: str, text: str):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(text)


# --------------- Load base + adapter (PEFT-safe) ---------------
def load_models(args):
    root = Path(args.models_root)
    base_path = root / args.base
    adapter_root = root / args.adapter
    adapter_path = adapter_root / args.subfolder

    assert_exists(base_path, "base model directory")
    assert_exists(adapter_root, "adapter directory")
    assert_exists(adapter_path, "adapter checkpoint directory")

    dtype = pick_dtype(args.dtype)
    print(f"[load] base={base_path}\n[load] adapter={adapter_path}\n[load] dtype={dtype} device_map=auto")

    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_path),
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    tok = AutoTokenizer.from_pretrained(
        str(base_path),
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # PEFT config (sanitized)
    cfg_path = adapter_path / "adapter_config.json"
    assert_exists(cfg_path, "adapter_config.json")
    raw_cfg = _load_json(cfg_path)
    lora_cfg = LoraConfig(**_sanitize_lora_kwargs(raw_cfg))

    model = get_peft_model(base_model, lora_cfg)

    # Load adapter weights
    weights_path_safetensors = adapter_path / "adapter_model.safetensors"
    weights_path_pt = adapter_path / "adapter_model.bin"

    if weights_path_safetensors.exists():
        if safe_load_file is None:
            sys.exit("✗ 'safetensors' not installed. `pip install safetensors` or provide adapter_model.bin")
        state_dict = safe_load_file(str(weights_path_safetensors), device="cpu")
    elif weights_path_pt.exists():
        state_dict = torch.load(str(weights_path_pt), map_location="cpu")
    else:
        sys.exit(f"✗ Missing adapter weights: {weights_path_safetensors} or {weights_path_pt}")

    set_peft_model_state_dict(model, state_dict)
    model.eval()
    return model, tok


# -------------------- Prompt building utils --------------------
def apply_chat_template(tokenizer, messages, add_generation_prompt=True):
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt"
        )
        attn = torch.ones_like(encoded)
        return encoded, attn, None  # template text not easily available from tokenizer here
    except Exception:
        # Fallback formatting (simple role tags)
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


def num_tokens(tensor_ids):
    # tensor_ids is shape [1, T]
    return int(tensor_ids.shape[1])


# ------------------------- Generation --------------------------
def generate(model, tokenizer, messages, args, loggers):
    # Build prompt
    input_ids, attention_mask, prompt_text = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    input_tok = num_tokens(input_ids)

    eos_token_id = None if args.ignore_eos else getattr(tokenizer, "eos_token_id", None)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0.0,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    start = time.time()
    stop_reason = "length"

    if args.no_stream:
        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        gen_tokens = out[0, input_ids.shape[1]:]
        assistant_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        out_tok = int(gen_tokens.shape[0])
        # Detect EOS stop if applicable
        if not args.ignore_eos and eos_token_id is not None and out[0, -1].item() == eos_token_id:
            stop_reason = "eos_token"
    else:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        import threading
        chunks = []

        def _worker():
            with torch.no_grad():
                model.generate(**gen_kwargs)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        for piece in streamer:
            sys.stdout.write(piece)
            sys.stdout.flush()
            chunks.append(piece)
        print()

        assistant_text = "".join(chunks).strip()
        # token count (supported in newer HF)
        out_tok = None
        try:
            # streamer.generated_ids is a list of sequences per batch
            if hasattr(streamer, "generated_ids") and streamer.generated_ids:
                out_tok = len(streamer.generated_ids[0])
        except Exception:
            pass

        # We can't directly read the last token id from streamer; infer reason best-effort
        if out_tok is not None and out_tok < args.max_new_tokens:
            stop_reason = "eos_or_other"

    elapsed = time.time() - start

    # ---------- Logging ----------
    ts = iso_now()
    user_text = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""

    # JSONL log
    write_jsonl(loggers["jsonl"], {
        "ts": ts,
        "system": messages[0]["content"] if messages and messages[0]["role"] == "system" else "",
        "user": user_text,
        "assistant": assistant_text,
        "input_tokens": input_tok,
        "output_tokens": out_tok,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "dtype": str(model.dtype) if hasattr(model, "dtype") else None,
        "ignore_eos": args.ignore_eos,
        "stop_reason": stop_reason,
        "elapsed_sec": round(elapsed, 3),
        # If we have the raw templated prompt text, include it (fallback path only)
        "prompt_text": prompt_text,
    })

    # Transcript log (plain text)
    transcript_line = (
        f"[{ts}] SYSTEM: {messages[0]['content'] if messages and messages[0]['role']=='system' else ''}\n"
        f"[{ts}] USER: {user_text}\n"
        f"[{ts}] ASSISTANT: {assistant_text}\n\n"
    )
    write_transcript(loggers["transcript"], transcript_line)

    return assistant_text


# ---------------------------- Main -----------------------------
def main():
    args = parse_args()
    model, tok = load_models(args)

    # Prepare loggers
    loggers = {"jsonl": args.log_jsonl or "", "transcript": args.log_transcript or ""}

    messages = [{"role": "system", "content": args.system}]
    print("Tina chat (local) ready. Type your message, or /help for commands.\n")

    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n/exit")
            break

        if not user:
            continue

        if user == "/help":
            print("Commands: /reset | /system <text> | /save <path> | /exit")
            continue
        if user == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print("↺ conversation reset.")
            continue
        if user.startswith("/system "):
            args.system = user[len("/system "):].strip() or args.system
            messages = [{"role": "system", "content": args.system}]
            print("✓ system prompt updated and conversation reset.")
            continue
        if user.startswith("/save "):
            out_path = Path(user[len("/save "):].strip() or "conversation.jsonl")
            with out_path.open("w", encoding="utf-8") as f:
                for m in messages:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            print(f"✓ saved to {out_path}")
            continue
        if user in ("/exit", "exit", "quit", ":q"):
            break

        messages.append({"role": "user", "content": user})
        t0 = time.time()
        assistant = generate(model, tok, messages, args, loggers)
        messages.append({"role": "assistant", "content": assistant})
        print(f"[took {time.time()-t0:.2f}s]")


if __name__ == "__main__":
    main()
