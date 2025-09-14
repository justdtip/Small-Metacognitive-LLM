# tina/tokenizer_utils.py
from typing import Iterable, Tuple, Dict, List, Optional
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

REASONING_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]
DECOMP_TOKENS = ["<plan>", "</plan>", "<exec>", "</exec>", "<eval>", "</eval>"]
# Core closing tags used as stop sequences by default; service config may add more
STOP_SEQUENCES = ["</think>", "</answer>"]
# Register both reasoning and decomposition tags as atomic
SPECIAL_TOKENS = REASONING_TOKENS + DECOMP_TOKENS

def ensure_reasoning_tokens(tokenizer, model=None, extra: Iterable[str] | None = None) -> Dict[str, int]:
    """
    Ensure reasoning tags are present as atomic (non-splitting) tokens.
    - Registers as additional_special_tokens to guarantee single-token encoding.
    - Resizes model embeddings if a model is provided.
    - Asserts round-trip atomicy: encode(tag, add_special_tokens=False) has length==1 for each tag.
    """
    # Build full token list to register as atomic
    tokens_to_add: List[str] = list(SPECIAL_TOKENS)
    if extra:
        for t in extra:
            if t not in tokens_to_add:
                tokens_to_add.append(str(t))

    # Add as additional special tokens (idempotent: tokenizer merges duplicates)
    try:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    except Exception:
        # Fallback: older tokenizers
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)

    # Also guard against splitting during decoding/tokenization
    if hasattr(tokenizer, "unique_no_split_tokens"):
        for t in tokens_to_add:
            if t not in tokenizer.unique_no_split_tokens:
                tokenizer.unique_no_split_tokens.append(t)

    # Resolve IDs
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in tokens_to_add}
    # Log and assert IDs for telemetry & CI guardrails
    if any(v is None for v in ids.values()):
        raise ValueError(f"Special tag ID resolution failed: {ids}")
    # TODO: emit a digest for parity logs via caller's logger if available

    # Round-trip atomicity check (only if tokenizer supports encoding calls)
    has_encode = hasattr(tokenizer, "encode") or callable(getattr(tokenizer, "__call__", None))
    if has_encode:
        for t in tokens_to_add:
            try:
                tok_ids = tokenizer.encode(t, add_special_tokens=False)
            except Exception:
                tok_ids = tokenizer(t, add_special_tokens=False).input_ids
            if not isinstance(tok_ids, list):
                tok_ids = list(tok_ids)
            # Fail fast if any tag is not atomic
            if len(tok_ids) != 1:
                raise ValueError(f"Tag '{t}' splits into {tok_ids}. Check tokenizer rules.")

    # Additional guardrail using tokenizer.tokenize() if available
    if hasattr(tokenizer, "tokenize") and callable(getattr(tokenizer, "tokenize")):
        for t in tokens_to_add:
            try:
                pieces = tokenizer.tokenize(t)
                if isinstance(pieces, (list, tuple)) and len(pieces) != 1:
                    raise ValueError(f"Tag '{t}' tokenizes into {pieces} (len={len(pieces)}); expected a single atomic token.")
            except Exception:
                # If tokenize is not reliable for this tokenizer, skip; encode check above still enforces atomicity
                pass

    # Resize embeddings and optionally initialize new rows if a model is passed
    if model is not None and hasattr(model, "get_input_embeddings") and torch is not None:
        old = model.get_input_embeddings().weight.data.clone()
        model.resize_token_embeddings(len(tokenizer))
        emb = model.get_input_embeddings().weight

        # Seed init from em-dash and a few neutral tokens
        seeds: List[int] = []
        for probe in ["—", "-", ".", ","]:
            try:
                toks = tokenizer(probe, add_special_tokens=False).input_ids
            except Exception:
                toks = tokenizer.encode(probe, add_special_tokens=False)
            if toks:
                seeds.extend(toks)
        if not seeds and hasattr(tokenizer, "eos_token_id"):
            seeds = [tokenizer.eos_token_id]

        if seeds and torch is not None:
            with torch.no_grad():
                avg = old[seeds].mean(dim=0)
                for t in tokens_to_add:
                    tid = ids[t]
                    if tid is not None and tid < emb.shape[0]:
                        emb[tid].copy_(avg)

    return ids

def get_reasoning_tag_ids(tokenizer) -> Dict[str, int]:
    """
    Return mapping from each reasoning/decomposition tag to its token id.
    Ensures tags are registered and atomic first.
    """
    ensure_reasoning_tokens(tokenizer)
    out: Dict[str, int] = {}
    for t in SPECIAL_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is None:
            raise ValueError(f"Failed to resolve id for tag '{t}' after registration")
        out[t] = int(tid)
    return out

def format_chat(tokenizer, messages: List[Dict], add_generation_prompt: bool = True):
    """
    Apply chat template deterministically across server/CLI.
    Returns: (input_ids: torch.LongTensor[1,T], attention_mask: torch.LongTensor[1,T], prompt_text: Optional[str])
    """
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
            tag = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}.get(role, "<|user|>")
            content = m.get("content") or ""
            parts.append(f"{tag}\n{content}\n")
        parts.append("<|assistant|>\n")
        prompt_text = "".join(parts)
        enc = tokenizer(prompt_text, return_tensors="pt")
        return enc.input_ids, enc.attention_mask, prompt_text

def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)

def detok(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)

def find_span(input_ids: List[int], open_id: int, close_id: int) -> Optional[Tuple[int, int]]:
    """
    Find the first non-overlapping span delimited by open_id and close_id in input_ids.
    Returns (start_index, end_index) inclusive of the tag tokens or None if not found or malformed.
    """
    try:
        start = input_ids.index(open_id)
    except ValueError:
        return None
    try:
        end = input_ids.index(close_id, start + 1)
    except ValueError:
        return None
    if end <= start:
        return None
    return (start, end)

def segment_and_masks(text: str, tokenizer, loss_on: str = "answer") -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    """
    Tokenize a single think+answer string and derive segmentation masks.
    Returns: (input_ids, attention_mask, loss_mask, think_mask, answer_mask)
    - think_mask marks tokens strictly inside <think>...</think>
    - answer_mask marks tokens strictly inside <answer>...</answer>
    - loss_mask defaults to answer_mask (can be configured)
    """
    ids_map = ensure_reasoning_tokens(tokenizer)
    # Encode without adding extra specials
    try:
        input_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        input_ids = tokenizer(text, add_special_tokens=False).input_ids
    if not isinstance(input_ids, list):
        input_ids = list(input_ids)

    think_open, think_close = ids_map["<think>"], ids_map["</think>"]
    ans_open, ans_close = ids_map["<answer>"], ids_map["</answer>"]

    def find_idx(seq: List[int], token_id: int, start: int = 0) -> int:
        try:
            return seq.index(token_id, start)
        except ValueError:
            return -1

    # Locate spans (first matching pairs)
    t0 = find_idx(input_ids, think_open)
    t1 = find_idx(input_ids, think_close, t0 + 1) if t0 != -1 else -1
    a0 = find_idx(input_ids, ans_open, (t1 + 1) if t1 != -1 else 0)
    a1 = find_idx(input_ids, ans_close, a0 + 1) if a0 != -1 else -1

    L = len(input_ids)
    attention_mask = [1] * L
    think_mask = [0] * L
    answer_mask = [0] * L

    if t0 != -1 and t1 != -1 and t1 > t0:
        for i in range(t0 + 1, t1):
            think_mask[i] = 1
    if a0 != -1 and a1 != -1 and a1 > a0:
        for i in range(a0 + 1, a1):
            answer_mask[i] = 1

    if loss_on == "answer":
        loss_mask = answer_mask[:]
    elif loss_on == "think":
        loss_mask = think_mask[:]
    elif loss_on == "both":
        loss_mask = [1 if (think_mask[i] or answer_mask[i]) else 0 for i in range(L)]
    else:
        loss_mask = [0] * L

    return input_ids, attention_mask, loss_mask, think_mask, answer_mask
