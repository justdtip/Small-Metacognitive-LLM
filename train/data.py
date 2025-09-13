"""
JSONL schema and strictness (collator contract)
-----------------------------------------------

Each record MUST include a 'text' field containing exactly one
"<think> ... </think> <answer> ... </answer>" pair. These core tags are
strictly enforced when strict=True; the collator raises ValueError with the
record index if either pair is missing or malformed. This preserves stable
segmentation and prevents leakage.

Optional decomposition tags (<plan>/<exec>/<eval>) may appear inside the
<think> segment. The collator derives soft boolean masks for any present
sub‑segments and attaches them to the batch as plan_mask, exec_mask,
eval_mask (same length as input_ids). Malformed or missing sub‑segments do
not raise errors, even in strict mode; they are simply ignored (all‑zero
mask). Stopping and masking for training remain based on </think> and
</answer> only by default.
"""

from typing import List, Dict, Tuple, Any, Optional, Callable, Iterator
import os

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from tina.tokenizer_utils import segment_and_masks, get_reasoning_tag_ids, find_span


class ReasoningDataset:
    """
    Minimal in-memory dataset emitting reasoning samples with optional supervision.
    Each item is a dict with keys:
      - 'text': '<think>..</think> <answer>..</answer>'
      - optional: 'plan_class', 'target_budget', 'correct', 'difficulty_bin', 'style_tag', 'rewrite_pair_id'
    """

    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples or []

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


# ---- Glaive reasoning dataset loader -------------------------------------------
try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore


class GlaiveDataset:
    """
    Stream or load the glaiveai/reasoning-v1-20m dataset and transform each record
    to Tina's expected {'text': '<prompt> <think>..</think> <answer>..</answer>'} format.
    The upstream dataset has fields: {'prompt': str, 'response': str}, where response
    contains a '<think>...</think>' block followed by the final answer text.
    """
    def __init__(self, split: str = 'train', path: Optional[str] = None, streaming: bool = False):
        if load_dataset is None:
            raise RuntimeError("datasets is not installed; cannot load glaive dataset")
        kwargs: Dict[str, Any] = {}
        self.split = split
        if path:
            kwargs['data_files'] = {split: path}
        self.ds = load_dataset('glaiveai/reasoning-v1-20m', split=split, streaming=bool(streaming), **kwargs)
        self.streaming = bool(streaming)

    def __len__(self) -> int:  # pragma: no cover - streaming may not support len
        try:
            return len(self.ds)  # type: ignore[arg-type]
        except Exception:
            return 0

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for rec in self.ds:  # type: ignore[operator]
            prompt = (rec.get('prompt') or '').strip()
            resp = (rec.get('response') or '').strip()
            # find closing </think>
            end = resp.find('</think>')
            if end != -1:
                reasoning = resp[:end + len('</think>')].strip()
                answer = resp[end + len('</think>'):].strip()
                text = f"{prompt} {reasoning} <answer> {answer} </answer>"
            else:
                text = f"{prompt} <answer> {resp} </answer>"
            yield {'text': text}


def pad_and_stack(
    items: List[Tuple[List[int], List[int], List[int], List[int], List[int]]],
    *,
    pad_id: int = 0,
) -> Dict[str, Any]:
    """
    Pad a list of (ids, attn, loss_mask, think_mask, answer_mask) to equal length and stack.
    Returns tensors if torch is available; else lists.
    """
    if not items:
        return {
            "input_ids": [] if torch is None else torch.empty(0, dtype=torch.long),
            "attention_mask": [] if torch is None else torch.empty(0, dtype=torch.long),
            "loss_mask": [] if torch is None else torch.empty(0, dtype=torch.long),
            "think_mask": [] if torch is None else torch.empty(0, dtype=torch.long),
            "answer_mask": [] if torch is None else torch.empty(0, dtype=torch.long),
        }
    L = max(len(x[0]) for x in items)
    ids_batch: List[List[int]] = []
    attn_batch: List[List[int]] = []
    loss_batch: List[List[int]] = []
    think_batch: List[List[int]] = []
    ans_batch: List[List[int]] = []
    for (ids, attn, loss_m, think_m, ans_m) in items:
        ids_p = ids + [pad_id] * (L - len(ids))
        attn_p = attn + [0] * (L - len(attn))
        loss_p = loss_m + [0] * (L - len(loss_m))
        think_p = think_m + [0] * (L - len(think_m))
        ans_p = ans_m + [0] * (L - len(ans_m))
        ids_batch.append(ids_p)
        attn_batch.append(attn_p)
        loss_batch.append(loss_p)
        think_batch.append(think_p)
        ans_batch.append(ans_p)

    if torch is None:
        return {
            "input_ids": ids_batch,
            "attention_mask": attn_batch,
            "loss_mask": loss_batch,
            "think_mask": think_batch,
            "answer_mask": ans_batch,
        }
    return {
        "input_ids": torch.tensor(ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attn_batch, dtype=torch.long),
        "loss_mask": torch.tensor(loss_batch, dtype=torch.long),
        "think_mask": torch.tensor(think_batch, dtype=torch.long),
        "answer_mask": torch.tensor(ans_batch, dtype=torch.long),
    }


def make_collate_fn(
    tokenizer,
    *,
    loss_on: str = "answer",
    plan_buckets: Tuple[int, int] = (32, 128),
    budget_alpha: float = 1.0,  # reserved for future use
    strict: bool = True,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Factory for a collate function that tokenizes 'text', builds masks via segment_and_masks,
    pads/stacks, and attaches supervision labels as batch tensors/lists.
    Returns a function suitable for DataLoader(collate_fn=...).
    """

    def _has_exact_one_pair(s: str, open_t: str, close_t: str) -> bool:
        return s.count(open_t) == 1 and s.count(close_t) == 1 and (s.find(open_t) < s.rfind(close_t))

    def _heuristic_budget(diff_bin: Optional[int]) -> int:
        # Simple curriculum mapping for tests: 0->16, 1->64, 2->128 (clamped)
        if diff_bin is None:
            return 64
        table = [16, 64, 128]
        i = max(0, min(int(diff_bin), len(table) - 1))
        return table[i]

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        items: List[Tuple[List[int], List[int], List[int], List[int], List[int]]] = []
        plan_targets: List[int] = []
        target_budget: List[int] = []
        correctness: List[int] = []
        style_ids: List[int] = []
        style_tags: List[str] = []
        rewrite_pair_ids: List[int] = []

        # Style tag mapping per batch (stable order)
        style_map: Dict[str, int] = {}
        next_style_id = 0

        def _first_span(s: str, open_t: str, close_t: str) -> Optional[str]:
            i = s.find(open_t)
            j = s.find(close_t, i + len(open_t)) if i != -1 else -1
            if i != -1 and j != -1 and j > i:
                return s[i + len(open_t):j]
            return None

        def _wrap_minimally(s: str) -> str:
            # Build a minimal well-formed sample using the first found spans, else wrap whole text as answer.
            th = _first_span(s, "<think>", "</think>")
            an = _first_span(s, "<answer>", "</answer>")
            th_s = (th or "").strip()
            if an is None:
                # If no explicit answer, treat original content (sans any think span) as answer body
                body = s
                if th is not None:
                    body = s.replace(f"<think>{th}</think>", " ")
                an_s = body.strip()
            else:
                an_s = an.strip()
            return f"<think> {th_s} </think> <answer> {an_s} </answer>"

        for idx, ex in enumerate(batch):
            text = str(ex.get("text", ""))
            ok1 = _has_exact_one_pair(text, "<think>", "</think>")
            ok2 = _has_exact_one_pair(text, "<answer>", "</answer>")
            if strict:
                if not (ok1 and ok2):
                    raise ValueError(f"Malformed tags in record {idx}: missing or extra <think>/<answer> pairs")
            else:
                if not (ok1 and ok2):
                    text = _wrap_minimally(text)

            ids, attn, loss_m, think_m, ans_m = segment_and_masks(text, tokenizer, loss_on=loss_on)
            items.append((ids, attn, loss_m, think_m, ans_m))

            # Optional decomposition masks within <think>
            try:
                tag_ids = get_reasoning_tag_ids(tokenizer)
                tpair = find_span(ids, tag_ids["<think>"], tag_ids["</think>"])
                L = len(ids)
                plan_mask = [0] * L
                exec_mask = [0] * L
                eval_mask = [0] * L
                any_decomp = False
                if tpair is not None:
                    t0, t1 = tpair
                    # Precedence plan > exec > eval
                    occupied = [0] * L
                    def apply_span(open_tag: str, close_tag: str, target_mask: List[int]):
                        nonlocal any_decomp
                        op = tag_ids.get(open_tag); cl = tag_ids.get(close_tag)
                        if op is None or cl is None:
                            return
                        sp = find_span(ids, op, cl)
                        if sp is None:
                            return
                        s, e = sp
                        # exclude tag tokens, clamp within think span
                        s = max(s + 1, t0 + 1)
                        e = min(e - 1, t1 - 1)
                        if e < s:
                            return
                        for i in range(s, e + 1):
                            if not occupied[i]:
                                target_mask[i] = 1
                                occupied[i] = 1
                                any_decomp = True
                    apply_span("<plan>", "</plan>", plan_mask)
                    apply_span("<exec>", "</exec>", exec_mask)
                    apply_span("<eval>", "</eval>", eval_mask)
                # Stash on the example so we can pad after stacking
                ex["_plan_mask"] = plan_mask
                ex["_exec_mask"] = exec_mask
                ex["_eval_mask"] = eval_mask
                ex["_has_decomp"] = any_decomp
            except Exception as e:
                # Ignore malformed sub-segments; do not raise even in strict mode.
                # Optional debug warning via env toggle (DATA_DEBUG=1)
                if os.environ.get("DATA_DEBUG"):
                    print(f"[decomp-mask] warning: record {idx} decomposition parse ignored: {type(e).__name__}: {e}")
                ex["_has_decomp"] = False

            # Labels / targets (use -1 when missing)
            plan_targets.append(int(ex.get("plan_class", -1)) if ex.get("plan_class") is not None else -1)

            if ex.get("target_budget") is not None:
                target_budget.append(int(ex.get("target_budget")))
            else:
                db = ex.get("difficulty_bin")
                db_val = int(db) if db is not None else None
                target_budget.append(_heuristic_budget(db_val))

            correctness.append(int(ex.get("correct", -1)) if ex.get("correct") is not None else -1)

            tag = ex.get("style_tag")
            if isinstance(tag, str) and tag:
                if tag not in style_map:
                    style_map[tag] = next_style_id
                    next_style_id += 1
                style_ids.append(style_map[tag])
                style_tags.append(tag)
            else:
                style_ids.append(-1)
                style_tags.append("")

            rp = ex.get("rewrite_pair_id")
            rewrite_pair_ids.append(int(rp) if rp is not None else -1)

        # Stack
        pad_id = 0
        try:
            pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
        except Exception:
            pad_id = 0
        batch_out = pad_and_stack(items, pad_id=pad_id)

        # Attach padded decomposition masks if any present
        try:
            if any((ex.get("_has_decomp") for ex in batch)):
                Lp = int(batch_out["input_ids"].shape[1]) if torch is not None else len(batch_out["input_ids"][0])
                def _pad_mask(m: List[int]) -> List[int]:
                    return m + [0] * (Lp - len(m))
                plan_ms = []; exec_ms = []; eval_ms = []
                for ex in batch:
                    pm = _pad_mask(ex.get("_plan_mask", [0] * len(items[0][0])))
                    em = _pad_mask(ex.get("_exec_mask", [0] * len(items[0][0])))
                    vm = _pad_mask(ex.get("_eval_mask", [0] * len(items[0][0])))
                    plan_ms.append(pm); exec_ms.append(em); eval_ms.append(vm)
                if torch is None:
                    batch_out["plan_mask"] = plan_ms
                    batch_out["exec_mask"] = exec_ms
                    batch_out["eval_mask"] = eval_ms
                else:
                    device = None
                    batch_out["plan_mask"] = torch.tensor(plan_ms, dtype=torch.long, device=device)
                    batch_out["exec_mask"] = torch.tensor(exec_ms, dtype=torch.long, device=device)
                    batch_out["eval_mask"] = torch.tensor(eval_ms, dtype=torch.long, device=device)
        except Exception:
            pass

        # Attach label tensors
        if torch is None:
            batch_out["plan_targets"] = plan_targets
            batch_out["target_budget"] = target_budget
            batch_out["correctness"] = correctness
            batch_out["style_id"] = style_ids
            batch_out["style_tag"] = style_tags
            batch_out["rewrite_pair_id"] = rewrite_pair_ids
        else:
            device = None
            batch_out["plan_targets"] = torch.tensor(plan_targets, dtype=torch.long, device=device)
            batch_out["target_budget"] = torch.tensor(target_budget, dtype=torch.long, device=device)
            batch_out["correctness"] = torch.tensor(correctness, dtype=torch.long, device=device)
            batch_out["style_id"] = torch.tensor(style_ids, dtype=torch.long, device=device)
            batch_out["style_tag"] = style_tags
            batch_out["rewrite_pair_id"] = torch.tensor(rewrite_pair_ids, dtype=torch.long, device=device)

        # Build rewrite groups (list[list[int]]) and explicit pair map (list[tuple[int,int]])
        groups_map: Dict[int, List[int]] = {}
        for i, rp in enumerate(rewrite_pair_ids):
            if rp is None:
                continue
            rid = int(rp)
            if rid >= 0:
                groups_map.setdefault(rid, []).append(i)
        rewrite_groups = [idxs for _, idxs in groups_map.items()]
        pair_map: List[Tuple[int, int]] = []
        for g in rewrite_groups:
            if isinstance(g, (list, tuple)) and len(g) >= 2:
                for a in range(len(g)):
                    for b in range(a + 1, len(g)):
                        pair_map.append((int(g[a]), int(g[b])))

        batch_out["batch_meta"] = {"rewrite_groups": rewrite_groups, "rewrite_pair_map": pair_map}
        return batch_out

    return _collate
