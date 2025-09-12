from typing import List, Dict, Tuple, Any, Optional, Callable

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from tina.tokenizer_utils import segment_and_masks


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

        # Build rewrite groups (list[list[int]]) in sample order
        groups_map: Dict[int, List[int]] = {}
        for i, rp in enumerate(rewrite_pair_ids):
            if rp is None:
                continue
            rid = int(rp)
            if rid >= 0:
                groups_map.setdefault(rid, []).append(i)
        rewrite_groups = [idxs for _, idxs in groups_map.items()]

        batch_out["batch_meta"] = {"rewrite_groups": rewrite_groups}
        return batch_out

    return _collate
