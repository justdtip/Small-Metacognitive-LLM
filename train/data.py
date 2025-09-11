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
      - 'plan_class': int | None
      - 'target_budget': int | None
      - 'correct': 0/1 | None
    """

    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples or []

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def make_collate_fn(
    tokenizer,
    *,
    loss_on: str = "answer",
    plan_buckets: Tuple[int, int] = (32, 128),
    budget_alpha: float = 1.0,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Factory for a collate function that tokenizes 'text', builds masks via segment_and_masks,
    pads/stacks, and attaches supervision labels as batch tensors/lists.
    Returns a function suitable for DataLoader(collate_fn=...).
    """

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        items: List[Tuple[List[int], List[int], List[int], List[int], List[int]]] = []
        plan_targets: List[int] = []
        target_budget: List[int] = []
        correctness: List[int] = []
        plan_srcs: List[str] = []
        budget_srcs: List[str] = []
        think_lens: List[int] = []
        diff_bins: List[int] = []

        for ex in batch:
            text = (ex.get("text") or "").strip()
            ids, attn, loss_m, think_m, ans_m = segment_and_masks(text, tokenizer, loss_on=loss_on)
            items.append((ids, attn, loss_m, think_m, ans_m))
            # Think length (tokens) from mask
            # Prefer on-policy decode count if provided on the example
            th_len = int(ex.get("think_tokens_used")) if (ex.get("think_tokens_used") is not None) else int(sum(think_m))
            think_lens.append(th_len)

            # Plan class with provenance
            p = ex.get("plan_class")
            if p is None:
                lo, hi = plan_buckets
                if th_len <= int(lo):
                    p = 0
                elif th_len <= int(hi):
                    p = 1
                else:
                    p = 2
                plan_srcs.append("heuristic")
            else:
                plan_srcs.append(str(ex.get("plan_src") or "gold"))
            plan_targets.append(int(p))

            # Budget target with provenance
            b = ex.get("target_budget")
            if b is None:
                b = int(max(1, round(float(budget_alpha) * th_len)))
                budget_srcs.append("heuristic")
            else:
                budget_srcs.append(str(ex.get("budget_src") or "gold"))
            target_budget.append(int(b))

            # Correctness label (optional); -1 if unknown
            c = ex.get("correct")
            correctness.append(int(c) if c is not None else -1)

            # Difficulty bin (optional); default -1 if unknown
            db = ex.get("difficulty_bin", None)
            diff_bins.append(int(db) if db is not None else -1)

        batch_dict = pad_and_stack(items, pad_id=getattr(tokenizer, "pad_token_id", 0) or 0)

        if torch is not None:
            batch_dict["plan_targets"] = torch.tensor(plan_targets, dtype=torch.long)
            batch_dict["target_budget"] = torch.tensor(target_budget, dtype=torch.long)
            batch_dict["correctness"] = torch.tensor(correctness, dtype=torch.long)
            batch_dict["difficulty_bin"] = torch.tensor(diff_bins, dtype=torch.long)
        else:  # pragma: no cover
            batch_dict["plan_targets"] = plan_targets
            batch_dict["target_budget"] = target_budget
            batch_dict["correctness"] = correctness
            batch_dict["difficulty_bin"] = diff_bins
        # Attach provenance and diagnostics
        batch_dict["plan_src"] = plan_srcs
        batch_dict["budget_src"] = budget_srcs
        batch_dict["think_tokens_used"] = think_lens
        return batch_dict

    return _collate

def pad_and_stack(items: List[Tuple[List[int], List[int], List[int], List[int], List[int]]], pad_id: int = 0) -> Dict[str, Any]:
    """
    Pad a batch of (input_ids, attention_mask, loss_mask, think_mask, answer_mask) to equal length and stack.
    Returns dict with keys: input_ids, attention_mask, loss_mask, think_mask, answer_mask.
    If torch is available, returns Long/Bool tensors; otherwise lists.
    """
    if not items:
        return {"input_ids": [], "attention_mask": [], "loss_mask": [], "think_mask": [], "answer_mask": []}
    max_len = max(len(x[0]) for x in items)
    def _pad(seq: List[int], v: int, L: int) -> List[int]:
        return seq + [v] * (L - len(seq))
    ids_batch: List[List[int]] = []
    attn_batch: List[List[int]] = []
    loss_batch: List[List[int]] = []
    think_batch: List[List[int]] = []
    ans_batch: List[List[int]] = []
    for inp, attn, loss, think, ans in items:
        ids_batch.append(_pad(inp, pad_id, max_len))
        attn_batch.append(_pad(attn, 0, max_len))
        loss_batch.append(_pad(loss, 0, max_len))
        think_batch.append(_pad(think, 0, max_len))
        ans_batch.append(_pad(ans, 0, max_len))
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
