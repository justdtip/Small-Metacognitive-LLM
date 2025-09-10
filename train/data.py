from typing import List, Dict, Tuple, Any

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

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

