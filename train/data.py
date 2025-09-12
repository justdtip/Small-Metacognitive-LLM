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
    strict: bool = True,
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
        style_ids: List[int] = []
        style_tags: List[str] = []
        rubric_scores: List[float] = []
        evidence_meta: List[List[str]] = []
        wrongthink_labels: List[float] = []

        def _has_exact_one_pair(s: str, open_t: str, close_t: str) -> bool:
            return s.count(open_t) == 1 and s.count(close_t) == 1 and (s.find(open_t) < s.rfind(close_t))

        for idx, ex in enumerate(batch):
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
