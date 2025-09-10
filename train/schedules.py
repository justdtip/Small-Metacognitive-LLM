from __future__ import annotations
import torch
from contextlib import contextmanager
from typing import Optional
from train.losses import _QUIET_STAR_CTX

def linear_anneal(step: int, total: int, start: float, end: float) -> float:
    if total <= 0:
        return end
    t = max(0.0, min(1.0, step / float(total)))
    return float(start + (end - start) * t)

def unfreeze_top_n(model, n: int):
    if n <= 0:
        return
    # naive: unfreeze last n modules' parameters
    mods = list(model.modules())
    seen = 0
    for m in reversed(mods):
        for p in getattr(m, 'parameters', lambda: [])():
            p.requires_grad = True
        seen += 1
        if seen >= n:
            break

@contextmanager
def quiet_star_context(think_mask: torch.Tensor, *, tau: float = 2.0, sample_ratio: float = 0.5):
    """
    Enable the training-only Quiet-Star auxiliary consistency loss on a sampled subset of think tokens.
    No API changes to loss calls; compute_losses() will read this context.
    """
    tok = _QUIET_STAR_CTX.set({"mask": think_mask, "tau": float(tau), "sample_ratio": float(sample_ratio)})
    try:
        yield
    finally:
        _QUIET_STAR_CTX.reset(tok)
