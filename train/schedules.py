from __future__ import annotations
import torch
from contextlib import contextmanager
from typing import Optional, Dict, Callable
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


def build_loss_schedules(
    total_steps: int,
    *,
    warmup_steps: int = 0,
    final_weights: Optional[Dict[str, float]] = None,
) -> Callable[[int], Dict[str, float]]:
    """
    Construct a schedule function that returns auxiliary loss weights per step.
    - Warm up with main CE (aux=0) for 'warmup_steps', then linearly anneal to 'final_weights' by 'total_steps'.
    - Keys typically include: 'plan_ce', 'budget_reg', 'conf_cal'.
    """
    fw = final_weights or {"plan_ce": 0.5, "budget_reg": 0.1, "conf_cal": 0.1}
    warmup = max(0, int(warmup_steps))
    total = max(1, int(total_steps))

    def weights_for(step: int) -> Dict[str, float]:
        s = max(0, int(step))
        if s <= warmup:
            return {k: 0.0 for k in fw}
        # progress from warmup..total → 0..1
        prog = min(1.0, (s - warmup) / max(1.0, (total - warmup)))
        return {k: float(v) * prog for k, v in fw.items()}

    return weights_for


def quiet_star_schedule(total_steps: int, *, start: float = 0.1, end: float = 0.0, end_ratio: float = 0.5):
    """
    Anneal Quiet-Star aux_mix from 'start' to 'end' by step = total_steps * end_ratio.
    Returns a function f(step)->float.
    """
    total = max(1, int(total_steps))
    end_step = int(max(0, min(1.0, float(end_ratio))) * total)
    s0, s1 = float(start), float(end)

    def f(step: int) -> float:
        s = max(0, int(step))
        if s >= end_step:
            return s1
        prog = s / max(1.0, float(end_step))
        return s0 + (s1 - s0) * prog

    return f
