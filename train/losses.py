from __future__ import annotations
from typing import Iterable, Optional, Dict, Any
from contextvars import ContextVar
import torch

def gate_sparsity_regularizer(modules: Iterable[torch.nn.Module], weight: float = 1e-4) -> torch.Tensor:
    """
    L0-like sparsity: penalize average gate activity across provided modules.
    Looks for attribute `_last_gate_activity` or falls back to sigmoid(log_alpha).
    Returns a scalar tensor.
    """
    total = 0.0
    count = 0
    for m in modules:
        if hasattr(m, "_last_gate_activity") and getattr(m, "_last_gate_activity") is not None:
            val = float(getattr(m, "_last_gate_activity").item())
            total += val; count += 1
        elif hasattr(m, "hc_gate") and hasattr(getattr(m, "hc_gate"), "log_alpha"):
            la = getattr(m.hc_gate, "log_alpha")
            with torch.no_grad():
                total += torch.sigmoid(la).item(); count += 1
    if count == 0:
        return torch.tensor(0.0)
    avg = total / count
    return torch.tensor(avg * weight)


def compute_losses(logits: torch.Tensor, labels: torch.Tensor, *, gate_modules=None, weights=None):
    """
    Compute masked answer CE and optional regularizers.
    - logits: (B,T,V)
    - labels: (B,T) with -100 where ignored
    - gate_modules: iterable of modules with _last_gate_activity (optional)
    - weights: dict with keys like 'answer_ce', 'gate_reg'
    Returns dict with {'answer_ce','gate_reg','total'} where missing parts default to 0.
    """
    weights = weights or {"answer_ce": 1.0, "gate_reg": 0.0, "aux_mix": 0.0}
    loss_total = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    out = {}
    # Flatten for CE
    B, T, V = logits.shape
    ce = torch.nn.functional.cross_entropy(
        logits.view(B * T, V), labels.view(B * T), ignore_index=-100
    )
    out["answer_ce"] = ce
    loss_total = loss_total + weights.get("answer_ce", 1.0) * ce

    if gate_modules is not None and weights.get("gate_reg", 0.0) > 0:
        greg = gate_sparsity_regularizer(gate_modules, weight=weights.get("gate_reg", 0.0))
        out["gate_reg"] = greg
        loss_total = loss_total + greg
    else:
        out["gate_reg"] = torch.tensor(0.0)

    # Optional Quiet-Star auxiliary consistency loss (training-only)
    # Blends self-distillation on a sampled subset of think tokens, encouraging robustness to short 'silent thoughts'.
    aux_w = float(weights.get("aux_mix", 0.0) or 0.0)
    if aux_w > 0.0:
        cfg = _QUIET_STAR_CTX.get()
        if cfg is not None:
            mask = cfg.get("mask")  # [B,T] float/bool
            tau = float(cfg.get("tau", 2.0))
            sample_ratio = float(cfg.get("sample_ratio", 0.5))
            if isinstance(mask, torch.Tensor) and mask.numel() == logits.shape[0] * logits.shape[1]:
                B, T, V = logits.shape
                # sample subset of think tokens
                m = mask.to(logits.dtype)
                if sample_ratio < 1.0:
                    samp = (torch.rand_like(m) < sample_ratio).to(logits.dtype)
                    m = m * samp
                # teacher and student distributions
                with torch.no_grad():
                    p_teacher = torch.softmax(logits / max(tau, 1e-6), dim=-1)
                p_student = torch.log_softmax(logits, dim=-1)
                # per-position KL: sum p * (log p - log q)
                kl = (p_teacher * (torch.log(p_teacher + 1e-8) - p_student)).sum(dim=-1)  # [B,T]
                # Mask to sampled tokens
                denom = m.sum().clamp_min(1.0)
                aux_loss = (kl * m).sum() / denom
                out["aux_loss"] = aux_loss
                loss_total = loss_total + aux_w * aux_loss
            else:
                out["aux_loss"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        else:
            out["aux_loss"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
    else:
        out["aux_loss"] = torch.tensor(0.0)

    out["total"] = loss_total
    return out

# Training-time context for Quiet-Star auxiliary consistency
_QUIET_STAR_CTX: ContextVar[Optional[Dict[str, Any]]] = ContextVar("QUIET_STAR_CTX", default=None)
