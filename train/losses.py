from __future__ import annotations
from typing import Iterable, Optional, Dict, Any
from contextvars import ContextVar
import torch

# ---- Metacog auxiliary objectives -------------------------------------------------

def plan_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy over discrete plan classes.
    - logits: (N, K) or (..., K)
    - targets: (N,) or matching leading dims; int64 in [0..K-1]
    Returns a scalar mean loss.
    """
    if logits.ndim > 2:
        K = logits.shape[-1]
        logits = logits.view(-1, K)
        targets = targets.view(-1)
    return torch.nn.functional.cross_entropy(logits, targets)


def budget_reg(pred_budget: torch.Tensor, target_budget: torch.Tensor, *, huber_delta: float = 10.0,
               kind: str = "huber", quantile_alpha: float = 0.5) -> torch.Tensor:
    """
    Regression penalty for budget prediction.
    - Huber (default): robust to outliers; delta controls transition.
    - Quantile: asymmetric penalty for over/under budgeting.
    Shapes: pred and target should be broadcastable; reduced by mean.
    """
    pred = pred_budget.float()
    target = target_budget.float()
    if kind == "quantile":
        # Pinball (quantile) loss: max(alpha*e, (alpha-1)*e)
        e = target - pred
        a = float(quantile_alpha)
        return torch.maximum(a * e, (a - 1.0) * e).mean()
    # Default Huber (SmoothL1)
    return torch.nn.functional.smooth_l1_loss(pred, target, beta=float(huber_delta))


def confidence_cal(logits: torch.Tensor, labels: torch.Tensor, *, loss: str = "brier") -> torch.Tensor:
    """
    Calibration-oriented loss for a binary confidence signal.
    - logits: (N,1) or (N,)
    - labels: (N,1) or (N,) in {0,1}
    Returns a scalar mean loss using either Brier or NLL (BCE with logits).
    """
    x = logits.view(-1).float()
    y = labels.view(-1).float().clamp(0.0, 1.0)
    if loss.lower() == "nll":
        return torch.nn.functional.binary_cross_entropy_with_logits(x, y)
    # Brier as default on sigmoid probabilities
    p = torch.sigmoid(x)
    return torch.mean((p - y) ** 2)

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


def quiet_star_loss(
    logits: torch.Tensor,
    *,
    mask: torch.Tensor,
    tau: float = 2.0,
    sample_ratio: float = 0.5,
) -> torch.Tensor:
    """
    Quiet-Star style auxiliary consistency loss over think positions.
    Uses temperature-smoothed teacher (no grad) vs student log-softmax KL on logits at positions selected by 'mask'.
    Returns a scalar loss averaged over selected tokens.
    """
    B, T, V = logits.shape
    m = mask.to(logits.dtype)
    if sample_ratio < 1.0:
        samp = (torch.rand_like(m) < float(sample_ratio)).to(logits.dtype)
        m = m * samp
    with torch.no_grad():
        p_teacher = torch.softmax(logits / max(float(tau), 1e-6), dim=-1)
    p_student = torch.log_softmax(logits, dim=-1)
    kl = (p_teacher * (torch.log(p_teacher + 1e-8) - p_student)).sum(dim=-1)  # [B,T]
    denom = m.sum().clamp_min(1.0)
    return (kl * m).sum() / denom


def compute_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    gate_modules=None,
    weights=None,
    # metacog auxiliaries (optional)
    plan_logits: Optional[torch.Tensor] = None,
    plan_targets: Optional[torch.Tensor] = None,
    budget_pred: Optional[torch.Tensor] = None,
    budget_target: Optional[torch.Tensor] = None,
    conf_logits: Optional[torch.Tensor] = None,
    conf_labels: Optional[torch.Tensor] = None,
):
    """
    Compute masked answer CE and optional regularizers/auxiliaries.
    - logits: (B,T,V)
    - labels: (B,T) with -100 where ignored
    - gate_modules: iterable of modules with _last_gate_activity (optional)
    - weights: dict with keys like 'answer_ce','gate_reg','aux_mix','plan_ce','budget_reg','conf_cal'
    - plan_logits/targets: shapes (N,K) and (N,) for discrete plan classes
    - budget_pred/target: regression tensors (broadcastable)
    - conf_logits/labels: binary confidence logits and {0,1} labels
    Returns dict with {'answer_ce','gate_reg','plan_ce','budget_reg','conf_cal','aux_loss','total'} (zeros if not used).
    """
    weights = weights or {"answer_ce": 1.0, "gate_reg": 0.0, "aux_mix": 0.0,
                          "plan_ce": 0.0, "budget_reg": 0.0, "conf_cal": 0.0}
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
                aux_loss = quiet_star_loss(logits, mask=mask, tau=tau, sample_ratio=sample_ratio)
                out["aux_loss"] = aux_loss
                loss_total = loss_total + aux_w * aux_loss
            else:
                out["aux_loss"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        else:
            out["aux_loss"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
    else:
        out["aux_loss"] = torch.tensor(0.0)

    # --- Metacog auxiliaries ---
    # Plan CE
    if (plan_logits is not None) and (plan_targets is not None) and weights.get("plan_ce", 0.0) > 0:
        pce = plan_ce(plan_logits, plan_targets)
        out["plan_ce"] = pce
        loss_total = loss_total + float(weights.get("plan_ce", 0.0)) * pce
    else:
        out["plan_ce"] = torch.tensor(0.0)

    # Budget regression (Huber default)
    if (budget_pred is not None) and (budget_target is not None) and weights.get("budget_reg", 0.0) > 0:
        breg = budget_reg(budget_pred, budget_target)
        out["budget_reg"] = breg
        loss_total = loss_total + float(weights.get("budget_reg", 0.0)) * breg
    else:
        out["budget_reg"] = torch.tensor(0.0)

    # Confidence calibration
    if (conf_logits is not None) and (conf_labels is not None) and weights.get("conf_cal", 0.0) > 0:
        ccal = confidence_cal(conf_logits, conf_labels, loss="brier")
        out["conf_cal"] = ccal
        loss_total = loss_total + float(weights.get("conf_cal", 0.0)) * ccal
    else:
        out["conf_cal"] = torch.tensor(0.0)

    out["total"] = loss_total
    return out

# Training-time context for Quiet-Star auxiliary consistency
_QUIET_STAR_CTX: ContextVar[Optional[Dict[str, Any]]] = ContextVar("QUIET_STAR_CTX", default=None)
