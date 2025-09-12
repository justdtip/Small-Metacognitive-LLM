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


def style_invariance_kl(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask_answer: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric KL consistency over answer positions between two style variants.
    Computes mean over masked positions of KL(p_a || p_b) + KL(p_b || p_a), where p_* = softmax(logits_*).
    Shapes:
      - logits_a/logits_b: (B, T, V)
      - mask_answer: (B, T) boolean/float mask with 1 at answer positions
    Returns scalar tensor.
    """
    assert logits_a.shape == logits_b.shape, "logit tensors must have same shape"
    B, T, V = logits_a.shape
    m = mask_answer.to(dtype=logits_a.dtype)
    # Flatten masked positions
    p_a = torch.softmax(logits_a, dim=-1)
    p_b = torch.softmax(logits_b, dim=-1)
    # Safe logs
    log_pa = torch.log(p_a.clamp_min(1e-8))
    log_pb = torch.log(p_b.clamp_min(1e-8))
    # Per-token KLs
    kl_ab = (p_a * (log_pa - log_pb)).sum(dim=-1)  # (B,T)
    kl_ba = (p_b * (log_pb - log_pa)).sum(dim=-1)
    sym = kl_ab + kl_ba
    denom = m.sum().clamp_min(1.0)
    return (sym * m).sum() / denom


def rewrite_consistency_kl(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask_answer: torch.Tensor,
) -> torch.Tensor:
    """
    Paraphrase rewrite consistency: symmetric KL over answer positions between paraphrase variants.
    Identical to style_invariance_kl but kept distinct for readability.
    """
    return style_invariance_kl(logits_a, logits_b, mask_answer)


def rubric_reg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Simple regression objective for rubric supervision.
    - pred: (B,) predicted score in [0,1] (e.g., from a head or mapped conf_prob)
    - target: (B,) target score in [0,1]; use -1 to indicate missing and will be ignored
    Returns mean squared error over valid targets, or 0 if none valid.
    """
    y = target.view(-1).float()
    x = pred.view(-1).float().clamp(0.0, 1.0)
    mask = (y >= 0.0) & (y <= 1.0)
    if mask.any():
        return torch.mean((x[mask] - y[mask]) ** 2)
    return torch.tensor(0.0, dtype=x.dtype, device=x.device)


def evidence_anchor_loss(present_mask: torch.Tensor) -> torch.Tensor:
    """
    Penalize missing evidence keys.
    - present_mask: (B, K) float/bool where 1 indicates the key appears in the <think> span.
    Loss = mean over batch of (1 - mean(present)) per example.
    """
    if present_mask.numel() == 0:
        return torch.tensor(0.0)
    pm = present_mask.float()
    per_ex = 1.0 - pm.mean(dim=-1)
    return per_ex.mean()


def noise_marker_penalty(
    present: torch.Tensor,
    len_increase: torch.Tensor,
    acc_gain: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize superficial noise markers that increase think length without improving accuracy.
    - present: (B,) 1 if markers were injected/present for the sample, else 0
    - len_increase: (B,) observed increase in think tokens vs baseline (can be 0 if unknown)
    - acc_gain: (B,) change in accuracy (after - before), expected <= 0 under no-gain assumption
    Returns mean(mask * relu(len_increase) * 1[acc_gain<=0]).
    """
    p = present.view(-1).float()
    dL = len_increase.view(-1).float().clamp_min(0.0)
    g = acc_gain.view(-1).float()
    no_gain = (g <= 0).float()
    val = p * dL * no_gain
    return val.mean()


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
    # style invariance: optional pair of logits and answer mask
    style_pair: Optional[Dict[str, torch.Tensor]] = None,
    # rubric/evidence auxiliaries
    rubric_pred: Optional[torch.Tensor] = None,
    rubric_target: Optional[torch.Tensor] = None,
    evidence_present: Optional[torch.Tensor] = None,
    # adversarial wrong-think guard
    think_hidden: Optional[torch.Tensor] = None,
    think_mask: Optional[torch.Tensor] = None,
    wrongthink_labels: Optional[torch.Tensor] = None,
    # paraphrase rewrite consistency
    rewrite_pair: Optional[Dict[str, torch.Tensor]] = None,
    # noise marker robustness
    noise_present: Optional[torch.Tensor] = None,
    noise_len_increase: Optional[torch.Tensor] = None,
    noise_acc_gain: Optional[torch.Tensor] = None,
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
                          "plan_ce": 0.0, "budget_reg": 0.0, "conf_cal": 0.0, "style_inv": 0.0,
                          "rubric": 0.0, "evidence": 0.0, "wrongthink": 0.0, "noise_marker": 0.0,
                          "rewrite_consistency": 0.0}
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

    # Style invariance consistency (optional)
    w_si = float(weights.get("style_inv", 0.0) or 0.0)
    if w_si > 0.0 and style_pair is not None:
        try:
            la = style_pair.get("logits_a")
            lb = style_pair.get("logits_b")
            am = style_pair.get("answer_mask")
            if (la is not None) and (lb is not None) and (am is not None):
                si = style_invariance_kl(la, lb, am)
                out["style_inv"] = si
                loss_total = loss_total + w_si * si
            else:
                out["style_inv"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        except Exception:
            out["style_inv"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
    else:
        out["style_inv"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Rubric supervision (optional)
    w_rub = float(weights.get("rubric", 0.0) or 0.0)
    if w_rub > 0.0 and (rubric_pred is not None) and (rubric_target is not None):
        rub = rubric_reg(rubric_pred, rubric_target)
        out["rubric_loss"] = rub
        loss_total = loss_total + w_rub * rub
    else:
        out["rubric_loss"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Evidence anchoring (optional)
    w_e = float(weights.get("evidence", 0.0) or 0.0)
    if w_e > 0.0 and (evidence_present is not None):
        ev = evidence_anchor_loss(evidence_present)
        out["evidence_loss"] = ev
        loss_total = loss_total + w_e * ev
    else:
        out["evidence_loss"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Wrong-think penalty: predict wrong-think from pooled think hidden, penalize positives
    w_wt = float(weights.get("wrongthink", 0.0) or 0.0)
    if w_wt > 0.0 and (think_hidden is not None) and (think_mask is not None) and (wrongthink_labels is not None):
        try:
            h = think_hidden
            m = think_mask.to(h.dtype).unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1.0)
            pooled = (h * m).sum(dim=1) / denom  # (B,H)
            logits_wt = pooled.mean(dim=-1)  # (B,)
            y = wrongthink_labels.view(-1).to(logits_wt.dtype)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits_wt, y, reduction='none')
            mask_pos = (y > 0.5)
            wt = (bce[mask_pos].mean() if mask_pos.any() else torch.tensor(0.0, dtype=logits_wt.dtype, device=logits_wt.device))
            out["wrongthink_penalty"] = wt
            loss_total = loss_total + w_wt * wt
        except Exception:
            out["wrongthink_penalty"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
    else:
        out["wrongthink_penalty"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Noise marker penalty (optional)
    w_nm = float(weights.get("noise_marker", 0.0) or 0.0)
    if w_nm > 0.0 and (noise_present is not None) and (noise_len_increase is not None) and (noise_acc_gain is not None):
        try:
            nm = noise_marker_penalty(noise_present, noise_len_increase, noise_acc_gain)
            out["noise_marker_penalty"] = nm
            loss_total = loss_total + w_nm * nm
        except Exception:
            out["noise_marker_penalty"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
    else:
        out["noise_marker_penalty"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Rewrite consistency KL (optional)
    w_rc = float(weights.get("rewrite_consistency", 0.0) or 0.0)
    if w_rc > 0.0 and (rewrite_pair is not None):
        try:
            la = rewrite_pair.get("logits_a")
            lb = rewrite_pair.get("logits_b")
            am = rewrite_pair.get("answer_mask")
            if (la is not None) and (lb is not None) and (am is not None):
                rc = rewrite_consistency_kl(la, lb, am)
                out["rewrite_consistency"] = rc
                loss_total = loss_total + w_rc * rc
            else:
                out["rewrite_consistency"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        except Exception:
            out["rewrite_consistency"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
    else:
        out["rewrite_consistency"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    out["total"] = loss_total
    return out

# Training-time context for Quiet-Star auxiliary consistency
_QUIET_STAR_CTX: ContextVar[Optional[Dict[str, Any]]] = ContextVar("QUIET_STAR_CTX", default=None)
