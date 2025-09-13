from __future__ import annotations
from typing import Iterable, Optional, Dict, Any
from contextvars import ContextVar
import torch

# ---- Metacog auxiliary objectives -------------------------------------------------

def plan_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy over discrete plan classes with safe masking for invalid targets.
    - logits: (N, K) or (..., K)
    - targets: (N,) or matching leading dims; may include negatives for missing labels.
    Returns a scalar mean loss over valid targets; 0.0 if none valid.
    """
    if logits.ndim > 2:
        K = int(logits.shape[-1])
        logits = logits.view(-1, K)
        targets = targets.view(-1)
    else:
        K = int(logits.shape[-1])
    # Ensure long dtype for class indices
    tgt = targets.to(dtype=torch.long)
    valid = (tgt >= 0) & (tgt < K)
    if not torch.any(valid):
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    return torch.nn.functional.cross_entropy(logits[valid], tgt[valid])


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
    answer_mask_a: torch.Tensor,
    answer_mask_b: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    Paraphrase/rewrites answer-consistency on answer tokens only.
    - Extract answer positions using masks for each sample
    - Truncate to the minimum length to align positions across the pair
    - Compute symmetric KL per position across vocab with softmax(logits/tau) and average over positions
    Returns a scalar tensor.
    """
    # Normalize shapes
    if logits_a.dim() == 3:
        logits_a = logits_a.squeeze(0)
    if logits_b.dim() == 3:
        logits_b = logits_b.squeeze(0)
    if answer_mask_a.dim() == 2:
        answer_mask_a = answer_mask_a.squeeze(0)
    if answer_mask_b.dim() == 2:
        answer_mask_b = answer_mask_b.squeeze(0)

    idx_a = (answer_mask_a > 0).nonzero(as_tuple=False).view(-1)
    idx_b = (answer_mask_b > 0).nonzero(as_tuple=False).view(-1)
    if idx_a.numel() == 0 or idx_b.numel() == 0:
        return torch.tensor(0.0, dtype=logits_a.dtype, device=logits_a.device)
    L = int(min(idx_a.numel(), idx_b.numel()))
    if L <= 0:
        return torch.tensor(0.0, dtype=logits_a.dtype, device=logits_a.device)
    idx_a = idx_a[:L]
    idx_b = idx_b[:L]
    la = logits_a.index_select(dim=0, index=idx_a)
    lb = logits_b.index_select(dim=0, index=idx_b)
    tau = max(float(tau), 1e-6)
    pa = torch.softmax(la / tau, dim=-1)
    pb = torch.softmax(lb / tau, dim=-1)
    log_pa = torch.log(pa.clamp_min(1e-8))
    log_pb = torch.log(pb.clamp_min(1e-8))
    kl_ab = (pa * (log_pa - log_pb)).sum(dim=-1)  # (L,)
    kl_ba = (pb * (log_pb - log_pa)).sum(dim=-1)
    sym = kl_ab + kl_ba
    return sym.mean()


def masked_symmetric_kl(p_logits: torch.Tensor, q_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Symmetric KL divergence over vocabulary, masked over token positions.
    - p_logits, q_logits: (T,V)
    - mask: (T,) float/bool in {0,1}; only positions with mask==1 contribute.
    Returns scalar mean over masked positions.
    """
    if p_logits.dim() == 3:
        p_logits = p_logits.squeeze(0)
    if q_logits.dim() == 3:
        q_logits = q_logits.squeeze(0)
    T, V = p_logits.shape
    m = mask.view(-1).to(dtype=p_logits.dtype)
    # log-softmax for numerical stability
    lp = torch.log_softmax(p_logits, dim=-1)
    lq = torch.log_softmax(q_logits, dim=-1)
    p = torch.exp(lp)
    q = torch.exp(lq)
    kl_pq = (p * (lp - lq)).sum(dim=-1)  # (T,)
    kl_qp = (q * (lq - lp)).sum(dim=-1)
    skl = 0.5 * (kl_pq + kl_qp)
    denom = m.sum().clamp_min(1.0)
    return (skl * m).sum() / denom


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
    # batch-level masks and grouping for rewrite KL
    answer_mask: Optional[torch.Tensor] = None,
    rewrite_groups: Optional[list[list[int]]] = None,
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
    # per-layer diagnostics (optional)
    per_layer: Optional[Dict[str, torch.Tensor]] = None,
    # on-policy + think auxiliaries
    think_mask_tce: Optional[torch.Tensor] = None,
    think_ce_w: float = 0.0,
    # over-budget penalty
    think_tokens_used: Optional[torch.Tensor] = None,
    # alias for clarity in some call-sites and new synonym
    used_think_tokens: Optional[torch.Tensor] = None,
    used_tokens: Optional[torch.Tensor] = None,
    target_budget: Optional[torch.Tensor] = None,
    budget_penalty_w: float = 0.0,
    # length shaping (CoT-space inspired)
    target_L_opt: Optional[torch.Tensor] = None,
    target_length_bin: Optional[Any] = None,
    target_bin: Optional[Any] = None,
    solution_incomplete: Optional[torch.Tensor] = None,
    # correctness/verifier shaping
    correct_labels: Optional[torch.Tensor] = None,
    verifier_logits: Optional[torch.Tensor] = None,
    # expert diversity aux from heads
    aux: Optional[Dict[str, Any]] = None,
    # optional anchor KL to base logits on a small anchor set
    anchor_logits: Optional[torch.Tensor] = None,
    anchor_mask: Optional[torch.Tensor] = None,
    anchor_weight: float = 0.0,
    # quiet-star override (phase-driven)
    quiet_star_w: float = 0.0,
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
                          "var_reg": 0.0,
                          # new shaping weights
                          "correct": 0.0, "len_over": 0.0, "len_under": 0.0, "expert_entropy_reg": 0.0,
                          "rewrite_consistency": 0.0}
    loss_total = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    out = {}
    # Flatten for CE with safety: mask any labels outside [0, V-1]
    B, T, V = logits.shape
    lab = labels.to(dtype=torch.long, device=logits.device)
    invalid = (lab >= V) | (lab < 0)
    if invalid.any():
        lab = lab.masked_fill(invalid, -100)
    ce = torch.nn.functional.cross_entropy(
        logits.view(B * T, V), lab.view(B * T), ignore_index=-100
    )
    out["answer_ce"] = ce
    loss_total = loss_total + weights.get("answer_ce", 1.0) * ce

    if gate_modules is not None and weights.get("gate_reg", 0.0) > 0:
        greg = gate_sparsity_regularizer(gate_modules, weight=weights.get("gate_reg", 0.0))
        out["gate_reg"] = greg
        loss_total = loss_total + greg
    else:
        out["gate_reg"] = torch.tensor(0.0)

    # Correctness / verifier shaping
    w_corr = float(weights.get("correct", 0.0) or 0.0)
    if w_corr > 0.0 and (correct_labels is not None or verifier_logits is not None):
        try:
            if verifier_logits is not None and correct_labels is not None:
                x = verifier_logits.view(-1).float()
                y = correct_labels.view(-1).float().clamp(0.0, 1.0)
                lc = torch.nn.functional.binary_cross_entropy_with_logits(x, y)
            else:
                y = correct_labels.view(-1).float()
                lc = 1.0 - y.mean()
        except Exception:
            lc = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        out["loss_correct"] = lc
        loss_total = loss_total + w_corr * lc
    else:
        out["loss_correct"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Optional Quiet-Star auxiliary consistency loss (training-only)
    # Blends self-distillation on a sampled subset of think tokens, encouraging robustness to short 'silent thoughts'.
    aux_w = float(weights.get("aux_mix", 0.0) or 0.0)
    # Allow phase override via quiet_star_w
    if quiet_star_w > 0.0:
        aux_w = float(quiet_star_w)
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

    # Rewrite consistency KL (groups preferred; legacy pair path retained)
    w_rc = float(weights.get("rewrite_consistency", 0.0) or 0.0)
    rewrite_kl_val = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
    if w_rc > 0.0 and (answer_mask is not None) and (rewrite_groups is not None):
        try:
            B = logits.shape[0]
            per_group_vals = []
            for grp in rewrite_groups:
                if not isinstance(grp, (list, tuple)) or len(grp) < 2:
                    continue
                pair_vals = []
                for i in range(len(grp)):
                    for j in range(i + 1, len(grp)):
                        ia, ib = int(grp[i]), int(grp[j])
                        if 0 <= ia < B and 0 <= ib < B:
                            # Masked symmetric KL over intersection of answer masks at shared positions
                            inter = (answer_mask[ia].to(dtype=logits.dtype) * answer_mask[ib].to(dtype=logits.dtype)).view(-1)
                            val = masked_symmetric_kl(logits[ia], logits[ib], inter)
                            pair_vals.append(val)
                if pair_vals:
                    per_group_vals.append(torch.stack(pair_vals).mean())
            if per_group_vals:
                rewrite_kl_val = torch.stack(per_group_vals).mean()
        except Exception:
            pass

    # Optional explicit pair-map path (batch_meta)
    try:
        pair_map = None
        # Support both list of tuples and list of lists
        if isinstance(rewrite_groups, dict) and "pair_map" in rewrite_groups:
            pair_map = rewrite_groups.get("pair_map")
    except Exception:
        pair_map = None
    # Legacy single-pair path
    if (w_rc > 0.0) and (rewrite_kl_val.detach().item() == 0.0) and (rewrite_pair is not None):
        try:
            la = rewrite_pair.get("logits_a")
            lb = rewrite_pair.get("logits_b")
            am = rewrite_pair.get("answer_mask")
            if (la is not None) and (lb is not None) and (am is not None):
                rewrite_kl_val = masked_symmetric_kl(la, lb, am)
        except Exception:
            pass

    out["rewrite_kl_raw"] = rewrite_kl_val
    out["rewrite_kl"] = w_rc * rewrite_kl_val
    loss_total = loss_total + out["rewrite_kl"]
    # Back-compat: keep key present for older tooling
    out["rewrite_consistency"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Per-layer variance regularizer (agreement encouragement across layers)
    try:
        w_var = float(weights.get("var_reg", 0.0) or 0.0)
    except Exception:
        w_var = 0.0
    if w_var > 0.0 and isinstance(per_layer, dict):
        vloss = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        try:
            # Budget variance over layers (use sigmoid of raw to be scale-invariant)
            braw = per_layer.get("budget_raw_pl")
            if torch.is_tensor(braw) and braw.dim() == 3:
                pb = torch.sigmoid(braw).view(braw.shape[0], braw.shape[1])  # [B,L]
                # mean variance per example
                v_b = pb.var(dim=1, unbiased=False).mean()
                vloss = vloss + v_b
            # Plan disagreement: variance of per-layer plan probabilities vs mean
            plog = per_layer.get("plan_logits_pl")
            if torch.is_tensor(plog) and plog.dim() == 3:
                p = torch.softmax(plog, dim=-1)  # [B,L,K]
                pm = p.mean(dim=1, keepdim=True)  # [B,1,K]
                v_p = ((p - pm) ** 2).mean()  # scalar
                vloss = vloss + v_p
        except Exception:
            vloss = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        out["var_reg"] = vloss
        loss_total = loss_total + w_var * vloss
    else:
        out["var_reg"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Over-budget penalty (token basis)
    w_bp = float(budget_penalty_w or 0.0)
    if w_bp > 0.0 and (think_tokens_used is not None) and (target_budget is not None):
        try:
            used = think_tokens_used.view(-1).float()
            tb = target_budget.view(-1).float()
            over = (used - tb).clamp_min(0.0) / (tb.clamp_min(1.0))
            ob = over.mean()
        except Exception:
            ob = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        out["over_budget_penalty"] = ob
        loss_total = loss_total + w_bp * ob
    else:
        out["over_budget_penalty"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Think CE on masked region (optional)
    w_tce = float(think_ce_w or 0.0)
    if w_tce > 0.0 and (think_mask_tce is not None):
        try:
            B, T, V = logits.shape
            m = think_mask_tce.to(device=logits.device)
            lbl = labels.to(dtype=torch.long, device=logits.device)
            # mask labels outside vocab and outside think
            invalid = (lbl >= V) | (lbl < 0)
            lbl = torch.where((m > 0) & (~invalid), lbl, torch.full_like(lbl, -100))
            tce = torch.nn.functional.cross_entropy(logits.view(B*T, V), lbl.view(B*T), ignore_index=-100)
        except Exception:
            tce = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        out["think_ce"] = tce
        loss_total = loss_total + w_tce * tce
    else:
        out["think_ce"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Length shaping: overlength and underlength penalties
    try:
        # allow both key styles: our existing 'len_over/len_under' or requested 'lambda_over/lambda_under'
        w_over = float(weights.get("len_over", weights.get("lambda_over", 0.0)) or 0.0)
        w_under = float(weights.get("len_under", weights.get("lambda_under", 0.0)) or 0.0)
    except Exception:
        w_over, w_under = 0.0, 0.0
    if (w_over > 0.0 or w_under > 0.0):
        # Prefer explicit used_think_tokens if provided
        if isinstance(used_think_tokens, torch.Tensor):
            used = used_think_tokens.view(-1).float().to(device=logits.device)
        elif isinstance(think_tokens_used, torch.Tensor):
            used = think_tokens_used.view(-1).float().to(device=logits.device)
        else:
            used = torch.zeros(B, device=logits.device, dtype=loss_total.dtype)

        # Resolve per-sample bin bounds [min,max]
        def _bounds_for_label(label: str, i: int) -> tuple[torch.Tensor, torch.Tensor]:
            # If a target budget is provided, scale bins as fractions of it; else use static defaults
            if isinstance(target_budget, torch.Tensor):
                b = target_budget.view(-1).float().to(device=logits.device)
                Bi = b[i].clamp_min(1.0)
                if label == "short":
                    return torch.tensor(0.0, device=logits.device), (0.25 * Bi)
                if label == "mid":
                    return (0.25 * Bi), (0.6 * Bi)
                # long
                return (0.6 * Bi), Bi
            # Fallback static bins
            if label == "short":
                return torch.tensor(0.0, device=logits.device), torch.tensor(64.0, device=logits.device)
            if label == "mid":
                return torch.tensor(65.0, device=logits.device), torch.tensor(256.0, device=logits.device)
            return torch.tensor(257.0, device=logits.device), torch.tensor(512.0, device=logits.device)

        # Expand labels to a list per batch
        labels_list: list[str] = []
        target = target_bin if (target_bin is not None) else target_length_bin
        if isinstance(target, str):
            labels_list = [target for _ in range(len(used))]
        elif isinstance(target, (list, tuple)):
            labels_list = [str(x) for x in target]
            # pad/trim to batch
            if len(labels_list) < len(used):
                labels_list = labels_list + [labels_list[-1] if labels_list else "long"] * (len(used) - len(labels_list))
            labels_list = labels_list[: len(used)]
        elif isinstance(target, torch.Tensor):
            # map integer bins {0,1,2} -> short/mid/long
            arr = target.view(-1).tolist()
            map_int = {0: "short", 1: "mid", 2: "long"}
            labels_list = [map_int.get(int(v), "long") for v in arr]
            if len(labels_list) < len(used):
                labels_list = labels_list + ["long"] * (len(used) - len(labels_list))
            labels_list = labels_list[: len(used)]
        else:
            # Fallback: infer from target_L_opt if present, else mark all long
            labels_list = ["long"] * len(used)

        mins = []
        maxs = []
        for i, lab in enumerate(labels_list):
            mn, mx = _bounds_for_label(str(lab).lower(), i)
            mins.append(mn)
            maxs.append(mx)
        bin_min = torch.stack(mins) if mins else torch.zeros_like(used)
        bin_max = torch.stack(maxs) if maxs else torch.ones_like(used)

        over_vec = ((used - bin_max).clamp_min(0.0) / bin_max.clamp_min(1.0))
        under_vec = (used < bin_min).float()
        over_pen = over_vec.mean()
        under_pen = under_vec.mean()

        len_loss = w_over * over_pen + w_under * under_pen
        out["len_over_pen"] = over_pen
        out["len_under_pen"] = under_pen
        out["len_loss"] = len_loss
        out["length_penalty"] = len_loss
        loss_total = loss_total + len_loss
    else:
        out["len_over_pen"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        out["len_under_pen"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        out["len_loss"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        out["length_penalty"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Expert diversity: encourage higher selector entropy (if provided via aux)
    try:
        w_exp = float(weights.get("expert_entropy_reg", 0.0) or 0.0)
    except Exception:
        w_exp = 0.0
    if w_exp != 0.0 and isinstance(aux, dict):
        try:
            eH = aux.get("expert_entropy")
            if isinstance(eH, torch.Tensor):
                ediv = -w_exp * eH
            else:
                ediv = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        except Exception:
            ediv = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        out["expert_diversity"] = ediv
        loss_total = loss_total + ediv
    else:
        out["expert_diversity"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    # Optional anchor KL to base logits
    w_kl = float(anchor_weight or 0.0)
    if w_kl > 0.0 and isinstance(anchor_logits, torch.Tensor):
        try:
            val_list = []
            if logits.dim() == 3 and anchor_logits.dim() == 3 and logits.shape[:2] == anchor_logits.shape[:2]:
                for b in range(logits.shape[0]):
                    m = anchor_mask[b] if isinstance(anchor_mask, torch.Tensor) and anchor_mask.dim() >= 2 else torch.ones(logits.shape[1], device=logits.device, dtype=logits.dtype)
                    val_list.append(masked_symmetric_kl(logits[b], anchor_logits[b], m))
                akl = torch.stack(val_list).mean() if val_list else torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
            else:
                akl = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        except Exception:
            akl = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)
        out["anchor_kl"] = w_kl * akl
        loss_total = loss_total + (w_kl * akl)
    else:
        out["anchor_kl"] = torch.tensor(0.0, dtype=loss_total.dtype, device=loss_total.device)

    out["total"] = loss_total
    return out


# Training-time context for Quiet-Star auxiliary consistency
_QUIET_STAR_CTX: ContextVar[Optional[Dict[str, Any]]] = ContextVar("QUIET_STAR_CTX", default=None)
