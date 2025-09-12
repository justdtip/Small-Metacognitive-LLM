from __future__ import annotations
from typing import Tuple, List, Dict, Any, Iterable, Callable, Optional, Union
from pathlib import Path
import json as _json
import csv as _csv
import torch

def temperature_fit(logits: torch.Tensor, labels: torch.Tensor, *, max_iter: int = 1000, lr: float = 0.01,
                    bounds: Tuple[float, float] = (0.25, 4.0)) -> float:
    """
    Fit a scalar temperature for calibration via NLL minimization.
    Supports binary (logits shape [N] or [N,1]) and multiclass (logits [N,C], labels in [0..C-1]).
    Returns temperature T>0.
    """
    logits = logits.detach().float()
    if logits.ndim == 1:
        logits = logits[:, None]
    N, C = logits.shape
    if C == 1:
        # binary: sigmoid(logit / T)
        y = labels.detach().float().view(-1)
        T = torch.tensor(1.0, requires_grad=True)
        opt = torch.optim.LBFGS([T], max_iter=50, line_search_fn="strong_wolfe")
        bmin, bmax = bounds
        def closure():
            opt.zero_grad()
            t = torch.clamp(T, bmin, bmax)
            p = torch.sigmoid(logits.view(-1) / t)
            # NLL
            eps = 1e-8
            loss = -(y * torch.log(p + eps) + (1 - y) * torch.log(1 - p + eps)).mean()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            T_final = float(torch.clamp(T, bmin, bmax).item())
        return T_final
    else:
        # multiclass: softmax(logits / T)
        y = labels.detach().long().view(-1)
        T = torch.tensor(1.0, requires_grad=True)
        opt = torch.optim.LBFGS([T], max_iter=50, line_search_fn="strong_wolfe")
        bmin, bmax = bounds
        def closure():
            opt.zero_grad()
            t = torch.clamp(T, bmin, bmax)
            p = torch.log_softmax(logits / t, dim=-1)
            loss = torch.nn.functional.nll_loss(p, y)
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            T_final = float(torch.clamp(T, bmin, bmax).item())
        return T_final

def ece_binary(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    probs = probs.detach().float().view(-1)
    labels = labels.detach().float().view(-1)
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i].item(), bins[i + 1].item()
        mask = (probs >= lo) & (probs < hi)
        if mask.any():
            conf = probs[mask].mean().item()
            acc = (probs[mask] >= 0.5).float().eq(labels[mask]).float().mean().item()
            ece += (mask.float().mean().item()) * abs(acc - conf)
    return float(ece)

def brier_binary(probs: torch.Tensor, labels: torch.Tensor) -> float:
    probs = probs.detach().float().view(-1)
    labels = labels.detach().float().view(-1)
    return float(torch.mean((probs - labels) ** 2).item())


def brier_score(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Alias for Brier score to match naming in higher-level specs."""
    return brier_binary(probs, labels)


def leakage_rate(outputs: List[str]) -> float:
    """Fraction of outputs containing '<think>' when hidden mode is expected."""
    if not outputs:
        return 0.0
    hits = 0
    for s in outputs:
        try:
            if ("<think>" in (s or "")) or ("</think>" in (s or "")):
                hits += 1
        except Exception:
            pass
    return float(hits) / float(max(1, len(outputs)))


def ece_brier_report(confidences: torch.Tensor, correctness: torch.Tensor) -> Dict[str, float]:
    """
    Compute ECE and Brier given confidences in [0,1] and correctness {0,1}.
    Returns {'ece':..., 'brier':...} with float values; returns zeros if inputs invalid.
    """
    try:
        p = confidences.detach().float().view(-1)
        y = correctness.detach().float().view(-1)
        return {"ece": ece_binary(p, y), "brier": brier_binary(p, y)}
    except Exception:
        return {"ece": 0.0, "brier": 0.0}


# --------- Token-level F1 for long-form answers ---------

def _tokenize(s: str) -> List[str]:
    return [t for t in (s or "").strip().split() if t]


def f1_token(pred: str, gold: str) -> float:
    p, g = _tokenize(pred), _tokenize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    from collections import Counter
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    prec = overlap / max(1, sum(cp.values()))
    rec = overlap / max(1, sum(cg.values()))
    return 2 * prec * rec / max(1e-8, (prec + rec))


def exact_match(pred: str, gold: str) -> float:
    """Strict string equality after strip."""
    return 1.0 if (str(pred or '').strip() == str(gold or '').strip()) else 0.0


def token_f1(pred: str, gold: str) -> float:
    """Alias wrapper over f1_token for external callers."""
    return f1_token(pred, gold)


# --------- Rubric/teacher scoring for <think> ---------

def think_rubric_score(body: str, grader: Callable[[str], float] | None = None) -> float | None:
    """
    Apply a rubric/teacher grading function to the <think> span only.
    - grader: callable that returns a score in [0,1]; if None, returns None
    - body: full text containing <think> ... </think>
    Returns float in [0,1] or None if no span/grader unavailable.
    """
    if grader is None:
        return None
    try:
        i = body.find("<think>")
        if i == -1:
            return None
        j = body.find("</think>", i + len("<think>"))
        if j == -1:
            return None
        span = body[i:j + len("</think>")]
        s = float(grader(span))
        # clamp to [0,1]
        if s < 0.0:
            s = 0.0
        if s > 1.0:
            s = 1.0
        return s
    except Exception:
        return None


# --------- Aggregation and export utilities (dashboard support) ---------

def aggregate(
    records: List[Dict[str, Any]],
    compute_fn: Callable[[List[Dict[str, Any]]], Dict[str, Any]],
    *,
    slice_keys: Optional[Iterable[str]] = ("plan_src", "budget_src", "style_tag"),
) -> Dict[str, Any]:
    """
    Aggregate metrics for overall data and slices by provenance keys.
    - records: list of eval samples
    - compute_fn: function(records)->metrics dict (e.g., train.eval_loop.compute_eval_metrics)
    - slice_keys: keys to slice on (default: plan_src, budget_src)
    Returns {'overall': metrics, 'slices': {key: {value: metrics}}}.
    """
    out: Dict[str, Any] = {"overall": compute_fn(records), "slices": {}}
    # Determine keys to slice on; include difficulty_bin if present
    keys = list(slice_keys) if slice_keys else []
    if any("difficulty_bin" in r for r in records) and "difficulty_bin" not in keys:
        keys.append("difficulty_bin")
    if not keys:
        return out
    for key in keys:
        # Collect unique values
        vals = []
        for r in records:
            if key in r:
                v = r.get(key)
                if v not in vals:
                    vals.append(v)
        if not vals:
            continue
        out["slices"][key] = {}
        for v in vals:
            sub = [r for r in records if r.get(key) == v]
            if sub:
                try:
                    metrics = compute_fn(sub)
                    out["slices"][key][str(v)] = metrics
                    # Also expose a flattened convenience key for simple assertions/tests
                    out[f"slice:{key}={v}"] = metrics
                except Exception:
                    out["slices"][key][str(v)] = {}
    return out


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key))
        else:
            flat[key] = v
    return flat


def to_csv_json(data: Dict[str, Any], *, out_json: Optional[str] = None, out_csv: Optional[str] = None) -> None:
    """
    Save aggregated metrics to JSON and/or CSV. CSV flattens nested keys using dot notation.
    """
    if out_json:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(data), encoding="utf-8")
    if out_csv:
        # Flatten top-level sections into rows; each row corresponds to an entry in {'overall':..., 'slices':...}
        rows: List[Dict[str, Any]] = []
        # Overall
        rows.append({"section": "overall", **(data.get("overall") or {})})
        # Slices
        slices = data.get("slices") or {}
        for key, m in slices.items():
            for val, metrics in (m or {}).items():
                rows.append({"section": f"slice:{key}={val}", **(metrics or {})})
        # Flatten and write
        p = Path(out_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        flat_rows = [_flatten_dict(r) for r in rows]
        # Collect columns
        cols = sorted({c for r in flat_rows for c in r.keys()})
        with p.open("w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in flat_rows:
                w.writerow(r)


# --------- Rewrite consistency diagnostics ---------
def rewrite_kl_mean(records: List[Dict[str, Any]]) -> float:
    """
    Compute the mean rewrite KL from a list of logs/records.
    Accepts either {'rewrite_kl': value} or {'loss_rewrite_kl': value} style entries.
    Returns 0.0 when unavailable.
    """
    vals: List[float] = []
    for r in records or []:
        v = None
        if isinstance(r, dict):
            if "rewrite_kl" in r:
                v = r.get("rewrite_kl")
            elif "loss_rewrite_kl" in r:
                v = r.get("loss_rewrite_kl")
        if v is not None:
            try:
                vals.append(float(v))
            except Exception:
                pass
    if not vals:
        return 0.0
    import statistics
    try:
        return float(statistics.fmean(vals))
    except Exception:
        return float(sum(vals) / max(1, len(vals)))


# --------- Calibration artifact helpers ---------
def fit_confidence_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Convenience wrapper using temperature_fit for binary confidence calibration.
    Accepts logits shape [N] or [N,1] and labels [N] in {0,1}.
    Returns a positive scalar temperature.
    """
    if logits.ndim == 2 and logits.shape[-1] == 1:
        logits = logits.view(-1)
    return float(temperature_fit(logits, labels))


def fit_plan_thresholds(plan_logits: torch.Tensor, labels: torch.Tensor, num_plans: int = 4) -> Dict[str, float]:
    """
    Derive simple per-class probability thresholds for plan selection.
    For each class k, threshold = median of P(class=k) among samples with label==k.
    Names default to ['short','deliberate','verify','stop'] truncated to K.
    """
    with torch.no_grad():
        if plan_logits.ndim == 1:
            plan_logits = plan_logits.view(-1, 1)
        K = int(plan_logits.shape[-1])
        names = ["short", "deliberate", "verify", "stop"][:K]
        probs = torch.softmax(plan_logits.detach().float(), dim=-1)
        y = labels.detach().long().view(-1)
        out: Dict[str, float] = {}
        for k in range(K):
            mask = (y == k)
            if mask.any():
                vals = probs[mask, k].sort().values
                thr = float(vals[int(0.5 * (len(vals) - 1))].item()) if len(vals) > 0 else 0.5
            else:
                thr = 0.5
            out[names[k]] = float(max(0.0, min(1.0, thr)))
        return out


def choose_budget_clip(pred_budgets: torch.Tensor, targets: torch.Tensor, quantile: float = 0.98) -> int:
    """
    Choose a conservative integer clip for budgets, based on the target distribution (fallback to preds).
    Returns max(1, int(quantile(...))).
    """
    q = float(max(0.0, min(1.0, quantile)))
    try:
        base = targets.detach().float().view(-1)
        if base.numel() == 0 or torch.isnan(base).all():
            base = pred_budgets.detach().float().view(-1)
        val = torch.quantile(base, q).item() if base.numel() > 0 else 1.0
    except Exception:
        val = 1.0
    return int(max(1, round(val)))


def save_calibration(calib_path: Union[str, Path], *, conf_temp: float,
                     plan_thresholds: Dict[str, float] | None = None,
                     budget_clip: Optional[int] = None,
                     budget_head_temp: Optional[float] = None) -> None:
    """
    Persist a calibration JSON with keys: conf_temp, plan_thresholds (optional), budget_clip (optional).
    For backward-compat, also writes budget_posthoc_clip alias when clip is provided.
    """
    blob: Dict[str, Any] = {"conf_temp": float(conf_temp)}
    if isinstance(plan_thresholds, dict) and plan_thresholds:
        blob["plan_thresholds"] = {str(k): float(v) for k, v in plan_thresholds.items() if v is not None}
    if isinstance(budget_clip, (int, float)) and budget_clip:
        blob["budget_clip"] = int(budget_clip)
        blob["budget_posthoc_clip"] = int(budget_clip)
    if isinstance(budget_head_temp, (int, float)) and budget_head_temp:
        blob["budget_head_temp"] = float(budget_head_temp)
    p = Path(calib_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(_json.dumps(blob), encoding="utf-8")


def load_calibration(calib_path: Union[str, Path]) -> Dict[str, Any]:
    """Load calibration JSON and normalize keys. Returns dict with conf_temp, plan_thresholds, budget_clip."""
    p = Path(calib_path)
    d = _json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, Any] = {}
    if "conf_temp" in d:
        out["conf_temp"] = float(d["conf_temp"])
    if isinstance(d.get("plan_thresholds"), dict):
        out["plan_thresholds"] = {str(k): float(v) for k, v in d["plan_thresholds"].items() if v is not None}
    if "budget_clip" in d:
        out["budget_clip"] = int(d["budget_clip"]) if d["budget_clip"] is not None else None
    elif "budget_posthoc_clip" in d:
        out["budget_clip"] = int(d["budget_posthoc_clip"])
    if "budget_head_temp" in d:
        out["budget_head_temp"] = float(d["budget_head_temp"]) if d["budget_head_temp"] is not None else None
    return out


# --------- Layer diagnostics helpers (optional) ---------
def plan_agreement_rate(plan_logits_pl: torch.Tensor, plan_logits_agg: torch.Tensor) -> float:
    """
    Fraction of layers whose argmax plan equals the aggregated argmax plan.
    - plan_logits_pl: [B,L,K]
    - plan_logits_agg: [B,K]
    Returns mean across batch in [0,1].
    """
    if not (torch.is_tensor(plan_logits_pl) and torch.is_tensor(plan_logits_agg)):
        return 0.0
    try:
        pa = torch.argmax(plan_logits_agg, dim=-1).view(-1, 1)  # [B,1]
        pl = torch.argmax(plan_logits_pl, dim=-1)               # [B,L]
        agree = (pl == pa).float().mean(dim=1)
        return float(agree.mean().item())
    except Exception:
        return 0.0


def budget_variance_mean(budget_raw_pl: torch.Tensor) -> float:
    """
    Mean variance over layers of per-layer budgets (sigmoid(raw) for stability).
    - budget_raw_pl: [B,L,1]
    Returns scalar float.
    """
    if not torch.is_tensor(budget_raw_pl):
        return 0.0
    try:
        x = torch.sigmoid(budget_raw_pl.view(budget_raw_pl.shape[0], budget_raw_pl.shape[1]))
        v = x.var(dim=1, unbiased=False).mean()
        return float(v.item())
    except Exception:
        return 0.0


def alpha_entropy_mean(alpha: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Mean entropy of attention weights over layers; normalized by log(L).
    - alpha: [B,L]
    """
    if not torch.is_tensor(alpha):
        return 0.0
    try:
        p = alpha.clamp_min(eps)
        H = -(p * p.log()).sum(dim=1)
        norm = torch.log(torch.tensor(float(p.shape[1]), dtype=H.dtype, device=H.device)).clamp_min(1.0)
        Hn = (H / norm).mean().clamp(0.0, 1.0)
        return float(Hn.item())
    except Exception:
        return 0.0
