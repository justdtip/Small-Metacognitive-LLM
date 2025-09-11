from __future__ import annotations
from typing import Tuple, List, Dict, Any, Iterable, Callable, Optional
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
    slice_keys: Optional[Iterable[str]] = ("plan_src", "budget_src"),
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
                    out["slices"][key][str(v)] = compute_fn(sub)
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
