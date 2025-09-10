from __future__ import annotations
from typing import Tuple
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

