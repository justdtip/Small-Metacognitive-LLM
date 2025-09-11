from __future__ import annotations
"""
Minimal RL / preference loop for budget control.

Implements a tiny policy over budget caps and two update styles:
 - REINFORCE with a Gaussian policy over an unconstrained action a, mapped to a budget cap via sigmoid(a)*Bmax.
 - A simple DPO-style pairwise update that prefers shorter budgets when both candidates are correct.

These utilities are synthetic-friendly and decoupled from HF; suitable for smoke tests or prototyping.
"""
from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn

from train.reward import budget_aware_reward


class GaussianBudgetPolicy(nn.Module):
    def __init__(self, in_dim: int, max_budget: int = 256, init_scale: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        with torch.no_grad():
            self.linear.weight.fill_(0.0)
            self.linear.bias.fill_(init_scale)
        self.max_budget = int(max(1, max_budget))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean parameter in R
        return self.linear(x).squeeze(-1)

    def to_budget(self, a: torch.Tensor) -> torch.Tensor:
        # map unconstrained action to [1, Bmax]
        return (torch.sigmoid(a) * float(self.max_budget - 1) + 1.0)


def reinforce_step(
    policy: GaussianBudgetPolicy,
    feats: torch.Tensor,
    think_len: torch.Tensor,
    correct: torch.Tensor,
    *,
    alpha: float = 0.01,
    format_bonus: float = 0.5,
    sigma: float = 5.0,
    K: int = 4,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, Any]:
    """
    One REINFORCE step: sample K budgets per example, compute budget-aware reward, and update policy.
    - policy: GaussianBudgetPolicy mapping feats→Normal(mean, sigma) over action a
    - feats: (B,D)
    - think_len: (B,) token counts
    - correct: (B,) in {0,1}
    Returns {'reward_mean','mu_mean','mu_after'}.
    """
    B = feats.shape[0]
    device = feats.device
    mu = policy(feats)  # (B,)
    dist = torch.distributions.Normal(mu, torch.tensor(float(sigma), device=device))
    # Sample K actions
    a = dist.rsample((K,))  # (K,B)
    budgets = policy.to_budget(a)  # (K,B)
    # Compute rewards per sample
    rewards = []
    for k in range(K):
        r = []
        for i in range(B):
            bcap = float(budgets[k, i].item())
            tl = int(think_len[i].item())
            cq = float(correct[i].item())
            # Penalize budgets that exceed observed think length (encourage concise thoughts)
            penalty = float(alpha) * max(0.0, bcap - float(tl))
            r.append(float(cq) + float(format_bonus) - penalty)
        rewards.append(torch.tensor(r, dtype=mu.dtype, device=device))
    R = torch.stack(rewards, dim=0)  # (K,B)
    baseline = R.mean()
    # REINFORCE loss: -(R - baseline) * log_prob(a)
    logp = dist.log_prob(a)  # (K,B)
    loss = -((R - baseline).detach() * logp).mean()
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        mu_after = policy(feats).mean().item()
    return {
        "reward_mean": float(R.mean().item()),
        "mu_mean": float(mu.mean().item()),
        "mu_after": float(mu_after),
    }


def dpo_step(
    policy: GaussianBudgetPolicy,
    feats_pref: torch.Tensor,
    feats_rej: torch.Tensor,
    *,
    beta: float = 0.1,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, Any]:
    """
    Simple pairwise DPO-like update: encourage lower budgets for preferred examples vs rejected ones.
    Uses logistic loss over the difference of means.
    """
    mu_pref = policy(feats_pref)  # (B,)
    mu_rej = policy(feats_rej)    # (B,)
    # Want mu_pref < mu_rej → loss = -log(sigmoid(beta*(mu_rej - mu_pref)))
    diff = beta * (mu_rej - mu_pref)
    loss = -torch.nn.functional.logsigmoid(diff).mean()
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        gap = float((mu_rej.mean() - mu_pref.mean()).item())
    return {"loss": float(loss.item()), "gap": gap}


def dry_run_budget_rl(steps: int = 50, seed: int = 0) -> Dict[str, Any]:
    """
    Synthetic dry run showing that REINFORCE with a budget penalty reduces the mean predicted budget
    without impacting correctness (fixed to 1). Returns before/after means.
    """
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    B, D = 64, 4
    feats = torch.randn(B, D, generator=g)
    # Difficulty → base think length; harder → longer
    diff = torch.sigmoid(feats.mean(dim=1))  # [0,1]
    think_len = (10 + 100 * diff).round().to(torch.long)
    correct = torch.ones(B)  # fixed correctness
    policy = GaussianBudgetPolicy(D, max_budget=128, init_scale=0.0)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
    stats = []
    mu0 = float(policy(feats).mean().item())
    for t in range(steps):
        s = reinforce_step(policy, feats, think_len, correct, alpha=0.01, format_bonus=0.5, sigma=8.0, K=4, optimizer=opt)
        stats.append(s)
    mu1 = float(policy(feats).mean().item())
    return {"mu_before": mu0, "mu_after": mu1, "rewards": [s["reward_mean"] for s in stats]}
