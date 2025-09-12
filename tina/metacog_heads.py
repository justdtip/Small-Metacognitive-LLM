# tina/metacog_heads.py
from dataclasses import dataclass
from typing import Sequence, Dict, Union
import torch
import torch.nn as nn

@dataclass
class MetacogConfig:
    hidden_size: int
    taps: Sequence[int]         # which layers we pool (e.g., [6, 10, 14])
    proj_dim: int = 128         # compact metacog space
    head_temp: float = 1.0      # temperature for plan logits calibration
    budget_head_temperature: float = 1.0  # temperature scaling for budget head logits

class MetacogHeads(nn.Module):
    """
    Small heads for plan/budget/confidence, reading a compact pooled z.
    You can train these later (aux losses / RL).
    """
    def __init__(self, cfg: MetacogConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.pool = nn.Linear(H, cfg.proj_dim)
        self.act = nn.SiLU()
        D = cfg.proj_dim

        self.plan = nn.Linear(D, 4)         # outline/verify/decompose/stop
        self.budget = nn.Linear(D, 1)       # raw budget logit → scaled via sigmoid to [0..Bmax]
        self.confidence = nn.Linear(D, 1)   # scalar

        self._tap_cache = {}  # layer_idx -> last hidden (B,T,H)
        # Non-trainable temperature for budget head (can be tuned via calibration)
        self._budget_temp = nn.Parameter(torch.tensor(float(cfg.budget_head_temperature)), requires_grad=False)

    def clear_cache(self):
        self._tap_cache.clear()

    def register_tap(self, layer_idx: int, h):
        # h: [B, T, H]; cache last token
        if not torch.is_tensor(h) or h.dim() != 3:
            raise ValueError(f"tap expects Tensor[B,T,H], got {type(h)} with dim {getattr(h, 'dim', lambda: None)()}")
        if h.shape[-1] != self.cfg.hidden_size:
            raise ValueError(f"tap hidden size mismatch: got {h.shape[-1]}, expected {self.cfg.hidden_size}")
        self._tap_cache[layer_idx] = h[:, -1:, :].detach()  # keep last-step feature

    def set_budget_temp(self, value: Union[float, int]):
        with torch.no_grad():
            self._budget_temp.copy_(torch.tensor(float(value), dtype=self._budget_temp.dtype, device=self._budget_temp.device))

    def forward(self, B_max: Union[int, float, torch.Tensor] = 256) -> Dict[str, torch.Tensor]:
        if not self._tap_cache:
            # default zeros if not populated (e.g. before a dry forward)
            z = self.pool.weight.new_zeros((1, self.cfg.proj_dim))
        else:
            # concat pooled taps
            hs = []
            for i in self.cfg.taps:
                if i in self._tap_cache:
                    hs.append(self._tap_cache[i])  # [B,1,H]
            if not hs:
                z = self.pool.weight.new_zeros((1, self.cfg.proj_dim))
            else:
                hcat = torch.cat(hs, dim=1).mean(dim=1)  # [B,H]
                # Align input dtype to layer weight dtype to avoid matmul dtype mismatch (e.g., BF16 vs FP32)
                in_dtype = self.pool.weight.dtype
                z = self.act(self.pool(hcat.to(in_dtype)))            # [B,D]

        # Clamp and temperature-scale plan logits for stability
        temp = max(float(self.cfg.head_temp), 1e-6)
        plan_logits = (self.plan(z) / temp).clamp(-20, 20)          # [B,4]
        # Budget head: sigmoid over temperature-scaled logit, then scale to [0..B_max]
        raw = self.budget(z) / torch.clamp(self._budget_temp, min=torch.tensor(1e-6, dtype=self._budget_temp.dtype, device=self._budget_temp.device))  # [B,1]
        # Broadcastable B_max tensor
        if not torch.is_tensor(B_max):
            Bm = torch.tensor(float(B_max), dtype=raw.dtype, device=raw.device)
        else:
            Bm = B_max.to(dtype=raw.dtype, device=raw.device)
        budget = torch.sigmoid(raw) * Bm
        # Clamp to [0, B_max]
        zero = torch.zeros_like(budget)
        budget = torch.maximum(zero, torch.minimum(budget, Bm))
        confidence = torch.sigmoid(self.confidence(z))  # [B,1]
        return {"plan_logits": plan_logits, "budget": budget, "confidence": confidence}
