# tina/metacog_heads.py
from dataclasses import dataclass
from typing import Sequence, Dict
import torch
import torch.nn as nn

@dataclass
class MetacogConfig:
    hidden_size: int
    taps: Sequence[int]         # which layers we pool (e.g., [6, 10, 14])
    proj_dim: int = 128         # compact metacog space
    head_temp: float = 1.0      # temperature for plan logits calibration

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
        self.budget = nn.Linear(D, 1)       # 0..Bmax (scaled at serve time)
        self.confidence = nn.Linear(D, 1)   # scalar

        self._tap_cache = {}  # layer_idx -> last hidden (B,T,H)

    def clear_cache(self):
        self._tap_cache.clear()

    def register_tap(self, layer_idx: int, h):
        # h: [B, T, H]; cache last token
        if not torch.is_tensor(h) or h.dim() != 3:
            raise ValueError(f"tap expects Tensor[B,T,H], got {type(h)} with dim {getattr(h, 'dim', lambda: None)()}")
        if h.shape[-1] != self.cfg.hidden_size:
            raise ValueError(f"tap hidden size mismatch: got {h.shape[-1]}, expected {self.cfg.hidden_size}")
        self._tap_cache[layer_idx] = h[:, -1:, :].detach()  # keep last-step feature

    def forward(self, B_max: int = 256) -> Dict[str, torch.Tensor]:
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
        budget_raw = self.budget(z).clamp_min(0)                     # [B,1]
        confidence = torch.sigmoid(self.confidence(z))  # [B,1]
        # Scale budget to an integer cap
        budget = (budget_raw * B_max).clamp(0, B_max)
        return {"plan_logits": plan_logits, "budget": budget, "confidence": confidence}
