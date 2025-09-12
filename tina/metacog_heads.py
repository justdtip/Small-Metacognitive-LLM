# tina/metacog_heads.py
from dataclasses import dataclass
from typing import Sequence, Dict, Union, Optional, Tuple
import torch
import torch.nn as nn

@dataclass
class MetacogConfig:
    hidden_size: int
    taps: Sequence[int]         # which layers we pool (e.g., [6, 10, 14])
    proj_dim: int = 128         # compact metacog space
    head_temp: float = 1.0      # temperature for plan logits calibration
    budget_head_temperature: float = 1.0  # temperature scaling for budget head logits
    # Linked-all-layers mode flags
    linked_all_layers: bool = False
    agg: str = "attn"           # 'attn' | 'mean'
    dump_per_layer: bool = False


class LinkedTrunk(nn.Module):
    """
    Shared trunk MLP applied to last-token states from each decoder layer.
    Input:  h_last [B, L, H]
    Output: z      [B, L, D]
    """
    def __init__(self, hidden_size: int, proj_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, proj_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity(),
            nn.Linear(proj_dim, proj_dim),
            nn.SiLU(),
        )

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(h_last) or h_last.dim() != 3:
            raise ValueError("LinkedTrunk expects Tensor[B,L,H]")
        B, L, H = h_last.shape
        x = h_last.reshape(B * L, H).float()
        z = self.proj(x)
        return z.view(B, L, -1)


class PerLayerTrunk(LinkedTrunk):
    """
    Alias of LinkedTrunk to match requested naming. Applies an H→D→D MLP per layer.
    """
    pass


class PerLayerHeads(nn.Module):
    """
    Small heads over per-layer projected features.
    Input:  z [B, L, D]
    Output: dict with plan/budget/conf per-layer tensors
    """
    def __init__(self, proj_dim: int, num_plans: int = 4):
        super().__init__()
        self.plan_head = nn.Linear(proj_dim, num_plans)
        self.budget_head = nn.Linear(proj_dim, 1)
        self.conf_head = nn.Linear(proj_dim, 1)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not torch.is_tensor(z) or z.dim() != 3:
            raise ValueError("PerLayerHeads expects Tensor[B,L,D]")
        B, L, D = z.shape
        x = z.reshape(B * L, D)
        plan_logits_pl = self.plan_head(x).view(B, L, -1)
        budget_raw_pl = self.budget_head(x).view(B, L, 1)
        conf_raw_pl = self.conf_head(x).view(B, L, 1)
        return {
            "plan_logits_pl": plan_logits_pl,
            "budget_raw_pl": budget_raw_pl,
            "conf_raw_pl": conf_raw_pl,
        }


class LayerAggregator(nn.Module):
    """
    Aggregate per-layer embeddings z_l into a global g via attention or mean.
    - agg=='attn': e_l = v^T tanh(W z_l); alpha = softmax(e); g = sum_l alpha_l z_l
    - agg=='mean': alpha is uniform; g = mean(z, dim=1)
    """
    def __init__(self, proj_dim: int, agg: str = "attn"):
        super().__init__()
        self.agg = (agg or "attn").lower()
        if self.agg == "attn":
            self.W = nn.Linear(proj_dim, proj_dim)
            self.v = nn.Linear(proj_dim, 1, bias=False)
        else:
            self.W = None
            self.v = None

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not torch.is_tensor(z) or z.dim() != 3:
            raise ValueError("LayerAggregator expects Tensor[B,L,D]")
        if self.agg != "attn":
            g = z.mean(dim=1)
            return g, None
        # attention
        B, L, D = z.shape
        x = z.reshape(B * L, D)
        score = self.v(torch.tanh(self.W(x))).view(B, L)  # [B,L]
        alpha = torch.softmax(score, dim=1)
        g = (alpha.unsqueeze(-1) * z).sum(dim=1)  # [B,D]
        return g, alpha

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

        # Linked-all-layers components
        if bool(getattr(cfg, "linked_all_layers", False)):
            self._trunk = LinkedTrunk(hidden_size=H, proj_dim=D)
            self._pl_heads = PerLayerHeads(proj_dim=D, num_plans=self.plan.out_features)
            self._aggregator = LayerAggregator(proj_dim=D, agg=getattr(cfg, "agg", "attn"))

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

    def _gather_linked_h_last(self) -> Optional[torch.Tensor]:
        """Assemble [B,L,H] from cached per-layer last-step states in layer index order."""
        if not self._tap_cache:
            return None
        keys = sorted(k for k in self._tap_cache.keys() if isinstance(k, int))
        if not keys:
            return None
        hs = [self._tap_cache[k] for k in keys if k in self._tap_cache]  # each [B,1,H]
        if not hs:
            return None
        try:
            hcat = torch.cat(hs, dim=1)  # [B,L,H]
            return hcat
        except Exception:
            return None

    def forward(self, B_max: Union[int, float, torch.Tensor] = 256, return_state: bool = False) -> Dict[str, torch.Tensor]:
        linked = bool(getattr(self.cfg, "linked_all_layers", False))
        out: Dict[str, torch.Tensor] = {}

        if linked:
            # Linked-all-layers: gather last-token states from all cached layers → trunk → per-layer heads → aggregate
            h_last = self._gather_linked_h_last()
            if h_last is None:
                # default zeros (single dummy layer) if no taps
                h_last = self.pool.weight.new_zeros((1, 1, self.cfg.hidden_size))
            # FP32 trunk for stability
            z_pl = self._trunk(h_last.to(dtype=self.pool.weight.dtype))  # [B,L,D]
            # Per-layer diagnostics
            pl = self._pl_heads(z_pl)
            # Aggregate embedding
            g, alpha = self._aggregator(z_pl)
            # Final heads from aggregated state
            temp = max(float(self.cfg.head_temp), 1e-6)
            plan_logits = (self.plan(g) / temp).clamp(-20, 20)  # [B,K]
            raw = self.budget(g) / torch.clamp(self._budget_temp, min=torch.tensor(1e-6, dtype=self._budget_temp.dtype, device=self._budget_temp.device))
            if not torch.is_tensor(B_max):
                Bm = torch.tensor(float(B_max), dtype=raw.dtype, device=raw.device)
            else:
                Bm = B_max.to(dtype=raw.dtype, device=raw.device)
            budget = torch.sigmoid(raw) * Bm
            zero = torch.zeros_like(budget)
            budget = torch.maximum(zero, torch.minimum(budget, Bm))
            confidence = torch.sigmoid(self.confidence(g))
            out.update({"plan_logits": plan_logits, "budget": budget, "confidence": confidence})
            if return_state:
                out["state"] = g
            if bool(getattr(self.cfg, "dump_per_layer", False)):
                # Provide compact per-layer diagnostics
                diag: Dict[str, torch.Tensor] = {
                    "z": z_pl,
                    "plan_logits_pl": pl.get("plan_logits_pl"),
                    "budget_raw_pl": pl.get("budget_raw_pl"),
                    "conf_raw_pl": pl.get("conf_raw_pl"),
                }
                if alpha is not None:
                    diag["alpha"] = alpha
                out["per_layer"] = diag
            return out

        # Legacy (three-tap) path: average selected taps then project to D
        if not self._tap_cache:
            # default zeros if not populated (e.g. before a dry forward)
            z = self.pool.weight.new_zeros((1, self.cfg.proj_dim))
        else:
            hs = []
            for i in self.cfg.taps:
                if i in self._tap_cache:
                    hs.append(self._tap_cache[i])  # [B,1,H]
            if not hs:
                z = self.pool.weight.new_zeros((1, self.cfg.proj_dim))
            else:
                hcat = torch.cat(hs, dim=1).mean(dim=1)  # [B,H]
                in_dtype = self.pool.weight.dtype
                z = self.act(self.pool(hcat.to(in_dtype)))            # [B,D]

        temp = max(float(self.cfg.head_temp), 1e-6)
        plan_logits = (self.plan(z) / temp).clamp(-20, 20)          # [B,4]
        raw = self.budget(z) / torch.clamp(self._budget_temp, min=torch.tensor(1e-6, dtype=self._budget_temp.dtype, device=self._budget_temp.device))  # [B,1]
        if not torch.is_tensor(B_max):
            Bm = torch.tensor(float(B_max), dtype=raw.dtype, device=raw.device)
        else:
            Bm = B_max.to(dtype=raw.dtype, device=raw.device)
        budget = torch.sigmoid(raw) * Bm
        zero = torch.zeros_like(budget)
        budget = torch.maximum(zero, torch.minimum(budget, Bm))
        confidence = torch.sigmoid(self.confidence(z))  # [B,1]
        out = {"plan_logits": plan_logits, "budget": budget, "confidence": confidence}
        if return_state:
            out["state"] = z
        return out
