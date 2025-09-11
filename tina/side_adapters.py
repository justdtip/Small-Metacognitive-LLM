# tina/side_adapters.py
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import contextvars
from contextvars import ContextVar
import torch
import torch.nn as nn

@dataclass
class ResidualAdapterConfig:
    hidden_size: int
    rank: int = 8
    nonlin: str = "silu"      # "silu" | "tanh" | "none"
    init_gate: float = 0.0    # start at 0 → no behavior change
    layers: Optional[Sequence[int]] = None  # if None → all layers

class LowRankAdapter(nn.Module):
    """
    Additive low-rank delta on a layer's residual: Δh = g * U(φ(V^T h))
    g is a learned scalar gate (starts at 0).
    """
    def __init__(self, cfg: ResidualAdapterConfig):
        super().__init__()
        h, r = cfg.hidden_size, cfg.rank
        self.U = nn.Parameter(torch.zeros(h, r))
        self.V = nn.Parameter(torch.zeros(h, r))
        nn.init.kaiming_uniform_(self.U, a=5**0.5)
        nn.init.kaiming_uniform_(self.V, a=5**0.5)

        if cfg.nonlin == "silu":
            self.act = nn.SiLU()
        elif cfg.nonlin == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()

        self.gate = nn.Parameter(torch.tensor(cfg.init_gate, dtype=torch.float32))
        # per-token gate
        self.hc_gate = HardConcreteGate()
        self._last_gate_activity: Optional[torch.Tensor] = None
        self._last_gate_coverage: Optional[torch.Tensor] = None
        self._enabled = True

    def enable(self, flag: bool = True):
        self._enabled = flag

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return h
        g = torch.clamp(self.gate, 0.0, 1.0)
        if g.item() <= 1e-6:
            return h
        # h: [B, T, H]
        B, T, H = h.shape
        # matmul via view → (B*T,H) x (H,r)
        x = h.view(B * T, H)
        # do math in float32 for numerical stability, then cast back
        x32 = x.to(torch.float32)
        V32 = self.V.to(torch.float32)
        U32 = self.U.to(torch.float32)
        v = x32 @ V32   # [B*T, r]
        v = self.act(v)
        u = v @ U32.t()  # [B*T, H]
        delta = (g.to(torch.float32) * u).view(B, T, H).to(h.dtype)
        # token-level gate + optional think-mask
        y_gated, s = self.hc_gate(delta, self.training)
        # record avg gate activity
        try:
            self._last_gate_activity = s.mean().detach()
        except Exception:
            self._last_gate_activity = None
        m = _ctx_think_mask.get()
        if m is not None:
            m_t = m.to(y_gated.dtype).unsqueeze(-1)
            full = y_gated
            masked = full * m_t
            # coverage: proportion of post-gate activity aligned to think positions
            try:
                num = torch.sum(torch.abs(masked)).detach()
                den = torch.sum(torch.abs(full)).detach().clamp_min(1e-8)
                self._last_gate_coverage = (num / den)
            except Exception:
                self._last_gate_coverage = None
            y_gated = masked
        else:
            self._last_gate_coverage = None
        return h + y_gated

class IntrospectionScaffold(nn.Module):
    """
    Wraps a HF model (CausalLM). Attaches forward hooks to decoder layers
    to add LowRankAdapter deltas additively (strictly additive path).
    """
    def __init__(self, model, hidden_size: int, num_layers: int, cfg: Optional[ResidualAdapterConfig]):
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if cfg is None:
            cfg = ResidualAdapterConfig(hidden_size=hidden_size)
        self.cfg = cfg
        # Register as ModuleList so params move with .to() and are tracked
        self.adapters: nn.ModuleList = nn.ModuleList()
        self._hooks = []
        # context-local think flag; per-request isolation
        self._think_flag: contextvars.ContextVar = contextvars.ContextVar("tina_think", default=False)

        target_layers = cfg.layers if cfg.layers is not None else list(range(num_layers))
        # create per-layer adapter module
        for _ in target_layers:
            self.adapters.append(LowRankAdapter(cfg))

        # attach hooks
        dec_layers = self._get_decoder_layers(model)
        idx_map = {i: k for k, i in enumerate(target_layers)}
        for i, layer in enumerate(dec_layers):
            if i not in idx_map:
                continue
            k = idx_map[i]
            adap = self.adapters[k]

            def hook_fn(module, inputs, output, _adap=adap):
                # Apply adapter only in think mode (context-local), and handle tuple outputs
                try:
                    if not self._think_flag.get():
                        return output
                except LookupError:
                    return output
                # Direct tensor output
                if torch.is_tensor(output):
                    return _adap(output)
                # Tuple/list: assume first element is hidden_states tensor
                if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
                    new_h = _adap(output[0])
                    return (new_h, *output[1:])
                if isinstance(output, list) and len(output) > 0 and torch.is_tensor(output[0]):
                    new_h = _adap(output[0])
                    rest = list(output[1:])
                    return [new_h, *rest]
                # Unknown structure; no-op to avoid breaking model
                return output

            self._hooks.append(layer.register_forward_hook(hook_fn))  # mutate return

    @staticmethod
    def _get_decoder_layers(model) -> List[nn.Module]:
        # Try a few common paths for Qwen2 + PEFT
        for path in [
            "model.model.layers",                       # plain HF
            "base_model.model.model.layers",            # PEFT wrapped
            "model.base_model.model.model.layers",      # extra nesting
        ]:
            cur = model
            ok = True
            for name in path.split("."):
                if not hasattr(cur, name):
                    ok = False
                    break
                cur = getattr(cur, name)
            if ok and isinstance(cur, (nn.ModuleList, list)) and len(cur) > 0:
                return list(cur)
        # If we cannot locate layers (e.g., during lightweight tests with Identity model), return empty list
        return []

    def set_think_mode(self, flag: bool):
        # Back-compat toggle; sets context var globally for this thread
        if flag:
            self._token = self._think_flag.set(True)
        else:
            try:
                if hasattr(self, "_token"):
                    self._think_flag.reset(self._token)
            except Exception:
                self._think_flag.set(False)

    def think(self):
        scaffold = self
        class _Ctx:
            def __enter__(self_):
                self_._tok = scaffold._think_flag.set(True)
                return self_
            def __exit__(self_, exc_type, exc, tb):
                try:
                    scaffold._think_flag.reset(self_._tok)
                except Exception:
                    scaffold._think_flag.set(False)
        return _Ctx()

    def forward(self, *args, **kwargs):
        # pass-through (generation will call into self.model)
        return self.model(*args, **kwargs)

    def close(self):
        # remove hooks to prevent leaks
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

# Context variable for per-token think mask provided at train time
_ctx_think_mask: ContextVar = ContextVar("_ctx_think_mask", default=None)

class HardConcreteGate(nn.Module):
    """
    Lightweight hard-concrete style gate producing s in [0,1] per token.
    Uses logistic sampling during training; deterministic sigmoid at eval.
    """
    def __init__(self, init_log_alpha: float = -6.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha, dtype=torch.float32))

    def forward(self, x: torch.Tensor, training: bool = True):
        # Gate shape matches [B,T,1] for broadcasting over H
        shape = x[..., :1].shape
        if training:
            u = torch.rand(shape, device=x.device, dtype=torch.float32)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha))
        else:
            s = torch.sigmoid(self.log_alpha).expand(shape)
        s = s.clamp(0.0, 1.0)
        return x * s, s


def attach_residual_adapters(model, hidden_size: int, num_layers: int,
                             rank: int = 8, layers: Optional[Sequence[int]] = None,
                             init_gate: float = 0.0) -> IntrospectionScaffold:
    cfg = ResidualAdapterConfig(hidden_size=hidden_size, rank=rank, layers=layers, init_gate=init_gate)
    return IntrospectionScaffold(model, hidden_size, num_layers, cfg)
