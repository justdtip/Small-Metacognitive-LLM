import torch
import torch.nn as nn

class MetacogHeads(nn.Module):
    """
    Training-friendly stabilized heads.
    - forward(hidden_states_list) returns (plan_logits, budget_cap, conf_logit)
    - float32 projections; logits clamped to [-20, 20]; budget clamped to [0, 4096]
    - confidence scaled by a learnable temperature (for calibration)
    """
    def __init__(self, hidden_size: int, taps=(6, 10, 14), plan_k: int = 3, proj_dim: int | None = None):
        super().__init__()
        self.taps = tuple(int(t) for t in taps)
        hd = hidden_size * len(self.taps)
        self.plan = nn.Linear(hd, plan_k, dtype=torch.float32)
        self.budget = nn.Sequential(
            nn.Linear(hd, 64, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(64, 1, dtype=torch.float32),
        )
        self.confidence = nn.Linear(hd, 1, dtype=torch.float32)
        self._temp = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, hidden_states, B_max: int | None = None, temperature: float | None = None):
        """
        hidden_states: list of tensors (B, T, H) or a single tensor replicated to cover taps.
        Returns:
          - plan_logits: (B, plan_k)
          - budget_cap: (B, 1) clamped to [0, B_max] if provided else [0, 4096]
          - conf_logit: (B, 1) temperature-scaled and clamped to [-20, 20]
        """
        tensor_input = isinstance(hidden_states, torch.Tensor)
        if tensor_input:
            max_tap = max(self.taps) if self.taps else 0
            hidden_states = [hidden_states for _ in range(max_tap + 1)]
        last = [hidden_states[t][:, -1, :] for t in self.taps]
        z = torch.cat(last, dim=-1).to(torch.float32)
        plan = torch.clamp(self.plan(z), -20, 20)
        budget = torch.clamp(self.budget(z), 0, B_max if B_max is not None else 4096)
        # allow override of temperature at call time
        t_use = torch.tensor(float(temperature), dtype=torch.float32) if (temperature is not None) else self._temp
        temp = torch.clamp(t_use, 0.25, 4.0)
        conf = torch.clamp(self.confidence(z) / temp, -20, 20)
        if tensor_input:
            # Back-compat path returning a dict when called with a single tensor
            return {"plan_logits": plan, "budget": budget, "confidence_logit": conf}
        return plan, budget, conf

# Backward-compat for earlier tests: expose a shim matching previous signature
class _LegacyShim(nn.Module):
    def __init__(self, hidden_size: int, taps, proj_dim: int = 128, head_temp: float = 1.0):
        super().__init__()
        # emulate older API by wrapping the new heads; plan_k fixed to 4 for test stability
        self.h = MetacogHeads(hidden_size=hidden_size, taps=taps, plan_k=4)
        with torch.no_grad():
            self.h._temp.copy_(torch.tensor(float(head_temp)))

    def __call__(self, hidden_states, B_max: int = 256, temperature: float = 1.0):
        with torch.no_grad():
            self.h._temp.copy_(torch.tensor(float(temperature)))
        # Expect hidden_states with shape (B,T,H); construct minimal list covering taps
        if isinstance(hidden_states, torch.Tensor):
            # fabricate layers as copies for required tap indices
            max_tap = max(self.h.taps) if self.h.taps else 0
            hs_list = [hidden_states for _ in range(max_tap + 1)]
        else:
            hs_list = hidden_states
        plan, budget, conf = self.h(hs_list)
        # Return dict with 'budget' key to satisfy existing tests
        out = {
            "plan_logits": plan,
            "budget": torch.clamp(budget, 0, B_max),
            "confidence_logit": conf,
        }
        return out

# Alias used by earlier tests
LegacyMetacogHeads = _LegacyShim
