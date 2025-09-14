# tina/serve.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Sequence
from pathlib import Path
import inspect
import json
import threading
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import torch.nn as nn
import re

from .tokenizer_utils import ensure_reasoning_tokens, STOP_SEQUENCES

__all__ = [
    "EngineConfig",
    "IntrospectiveEngine",
    "StopOnTags",
    "SlackStop",
    "_extract_answer",
    "_count_think_tokens",
]

@dataclass
class EngineConfig:
    """Runtime engine configuration.

    Fields
    - visible_cot: default visibility of <think> (False hides CoT)
    - max_think_tokens / budget_cap / min_think_tokens: THINK budget controls
    - taps: indices for metacog taps (legacy three-tap mode)
    - linked_all_layers / proj_dim / agg / dump_per_layer: metacog settings passed to MetacogConfig
      (see tina.metacog_heads.MetacogConfig). When linked_all_layers=True, heads pool last-token
      hidden states from every decoder layer and aggregate via 'attn' or 'mean'. When dump_per_layer=True,
      per-layer diagnostics (alpha, per-layer plans) are returned for debugging/telemetry.
    - conditioning_film*: optional FiLM modulation settings (applied during THINK only)
    - calibration_path: optional JSON path for confidence temperature, plan thresholds, budget clip
    - style_tag: optional style hint injected before THINK
    """
    # reasoning behavior
    visible_cot: Optional[bool] = None
    max_think_tokens: int = 256
    use_dynamic_budget: bool = True
    budget_cap: int = 256
    # minimum tokens to spend thinking (guard rails)
    min_think_tokens: int = 8
    # adapters
    side_rank: int = 8
    adapter_layers: Optional[List[int]] = None   # None → all
    # taps for metacog heads
    taps: Sequence[int] = (6, 10, 14)
    # linked-all-layers metacog options
    linked_all_layers: bool = False
    proj_dim: int = 128
    agg: str = "attn"
    dump_per_layer: bool = False
    # FiLM conditioning (think-only)
    conditioning_film: bool = False
    conditioning_film_scale: float = 0.05
    conditioning_per_layer: bool = True
    # optional calibration blob path (JSON): {"conf_temp": float}
    calibration_path: Optional[str] = None
    # optional reasoning style tag to hint at serve (e.g., 'checklist','explainer')
    style_tag: Optional[str] = None
    # Multi-expert + feedback options
    num_experts: int = 1
    feedback: bool = False
    feedback_dim: Optional[int] = None
    # Apply feedback gate to this decoder layer index during THINK (None disables)
    feedback_apply_layer: Optional[int] = None

def _extract_answer(body: str, include_think: bool = False) -> str:
    def _slice(s: str, open_t: str, close_t: str) -> str:
        i = s.find(open_t)
        j = s.find(close_t, i + len(open_t)) if i != -1 else -1
        return s[i + len(open_t):j] if (i != -1 and j != -1) else s
    if include_think:
        return body.strip()
    ans = _slice(body, "<answer>", "</answer>")
    return ans.strip()

class StopOnTags(StoppingCriteria):
    def __init__(self, tokenizer, stop_strs=("</answer>",), max_new: Optional[int] = None):
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strs]
        self.max_new = max_new
        self.seen = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.seen += 1
        if self.max_new is not None and self.seen >= self.max_new:
            return True
        for sid in self.stop_ids:
            k = len(sid)
            if k and input_ids.size(1) >= k and input_ids[0, -k:].tolist() == sid:
                return True
        return False

class SlackStop(StoppingCriteria):
    """
    Soft cap for THINK: allow up to budget * (1 + slack_ratio) generated tokens (since base_len)
    before stopping if the closing tag has not appeared yet.
    """
    def __init__(self, *, base_len: int, budget: int, slack_ratio: float = 0.2):
        self.base_len = int(base_len)
        self.budget = int(max(0, budget))
        self.slack_ratio = float(max(0.0, slack_ratio))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        cur_new = int(max(0, input_ids.size(1) - self.base_len))
        limit = int(self.budget * (1.0 + self.slack_ratio))
        return cur_new > limit

def _to_list(x):
    try:
        import torch as _t
        if _t.is_tensor(x):
            return list(x.view(-1).detach().cpu().tolist())
    except Exception:
        pass
    return list(x)

def _count_think_tokens(gen_ids, close_ids_list, base_len: int, cap: int, slack_ratio: float) -> int:
    """
    Count number of <think> tokens generated after base_len and up to the earliest of:
    - the first occurrence of any closing tag in close_ids_list, or
    - the soft-cap boundary: base_len + cap * (1 + slack_ratio), clamped by actual generated length.

    Robust to empty/None close sequences and negative base_len.
    """
    g = _to_list(gen_ids)
    n = len(g)
    base = max(0, int(base_len))
    cap_i = max(0, int(cap))
    limit = int(cap_i * (1.0 + float(slack_ratio)))
    end_soft = min(n, base + limit)
    earliest = end_soft
    if close_ids_list:
        for cid in close_ids_list:
            if not cid:
                continue
            k = len(cid)
            if k <= 0 or base + k > n:
                continue
            j = base
            while j <= n - k:
                if g[j:j + k] == cid:
                    if j < earliest:
                        earliest = j
                    break
                j += 1
    used = max(0, earliest - base)
    return used


class DecodingController:
    """
    Translate metacog policy signals to generation settings.
    Maps (plan_logits, confidence, policy_vector[, selector_entropy]) → params.
    """
    def __init__(self):
        pass

    @staticmethod
    def _sigmoid(x):
        import math
        try:
            if hasattr(x, 'detach'):
                import torch
                return torch.sigmoid(x)
        except Exception:
            pass
        # scalar fallback
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return 1.0 / (1.0 + math.exp(-v))

    @staticmethod
    def _softplus(x):
        try:
            import torch
            return torch.nn.functional.softplus(x)
        except Exception:
            import math
            v = float(x)
            return math.log1p(math.exp(v))

    def compute(self, *, plan_logits=None, confidence=None, policy_raw=None, selector_entropy=None, B_max=None):
        """
        Returns a dict with: temperature, top_p, repetition_penalty, think_ratio.
        Uses provided tensors when available; falls back to scalars.
        """
        import torch
        # Defaults
        out = {
            'temperature': 0.6,
            'top_p': 0.95,
            'repetition_penalty': 1.1,
            'think_ratio': 1.0,
        }
        if policy_raw is not None:
            try:
                pv = policy_raw[0] if (torch.is_tensor(policy_raw) and policy_raw.dim() >= 2) else policy_raw
                # mapping per spec: temp/top_p/rep/think_ratio from raw
                t = 0.3 + 1.7 * self._sigmoid(pv[0])
                tp = 0.2 + 0.78 * self._sigmoid(pv[1])
                rp = 0.8 + 1.2 * self._softplus(pv[2])
                tr = self._sigmoid(pv[3])
                # clamp sensible bounds
                if torch.is_tensor(t):
                    t = t.clamp(0.1, 2.5)
                    tp = tp.clamp(0.05, 1.0)
                    rp = rp.clamp(0.8, 3.0)
                    tr = tr.clamp(0.05, 1.0)
                else:
                    t = max(0.1, min(2.5, float(t)))
                    tp = max(0.05, min(1.0, float(tp)))
                    rp = max(0.8, min(3.0, float(rp)))
                    tr = max(0.05, min(1.0, float(tr)))
                out.update({'temperature': t, 'top_p': tp, 'repetition_penalty': rp, 'think_ratio': tr})
            except Exception:
                pass
        # Confidence and entropy can adjust conservativeness (optional heuristic)
        try:
            if torch.is_tensor(confidence):
                c = float(confidence.detach().view(-1)[0].item())
            else:
                c = float(confidence) if confidence is not None else None
        except Exception:
            c = None
        try:
            if torch.is_tensor(selector_entropy):
                se = float(selector_entropy.detach().view(-1)[0].item())
            elif selector_entropy is not None:
                se = float(selector_entropy)
            else:
                se = None
        except Exception:
            se = None
        # Slightly increase temperature if low confidence/high entropy; else be more conservative
        try:
            if c is not None:
                if c < 0.3:
                    out['temperature'] = (out['temperature'] if not torch.is_tensor(out['temperature']) else float(out['temperature'])) * 1.1
                elif c > 0.8:
                    out['temperature'] = (out['temperature'] if not torch.is_tensor(out['temperature']) else float(out['temperature'])) * 0.9
        except Exception:
            pass
        out['selector_entropy'] = se
        if B_max is not None:
            try:
                out['B_max'] = int(B_max if not torch.is_tensor(B_max) else int(B_max.detach().view(-1)[0].item()))
            except Exception:
                pass
        return out

class IntrospectiveEngine:
    """
    Wrap a (peft) CausalLM with additive scaffolding and serve visible/hidden CoT.
    """
    def __init__(self, model, tokenizer, cfg: EngineConfig, hidden_size: int, num_layers: int):
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg
        self.last_stats: Dict[str, Any] = {}
        self._gen_lock = threading.RLock()
        self._conf_temp: Optional[float] = None
        self._plan_thresholds: Optional[Dict[str, float]] = None
        self._budget_clip: Optional[int] = None
        self._decode_params: Dict[str, Any] = {}
        # Lazy imports to avoid heavy deps at module import time
        from .side_adapters import attach_residual_adapters, IntrospectionScaffold
        from .metacog_heads import MetacogHeads, MetacogConfig

        # Ensure CoT tokens and default stop sequences are atomic
        self.reason_ids = ensure_reasoning_tokens(self.tok, self.model, extra=STOP_SEQUENCES)

        # Attach residual adapters (gated; start at 0 → no behavior change)
        self.scaffold = attach_residual_adapters(
            self.model, hidden_size=hidden_size, num_layers=num_layers,
            rank=cfg.side_rank, layers=cfg.adapter_layers, init_gate=0.0
        )

        # Light metacognitive heads (you'll train later)
        self.metacog = MetacogHeads(MetacogConfig(
            hidden_size=hidden_size,
            taps=cfg.taps,
            proj_dim=int(getattr(cfg, "proj_dim", 128)),
            head_temp=1.0,
            budget_head_temperature=1.0,
            linked_all_layers=bool(getattr(cfg, "linked_all_layers", False)),
            agg=str(getattr(cfg, "agg", "attn")),
            dump_per_layer=bool(getattr(cfg, "dump_per_layer", False)),
            # new: experts + feedback
            num_experts=int(getattr(cfg, "num_experts", 1) or 1),
            feedback=bool(getattr(cfg, "feedback", False)),
            feedback_dim=(getattr(cfg, "feedback_dim", None)),
        ))

        # Ensure side modules live on the same device as the base model
        try:
            _p = next(self.model.parameters())
            model_device = _p.device
            model_dtype = _p.dtype
        except StopIteration:
            model_device = getattr(self.model, "device", torch.device("cpu"))
            model_dtype = getattr(self.model, "dtype", torch.float32)
        # Align device and dtype with the base model
        self.metacog.to(device=model_device, dtype=model_dtype)
        # Move adapter params as well (the wrapped model's params stay where they are)
        self.scaffold.to(device=model_device, dtype=model_dtype)

        # Optional FiLM projection heads for conditioning
        self._film_enabled = bool(getattr(self.cfg, "conditioning_film", False))
        self._film_scale = float(getattr(self.cfg, "conditioning_film_scale", 0.05) or 0.05)
        self._film_per_layer = bool(getattr(self.cfg, "conditioning_per_layer", True))
        self._film_gamma_head = None
        self._film_beta_head = None
        self._film_gamma_layers = None
        self._film_beta_layers = None
        self._last_state_g: Optional[torch.Tensor] = None
        if self._film_enabled:
            D = int(getattr(self.metacog.cfg, "proj_dim", 128))
            if self._film_per_layer:
                self._film_gamma_layers = nn.ModuleList([nn.Linear(D, hidden_size) for _ in range(num_layers)])
                self._film_beta_layers = nn.ModuleList([nn.Linear(D, hidden_size) for _ in range(num_layers)])
                self._film_gamma_layers.to(device=model_device, dtype=model_dtype)
                self._film_beta_layers.to(device=model_device, dtype=model_dtype)
            else:
                self._film_gamma_head = nn.Linear(D, hidden_size).to(device=model_device, dtype=model_dtype)
                self._film_beta_head = nn.Linear(D, hidden_size).to(device=model_device, dtype=model_dtype)

        # Resolve visible_cot default, stop sequences, and calibration path from service config if not explicitly provided
        try:
            if self.cfg.visible_cot is None:
                root = Path(__file__).resolve().parents[1]
                svc = root / "config" / "service_config.json"
                vis_default = False
                slack_ratio = 0.2
                if svc.exists():
                    with svc.open("r", encoding="utf-8") as f:
                        svc_cfg = json.load(f)
                        if isinstance(svc_cfg.get("visible_cot_default"), bool):
                            vis_default = bool(svc_cfg["visible_cot_default"])
                        # Stop sequences (answer and think)
                        self._stop_answer_tags = tuple(svc_cfg.get("stop_sequences") or ["</answer>"])
                        self._stop_think_tags = tuple(svc_cfg.get("think_stop_sequences") or ["</think>"])
                        # Register any service-defined stop tags as atomic tokens as well
                        try:
                            ensure_reasoning_tokens(self.tok, self.model, extra=list(self._stop_answer_tags) + list(self._stop_think_tags))
                        except Exception:
                            pass
                        # Calibration path if not provided explicitly
                        if not getattr(self.cfg, "calibration_path", None):
                            cpath = svc_cfg.get("confidence_calibration_path")
                            if isinstance(cpath, str) and cpath:
                                self.cfg.calibration_path = cpath
                        # Soft-cap slack ratio
                        try:
                            slack_ratio = float(svc_cfg.get("soft_cap_slack_ratio", 0.2))
                        except Exception:
                            slack_ratio = 0.2
                self.cfg.visible_cot = vis_default
                self.last_stats["visible_cot_default"] = vis_default
                self._slack_ratio = float(slack_ratio)
                self.last_stats["soft_cap_slack_ratio"] = self._slack_ratio
            else:
                # also expose the configured default for telemetry if available
                root = Path(__file__).resolve().parents[1]
                svc = root / "config" / "service_config.json"
                if svc.exists():
                    with svc.open("r", encoding="utf-8") as f:
                        svc_cfg = json.load(f)
                        if isinstance(svc_cfg.get("visible_cot_default"), bool):
                            self.last_stats["visible_cot_default"] = bool(svc_cfg["visible_cot_default"])
                        self._stop_answer_tags = tuple(svc_cfg.get("stop_sequences") or ["</answer>"])
                        self._stop_think_tags = tuple(svc_cfg.get("think_stop_sequences") or ["</think>"])
                        try:
                            ensure_reasoning_tokens(self.tok, self.model, extra=list(self._stop_answer_tags) + list(self._stop_think_tags))
                        except Exception:
                            pass
                        try:
                            self._slack_ratio = float(svc_cfg.get("soft_cap_slack_ratio", 0.2))
                        except Exception:
                            self._slack_ratio = 0.2
                        self.last_stats["soft_cap_slack_ratio"] = self._slack_ratio
        except Exception:
            # fallback: keep current value and omit default
            self._stop_answer_tags = ("</answer>",)
            self._stop_think_tags = ("</think>",)
            self._slack_ratio = 0.2

        # Tap registration: cache last-token states from chosen layers
        dec_layers = self.scaffold._get_decoder_layers(self.model)
        for i, layer in enumerate(dec_layers):
            if bool(getattr(cfg, "linked_all_layers", False)) or (i in cfg.taps):
                def _hook(mod, inp, out, _i=i):
                    # out can be Tensor or a tuple containing hidden states as first element
                    hs = None
                    if torch.is_tensor(out):
                        hs = out
                    elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                        hs = out[0]
                    elif hasattr(out, "hidden_states") and torch.is_tensor(getattr(out, "hidden_states")):
                        hs = out.hidden_states
                    # Only register if we found a tensor
                    if hs is not None:
                        self.metacog.register_tap(_i, hs.detach())
                    return out
                layer.register_forward_hook(_hook)

        # Optional feedback gate hook (applied during THINK only)
        self._gate_hook = None
        try:
            if bool(getattr(cfg, "feedback", False)) and getattr(cfg, "feedback_apply_layer", None) is not None:
                idx = int(cfg.feedback_apply_layer)
                layers = self.scaffold._get_decoder_layers(self.model)
                if 0 <= idx < len(layers):
                    def _gate_hook_fn(module, inputs, output):
                        # apply only during THINK
                        try:
                            if not self.scaffold._think_flag.get():
                                return output
                        except Exception:
                            pass
                        gate = getattr(self.metacog, 'last_gate', None)
                        if gate is None or (not torch.is_tensor(gate)):
                            return output
                        # Determine output hidden state tensor
                        def _apply(h: torch.Tensor) -> torch.Tensor:
                            try:
                                g = gate
                                # Use first sample if batch mismatch
                                if g.dim() == 2 and g.shape[0] != h.shape[0]:
                                    g = g[:1]
                                if g.dim() == 2:
                                    # expect [B,H]
                                    if g.shape[-1] != h.shape[-1]:
                                        return h
                                    g = g.to(device=h.device, dtype=h.dtype)
                                    return h * g.unsqueeze(1)
                                elif g.dim() == 1:
                                    if g.numel() != h.shape[-1]:
                                        return h
                                    g = g.to(device=h.device, dtype=h.dtype)
                                    return h * g.view(1,1,-1)
                            except Exception:
                                return h
                            return h
                        if torch.is_tensor(output):
                            return _apply(output)
                        if isinstance(output, tuple) and len(output) > 0 and torch.is_tensor(output[0]):
                            new_h = _apply(output[0])
                            return (new_h, *output[1:])
                        if isinstance(output, list) and len(output) > 0 and torch.is_tensor(output[0]):
                            new_h = _apply(output[0])
                            rest = list(output[1:])
                            return [new_h, *rest]
                        return output
                    self._gate_hook = layers[idx].register_forward_hook(_gate_hook_fn)
        except Exception:
            self._gate_hook = None

        # Load optional calibration blob (explicit path or from service config)
        try:
            cal_path: Optional[str] = None
            if cfg.calibration_path:
                cal_path = cfg.calibration_path
            else:
                root = Path(__file__).resolve().parents[1]
                svc = root / "config" / "service_config.json"
                if svc.exists():
                    with svc.open("r", encoding="utf-8") as f:
                        svc_cfg = json.load(f)
                    cpath = svc_cfg.get("confidence_calibration_path")
                    if isinstance(cpath, str) and cpath:
                        cal_path = cpath
            if cal_path:
                self._load_conf_temp_from_path(cal_path)
        except Exception:
            self._conf_temp = None

        # Parity digest for telemetry
        try:
            self.last_stats["parity_digest"] = {
                "tags": self.reason_ids,
                "stop_answer": list(getattr(self, "_stop_answer_tags", ("</answer>",))),
                "stop_think": list(getattr(self, "_stop_think_tags", ("</think>",))),
                "visible_cot": bool(self.cfg.visible_cot),
                "conf_temp": self._conf_temp,
                "plan_thresholds": getattr(self, "_plan_thresholds", None),
                "budget_clip": getattr(self, "_budget_clip", None),
                "soft_cap_slack_ratio": float(getattr(self, "_slack_ratio", 0.2)),
            }
        except Exception:
            pass

    def _estimate_budget(self, input_ids) -> int:
        # Dry forward to populate taps, then ask metacog for budget
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           use_cache=False, output_hidden_states=True, return_dict=True)
            # Fallback: if hooks didn't populate taps (e.g., wrapper output type), use hidden_states
            try:
                has_cache = bool(getattr(self.metacog, "_tap_cache", {})) and any(self.metacog._tap_cache.values())
                if (not has_cache) and isinstance(getattr(outputs, "hidden_states", None), (list, tuple)):
                    self.metacog.clear_cache()
                    hs_list = list(outputs.hidden_states)
                    # HF typically returns embeddings at index 0, then per-layer states
                    start = 1 if len(hs_list) >= 2 else 0
                    for li, hs in enumerate(hs_list[start:]):
                        if torch.is_tensor(hs):
                            self.metacog.register_tap(li, hs)
            except Exception:
                pass
        # Call metacog heads safely: some tests monkey-patch forward without 'return_state'
        try:
            sig = inspect.signature(self.metacog.forward)
            has_return_state = 'return_state' in sig.parameters
        except Exception:
            has_return_state = False
        _kwargs = {"B_max": self.cfg.budget_cap}
        if has_return_state:
            _kwargs["return_state"] = True
        out = self.metacog(**_kwargs)
        # Save expert mix and feedback gate for downstream use/telemetry
        try:
            ew = out.get('expert_weights') if isinstance(out, dict) else None
            if torch.is_tensor(ew):
                self.last_stats['expert_weights'] = ew[0].detach().float().cpu().tolist()
        except Exception:
            pass
        try:
            fg = out.get('feedback_gate') if isinstance(out, dict) else None
            if torch.is_tensor(fg):
                # store for gating hook
                self.metacog.last_gate = fg.detach()
                # also expose summary + first vector
                self.last_stats['feedback_gate_mean'] = float(fg.mean().item())
                try:
                    self.last_stats['feedback_gate'] = fg[0].detach().float().cpu().tolist()
                except Exception:
                    pass
        except Exception:
            self.metacog.last_gate = None
        # Compute decoding parameters from policy
        try:
            policy_raw = out.get('policy_raw') if isinstance(out, dict) else None
            conf = out.get('confidence') if isinstance(out, dict) else None
            H_e = out.get('H_e') if isinstance(out, dict) else None
            Bm = out.get('budget') if isinstance(out, dict) else None
            if torch.is_tensor(Bm):
                Bm_val = int(Bm[0].item())
            else:
                Bm_val = None
            ctrl = DecodingController()
            dec = ctrl.compute(plan_logits=out.get('plan_logits') if isinstance(out, dict) else None,
                               confidence=conf,
                               policy_raw=policy_raw,
                               selector_entropy=(H_e.mean() if torch.is_tensor(H_e) else H_e),
                               B_max=Bm_val)
            # Normalize tensor values to floats
            def _to_float(v):
                try:
                    import torch as _t
                    return float(v.detach().item()) if _t.is_tensor(v) else float(v)
                except Exception:
                    return v
            self._decode_params = {
                'temperature': _to_float(dec.get('temperature')),
                'top_p': _to_float(dec.get('top_p')),
                'repetition_penalty': _to_float(dec.get('repetition_penalty')),
                'think_ratio': _to_float(dec.get('think_ratio')),
                'selector_entropy': _to_float(dec.get('selector_entropy')) if dec.get('selector_entropy') is not None else None,
                'B_max': int(dec.get('B_max')) if dec.get('B_max') is not None else None,
            }
            # Telemetry
            self.last_stats.update({
                'gen_temperature': self._decode_params.get('temperature'),
                'gen_top_p': self._decode_params.get('top_p'),
                'gen_rep_penalty': self._decode_params.get('repetition_penalty'),
                'think_ratio': self._decode_params.get('think_ratio'),
                'B_max': self._decode_params.get('B_max'),
            })
            if self._decode_params.get('selector_entropy') is not None:
                self.last_stats['selector_entropy'] = self._decode_params['selector_entropy']
        except Exception:
            self._decode_params = {}
        out = self._postprocess_heads(out)
        try:
            st = out.get("state") if isinstance(out, dict) else None
            if torch.is_tensor(st):
                self._last_state_g = st.detach()
        except Exception:
            self._last_state_g = None
        # record metacog head outputs for observability
        try:
            plan_logits = out.get("plan_logits")
            conf_val = float(out["confidence"][0].item()) if out.get("confidence") is not None else None
            raw_budget = float(out["budget"][0].item()) if out.get("budget") is not None else None
            # Per-layer diagnostics summary if present
            try:
                per = out.get("per_layer") if isinstance(out, dict) else None
                alpha = per.get("alpha") if isinstance(per, dict) else None
                if torch.is_tensor(alpha):
                    a = alpha.detach().float()
                    self.last_stats["alpha_summary"] = {
                        "min": float(a.min().item()),
                        "mean": float(a.mean().item()),
                        "max": float(a.max().item()),
                    }
                # Plan agreement rate (first sample) if available
                try:
                    plog_pl = per.get("plan_logits_pl") if isinstance(per, dict) else None
                    if torch.is_tensor(plan_logits) and torch.is_tensor(plog_pl):
                        agg_arg = torch.argmax(plan_logits[0], dim=-1)
                        pl_args = torch.argmax(plog_pl[0], dim=-1)  # [L]
                        agree = (pl_args == agg_arg).float().mean()
                        self.last_stats["plan_agreement"] = float(agree.item())
                except Exception:
                    pass
            except Exception:
                pass
            # Plan selection with optional thresholds
            plan_idx = None
            plan_label = None
            plan_probs = None
            if plan_logits is not None:
                probs_t = torch.softmax(plan_logits[0], dim=-1)
                plan_probs = probs_t.detach().cpu().tolist()
                labels = getattr(self, "_plan_labels", ["short", "deliberate", "verify", "stop"])
                # Map thresholds to indices; allow keys by label or numeric index strings
                if getattr(self, "_plan_thresholds", None):
                    thr_map: Dict[int, float] = {}
                    unknown: List[str] = []
                    for k, v in self._plan_thresholds.items():
                        idx: Optional[int] = None
                        try:
                            if isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
                                idx = int(k)
                            else:
                                if k in labels:
                                    idx = labels.index(k)
                                else:
                                    idx = None
                        except Exception:
                            idx = None
                        if idx is None or not (0 <= idx < len(plan_probs)):
                            unknown.append(str(k))
                            continue
                        try:
                            thr_map[idx] = float(v)
                        except Exception:
                            unknown.append(str(k))
                    if unknown:
                        raise ValueError(f"Unknown plan keys in calibration: {unknown}; valid: {labels} or indices 0..{len(plan_probs)-1}")

                    # Choose among plans meeting their thresholds; otherwise fallback to a safe default
                    meets = [(i, p) for i, p in enumerate(plan_probs) if (i not in thr_map) or (p >= float(thr_map[i]))]
                    if meets:
                        # take the highest probability among those meeting thresholds
                        best_i, _ = max(meets, key=lambda t: t[1])
                        plan_idx = int(best_i)
                        plan_label = labels[plan_idx] if plan_idx < len(labels) else str(plan_idx)
                    else:
                        # Fallback: prefer 'stop' if present, else argmax
                        if "stop" in labels:
                            plan_idx = labels.index("stop")
                            plan_label = "stop"
                        else:
                            plan_idx = int(torch.argmax(probs_t).item())
                            plan_label = labels[plan_idx] if plan_idx < len(labels) else str(plan_idx)
                else:
                    plan_idx = int(torch.argmax(probs_t).item())
                    plan_label = labels[plan_idx] if plan_idx < len(labels) else str(plan_idx)
        except Exception:
            plan_idx, plan_label, conf_val, raw_budget, plan_probs = None, None, None, None, None
        # pick budget
        b = int(raw_budget) if self.cfg.use_dynamic_budget and raw_budget is not None else self.cfg.budget_cap
        # Optional calibration-time post-hoc budget clip
        if getattr(self, "_budget_clip", None):
            try:
                b = min(b, int(self._budget_clip))
            except Exception:
                pass
        b = max(self.cfg.min_think_tokens, min(b, self.cfg.budget_cap))
        # clear taps for next pass
        self.metacog.clear_cache()
        # save stats
        stats = {
            "plan": plan_idx,
            "confidence": conf_val,
            "conf_prob": conf_val,
            "conf_temp": self._conf_temp,
            "think_budget": b,
            "linked_all_layers": bool(getattr(self.cfg, "linked_all_layers", False)),
        }
        if plan_label is not None:
            stats["plan_label"] = plan_label
        if plan_probs is not None:
            labels = getattr(self, "_plan_labels", ["short", "deliberate", "verify", "stop"])
            stats["plan_probs"] = {labels[i] if i < len(labels) else str(i): float(plan_probs[i]) for i in range(len(plan_probs))}
        if getattr(self, "_budget_clip", None):
            stats["budget_clip"] = int(self._budget_clip)
        self.last_stats.update(stats)
        return b

    def _compute_film_params(self, g: torch.Tensor) -> Optional[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        if not self._film_enabled or (g is None) or (not torch.is_tensor(g)):
            return None
        try:
            with torch.no_grad():
                gv = g[0] if g.dim() == 2 else g.view(-1)
                if self._film_per_layer and self._film_gamma_layers is not None and self._film_beta_layers is not None:
                    gammas = []
                    betas = []
                    for lg, lb in zip(self._film_gamma_layers, self._film_beta_layers):
                        g_vec = torch.tanh(lg(gv))
                        b_vec = torch.tanh(lb(gv))
                        gammas.append(1.0 + self._film_scale * g_vec)
                        betas.append(self._film_scale * b_vec)
                else:
                    g_vec = torch.tanh(self._film_gamma_head(gv)) if self._film_gamma_head is not None else torch.zeros_like(gv)
                    b_vec = torch.tanh(self._film_beta_head(gv)) if self._film_beta_head is not None else torch.zeros_like(gv)
                    vec_g = 1.0 + self._film_scale * g_vec
                    vec_b = self._film_scale * b_vec
                    L = len(getattr(self.scaffold, 'adapters', []))
                    gammas = [vec_g for _ in range(max(1, L))]
                    betas = [vec_b for _ in range(max(1, L))]
                # Telemetry
                try:
                    dg = torch.stack([torch.abs(x - 1.0).mean() for x in gammas]).mean()
                    db = torch.stack([torch.abs(x).mean() for x in betas]).mean()
                    self.last_stats["film_gamma_mean_delta"] = float(dg.item())
                    self.last_stats["film_beta_mean_abs"] = float(db.item())
                except Exception:
                    pass
                return gammas, betas
        except Exception:
            return None

    @staticmethod
    def apply_conf_temperature(prob: Any, temp: float) -> Any:
        """Apply temperature to a probability by scaling its logit: sigmoid(logit(p)/T).
        Supports float or torch.Tensor.
        """
        try:
            import math
            if isinstance(prob, (int, float)):
                p = max(1e-8, min(1.0 - 1e-8, float(prob)))
                logit = math.log(p / (1.0 - p))
                t = max(float(temp), 1e-6)
                z = logit / t
                # sigmoid
                return 1.0 / (1.0 + math.exp(-z))
            elif torch.is_tensor(prob):
                p = prob.clamp(1e-8, 1 - 1e-8)
                logit = torch.log(p / (1 - p))
                t = torch.tensor(float(max(temp, 1e-6)), dtype=logit.dtype, device=logit.device)
                return torch.sigmoid(logit / t)
            else:
                return prob
        except Exception:
            return prob

    def _load_conf_temp_from_path(self, path: str) -> None:
        try:
            p = Path(path)
            if p.exists():
                with p.open("r", encoding="utf-8") as f:
                    blob = json.load(f)
                ct = blob.get("conf_temp")
                if isinstance(ct, (int, float)) and ct > 0:
                    self._conf_temp = float(ct)
                th = blob.get("plan_thresholds")
                if isinstance(th, dict) and th:
                    self._plan_thresholds = {str(k): float(v) for k, v in th.items() if v is not None}
                # support both keys for budget clip
                bc = blob.get("budget_clip", None)
                if bc is None:
                    bc = blob.get("budget_posthoc_clip")
                if isinstance(bc, (int, float)) and bc:
                    self._budget_clip = int(bc)
                # Optional budget head temperature for metacog head
                bht = blob.get("budget_head_temp")
                if isinstance(bht, (int, float)) and hasattr(self, "metacog") and hasattr(self.metacog, "set_budget_temp"):
                    try:
                        self.metacog.set_budget_temp(float(bht))
                    except Exception:
                        pass
        except Exception:
            self._conf_temp = None

    def _postprocess_heads(self, out: Dict[str, Any]) -> Dict[str, Any]:
        """Apply any serve-time calibration/transforms to metacog head outputs."""
        try:
            if (
                out is not None
                and isinstance(out, dict)
                and "confidence" in out
                and self._conf_temp is not None
                and self._conf_temp > 0
            ):
                c = out["confidence"]
                out["confidence"] = self.apply_conf_temperature(c, self._conf_temp)
        except Exception:
            pass
        return out

    @staticmethod
    def assemble_cot_output(think_text: str, answer_text: str, visible_cot: bool) -> str:
        """Compose final user-visible output from think and answer generations.
        Ensures hidden mode returns only the <answer> body without tags; visible mode returns raw concatenation.
        """
        if visible_cot:
            return (think_text + answer_text).strip()
        ans_open, ans_close = "<answer>", "</answer>"
        body = (think_text or "") + (answer_text or "")
        start = body.find(ans_open)
        end = body.find(ans_close, start + len(ans_open)) if start != -1 else -1
        if start != -1 and end != -1:
            return body[start + len(ans_open):end].strip()
        # Fallback: trim up to the first closing if present, else return stripped answer_text
        if ans_close in answer_text:
            out = answer_text.split(ans_close)[0].replace(ans_open, "").strip()
        else:
            out = answer_text.strip()
        # Strategy invariance: strip any control tokens like <strategy:...> from final answers
        try:
            out = re.sub(r"<strategy:[^>]+>", "", out).strip()
        except Exception:
            pass
        # Also strip any decomposition/think tags if they accidentally appear in answer text
        try:
            for tag in ("<think>", "</think>", "<plan>", "</plan>", "<exec>", "</exec>", "<eval>", "</eval>"):
                out = out.replace(tag, "")
            out = out.strip()
        except Exception:
            pass
        return out

    def generate_cot(self, messages: List[Dict[str, str]], max_new_tokens: int = 512,
                     temperature: float = 0.7, top_p: float = 0.95, repetition_penalty: float = 1.1,
                     ignore_eos: bool = False, stream: bool = False,
                     style_tag: Optional[str] = None) -> str:
        # serialize per-engine to avoid cross-request hook state issues
        with self._gen_lock:
            # initialize leakage counter for this request
            try:
                self.last_stats["leakage"] = 0
            except Exception:
                pass
            # Build chat with "<think>\n" prompt (your tokenizer template already does this if add_generation_prompt=True)
            # Resolve a safe device for inputs
            try:
                dev = getattr(self.model, 'device')
            except Exception:
                dev = None
            if dev is None:
                try:
                    dev = next(self.model.parameters()).device
                except Exception:
                    dev = torch.device('cpu')
            enc = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            input_ids = enc.to(dev)
            attention_mask = torch.ones_like(input_ids)
            # Optional style control: inject <style:TAG> just before THINK
            use_style = style_tag if style_tag is not None else getattr(self.cfg, "style_tag", None)
            try:
                if use_style:
                    hint = self.tok(f"<style:{use_style}>", add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
                    input_ids = torch.cat([input_ids, hint], dim=1)
                    attention_mask = torch.ones_like(input_ids)
                    self.last_stats["style_tag"] = str(use_style)
            except Exception:
                pass
            eos_id = None if ignore_eos else getattr(self.tok, "eos_token_id", None)

            # Decide reasoning budget and dynamic decoding params
            think_budget = self._estimate_budget(input_ids) if self.cfg.use_dynamic_budget else self.cfg.max_think_tokens
            # Apply optional think_ratio from policy
            try:
                tr = float(self._decode_params.get('think_ratio', 1.0)) if isinstance(getattr(self, '_decode_params', None), dict) else 1.0
                think_budget = max(self.cfg.min_think_tokens, int(think_budget * max(0.05, min(1.0, tr))))
            except Exception:
                pass
            think_budget = min(think_budget, self.cfg.budget_cap)

            # Step 1: THINK — stop at </think> OR budget
            stop_think = StopOnTags(self.tok, tuple(getattr(self, "_stop_think_tags", ("</think>",))), max_new=None)
            soft_cap = SlackStop(base_len=input_ids.shape[1], budget=think_budget, slack_ratio=float(getattr(self, "_slack_ratio", 0.2)))
            text1 = ""
            out1 = None
            if stream and self.cfg.visible_cot:
                streamer1 = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
                # use controller-derived decoding params when available
                _temp = float(self._decode_params.get('temperature', temperature))
                _top_p = float(self._decode_params.get('top_p', top_p))
                _rep = float(self._decode_params.get('repetition_penalty', repetition_penalty))
                gen_kwargs1 = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=_temp > 0,
                    temperature=_temp,
                    top_p=_top_p,
                    repetition_penalty=_rep,
                    eos_token_id=eos_id,
                    pad_token_id=self.tok.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([stop_think, soft_cap]),
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    streamer=streamer1,
                )
                chunks1 = []
                def _worker1():
                    with torch.no_grad():
                        with self.scaffold.think():
                            self.model.generate(**gen_kwargs1)
                t1 = threading.Thread(target=_worker1, daemon=True)
                # Optional FiLM conditioning (think-only)
                try:
                    fp = self._compute_film_params(self._last_state_g)
                    if fp is not None:
                        self.scaffold.set_film(*fp)
                except Exception:
                    pass
                t1.start()
                for piece in streamer1:
                    # stream visible think tokens directly
                    try:
                        import sys as _sys
                        _sys.stdout.write(piece)
                        _sys.stdout.flush()
                    except Exception:
                        pass
                    chunks1.append(piece)
                text1 = "".join(chunks1)
                # Reconstruct out1 ids from streamer
                try:
                    if hasattr(streamer1, "generated_ids") and streamer1.generated_ids:
                        gen_ids = torch.tensor(streamer1.generated_ids[0], device=input_ids.device, dtype=input_ids.dtype).view(1, -1)
                        out1 = torch.cat([input_ids, gen_ids], dim=1)
                    else:
                        # Retokenize streamed text to preserve context
                        think_ids = self.tok(text1, add_special_tokens=False, return_tensors="pt").input_ids.to(input_ids.device)
                        out1 = torch.cat([input_ids, think_ids], dim=1)
                except Exception:
                    try:
                        think_ids = self.tok(text1, add_special_tokens=False, return_tensors="pt").input_ids.to(input_ids.device)
                        out1 = torch.cat([input_ids, think_ids], dim=1)
                    except Exception:
                        out1 = input_ids
            else:
                with torch.no_grad():
                    # Optional FiLM conditioning (think-only)
                    try:
                        fp = self._compute_film_params(self._last_state_g)
                        if fp is not None:
                            self.scaffold.set_film(*fp)
                    except Exception:
                        pass
                    with self.scaffold.think():
                        _temp = float(self._decode_params.get('temperature', temperature))
                        _top_p = float(self._decode_params.get('top_p', top_p))
                        _rep = float(self._decode_params.get('repetition_penalty', repetition_penalty))
                        out1 = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,  # upper bound; stopping criteria will cut earlier
                        do_sample=_temp > 0,
                        temperature=_temp,
                        top_p=_top_p,
                        repetition_penalty=_rep,
                        eos_token_id=eos_id,
                        pad_token_id=self.tok.pad_token_id,
                        stopping_criteria=StoppingCriteriaList([stop_think, soft_cap]),
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                    )
            # Handle both tensor and GenerateOutput
            seqs1 = getattr(out1, "sequences", out1)
            # Use seqs1 for downstream concatenation as well
            out1 = seqs1
            gen1 = seqs1[0, input_ids.shape[1]:]
            text1 = self.tok.decode(gen1, skip_special_tokens=True)

            # Count think tokens actually used (tokens before closing tag or soft-cap)
            try:
                think_closes = []
                for tag in tuple(getattr(self, "_stop_think_tags", ("</think>",))):
                    try:
                        think_closes.append(self.tok.encode(tag, add_special_tokens=False))
                    except Exception:
                        pass
                used = _count_think_tokens(
                    out1[0], think_closes, base_len=input_ids.shape[1], cap=int(think_budget),
                    slack_ratio=float(getattr(self, "_slack_ratio", 0.2))
                )
            except Exception:
                used = int(gen1.shape[0])
            # Gate activity/coverage across adapters (think-only period)
            try:
                stats = self.scaffold.get_gate_stats()
                # Backward-compatible summaries
                self.last_stats["gate_activity"] = float(stats.get("mean_gate_activity") or 0.0)
                self.last_stats["gate_coverage"] = float(stats.get("mean_gate_coverage") or 0.0)
                # Full gate stats
                self.last_stats["gate_stats"] = stats
            except Exception:
                pass

            # If we didn't get </think>, forcibly close it
            if "</think>" not in text1:
                text1 = text1 + "</think>\n"

            # Adapters auto-disabled when leaving think() context

            # Compose for ANSWER: append the think text (even if hidden, model needs context)
            ans_tok = self.tok("<answer>", add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
            full_prompt_ids = torch.cat([out1, ans_tok], dim=1)
            full_attn = torch.ones_like(full_prompt_ids)

            # Step 2: ANSWER — stop at </answer> or max_new_tokens
            stop_answer = StopOnTags(self.tok, tuple(getattr(self, "_stop_answer_tags", ("</answer>",))), max_new=max_new_tokens)
            text2 = ""
            if stream:
                streamer2 = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
                _temp2 = float(self._decode_params.get('temperature', temperature))
                _top_p2 = float(self._decode_params.get('top_p', top_p))
                _rep2 = float(self._decode_params.get('repetition_penalty', repetition_penalty))
                gen_kwargs2 = dict(
                    input_ids=full_prompt_ids,
                    attention_mask=full_attn,
                    max_new_tokens=max_new_tokens,
                    do_sample=_temp2 > 0,
                    temperature=_temp2,
                    top_p=_top_p2,
                    repetition_penalty=_rep2,
                    eos_token_id=eos_id,
                    pad_token_id=self.tok.pad_token_id,
                    stopping_criteria=StoppingCriteriaList([stop_answer]),
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    streamer=streamer2,
                )
                chunks2 = []
                def _worker2():
                    with torch.no_grad():
                        self.model.generate(**gen_kwargs2)
                t2 = threading.Thread(target=_worker2, daemon=True)
                t2.start()
                for piece in streamer2:
                    try:
                        import sys as _sys
                        _sys.stdout.write(piece)
                        _sys.stdout.flush()
                    except Exception:
                        pass
                    chunks2.append(piece)
                text2 = "".join(chunks2)
            else:
                with torch.no_grad():
                    out2 = self.model.generate(
                        input_ids=full_prompt_ids,
                        attention_mask=full_attn,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        eos_token_id=eos_id,
                        pad_token_id=self.tok.pad_token_id,
                        stopping_criteria=StoppingCriteriaList([stop_answer]),
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                    )
                seqs2 = getattr(out2, "sequences", out2)
                gen2 = seqs2[0, full_prompt_ids.shape[1]:]
                text2 = self.tok.decode(gen2, skip_special_tokens=True)

        # Assemble final user-visible output
        # Save per-run stats, including gate activity if available
        try:
            ga = []
            for m in getattr(self.scaffold, 'adapters', []):
                v = getattr(m, '_last_gate_activity', None)
                if v is not None:
                    ga.append(float(v))
            gate_mean = float(sum(ga)/len(ga)) if ga else None
        except Exception:
            gate_mean = None
        # Record strategy/decomposition tags observed in outputs
        try:
            strat_tags = []
            for s in re.findall(r"<strategy:([^>]+)>", (text1 or "") + (text2 or "")):
                if s not in strat_tags:
                    strat_tags.append(s)
        except Exception:
            strat_tags = []
        try:
            txt_all = (text1 or "") + (text2 or "")
            decomp = {
                "plan": ("<plan>" in txt_all and "</plan>" in txt_all),
                "exec": ("<exec>" in txt_all and "</exec>" in txt_all),
                "eval": ("<eval>" in txt_all and "</eval>" in txt_all),
            }
        except Exception:
            decomp = {}
        self.last_stats.update({
            "answer_tokens_max": max_new_tokens,
            "gate_activity_mean": gate_mean,
            "think_tokens_used": int(locals().get("used", 0) or 0),
            "think_budget": int(locals().get("think_budget", 0) or 0),
            "visible_cot": bool(self.cfg.visible_cot),
            "strategy_tags": strat_tags,
            "decomp_present": decomp,
        })
        if self.cfg.visible_cot:
            return (text1 + text2).strip()
        out_text = _extract_answer((text1 or "") + (text2 or ""), include_think=False)
        # Defensive leakage check: no reasoning tags should remain in hidden mode
        try:
            tags = ("<think>", "</think>", "<answer>", "</answer>", "<plan>", "</plan>", "<exec>", "</exec>", "<eval>", "</eval>")
            if any(t in (out_text or "") for t in tags):
                # record leakage and hard-strip tags as failsafe
                self.last_stats["leakage"] = 1
                for t in tags:
                    out_text = out_text.replace(t, "") if out_text else out_text
                out_text = (out_text or "").strip()
            # Also strip any strategy control tokens from the final answer
            out_text = re.sub(r"<strategy:[^>]+>", "", out_text or "").strip()
        except Exception:
            pass
        return out_text
    
    # ---- Minimal FastAPI app with auth/limits/metrics ----
    @staticmethod
    def create_app(engine: "IntrospectiveEngine", *, bearer_token: Optional[str] = None,
                   auth_required: bool = True,
                   input_tokens_cap: int = 8192, max_new_tokens_cap: int = 1024,
                   rate_limit_per_min: int = 60,
                   max_body_bytes: int = 1_000_000,
                   request_timeout_sec: float = 30.0,
                   allowed_origins: Optional[List[str]] = None,
                   log_prompt_max_chars: int = 256,
                   log_raw_prompt: bool = False):
        try:
            from fastapi import FastAPI, Depends, HTTPException, Request, Response
            from pydantic import BaseModel, Field
            from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
            import uuid, time, re, json, asyncio, logging
            from collections import deque, defaultdict
            from starlette.middleware.base import BaseHTTPMiddleware
            from fastapi.middleware.cors import CORSMiddleware
        except Exception as e:
            raise RuntimeError("FastAPI or prometheus_client not installed") from e

        REQS = Counter("tina_requests_total", "Total requests")
        LAT = Histogram("tina_latency_seconds", "End-to-end latency")

        class ChatRequest(BaseModel):
            messages: list[dict] = Field(..., description="[{role, content}]")
            max_new_tokens: int = 512
            temperature: float = 0.6
            top_p: float = 0.95
            repetition_penalty: float = 1.1
            ignore_eos: bool = False
            visible_cot: bool = False

        app = FastAPI(title="Tina", version="0.1.0")
        rl = defaultdict(lambda: deque())

        # Logger
        logger = logging.getLogger("tina.service")
        if not logger.handlers:
            h = logging.StreamHandler()
            logger.addHandler(h)
        logger.setLevel(logging.INFO)

        # CORS - deny by default unless explicit allowlist
        if allowed_origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=list(allowed_origins),
                allow_credentials=False,
                allow_methods=["POST", "GET"],
                allow_headers=["Authorization", "Content-Type"],
            )

        # Max body size middleware (Content-Length based)
        class _MaxBody(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                try:
                    cl = request.headers.get("content-length")
                    if cl is not None and int(cl) > max_body_bytes:
                        return Response(status_code=413)
                except Exception:
                    pass
                return await call_next(request)

        app.add_middleware(_MaxBody)

        def _redact(s: str) -> str:
            if not s:
                return s
            try:
                s = re.sub(r"[\w\.-]+@[\w\.-]+", "[redacted-email]", s)
                s = re.sub(r"\d{4,}", lambda m: "X" * len(m.group(0)), s)
                return s
            except Exception:
                return s

        def _auth(request: Request):
            if not auth_required:
                return
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth.split(" ", 1)[1].strip() != bearer_token:
                raise HTTPException(status_code=401, detail="Unauthorized")

        def _ratelimit(request: Request):
            if rate_limit_per_min <= 0:
                return
            ip = request.client.host if request.client else "unknown"
            now = time.time()
            dq = rl[ip]
            # evict old
            while dq and now - dq[0] > 60.0:
                dq.popleft()
            if len(dq) >= rate_limit_per_min:
                from fastapi.responses import Response as _Resp
                # Return 429 with Retry-After
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            dq.append(now)

        @app.get("/healthz")
        def healthz():
            return {"ok": True}

        @app.get("/readyz")
        def readyz():
            try:
                _ = next(engine.model.parameters())
                return {"ready": True}
            except Exception:
                return {"ready": False}

        @app.get("/metrics")
        def metrics():
            data = generate_latest()
            return Response(data, media_type=CONTENT_TYPE_LATEST)

        @app.post("/chat")
        async def chat(req: ChatRequest, request: Request, _=Depends(_auth)):
            _ratelimit(request)
            REQS.inc()
            t0 = time.time()
            rid = str(uuid.uuid4())
            # enforce caps
            if req.max_new_tokens > max_new_tokens_cap:
                raise HTTPException(status_code=413, detail="max_new_tokens exceeds cap")
            # crude input token count estimate
            try:
                toks = 0
                for m in req.messages:
                    toks += len(engine.tok(m.get("content") or "", add_special_tokens=False).input_ids)
                if toks > input_tokens_cap:
                    raise HTTPException(status_code=413, detail="input tokens exceed cap")
            except HTTPException:
                raise
            except Exception:
                pass
            # run with timeout
            engine.cfg.visible_cot = bool(req.visible_cot)
            loop = asyncio.get_event_loop()
            def _work():
                return engine.generate_cot(
                    req.messages,
                    max_new_tokens=req.max_new_tokens,
                    temperature=req.temperature,
                    top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                    ignore_eos=req.ignore_eos,
                    stream=False,
                )
            try:
                text = await asyncio.wait_for(loop.run_in_executor(None, _work), timeout=request_timeout_sec)
                status = 200
                return_obj = {"text": text, "stats": engine.last_stats, "request_id": rid}
                return return_obj
            except asyncio.TimeoutError:
                status = 504
                raise HTTPException(status_code=504, detail="Request timed out")
            finally:
                dur = time.time() - t0
                LAT.observe(dur)
                # Prepare redacted/raw prompt preview and metrics
                try:
                    parts = [m.get("content") or "" for m in (req.messages or []) if (m.get("content") and isinstance(m.get("content"), str))]
                    preview = " ".join(parts)
                    if not log_raw_prompt:
                        preview = _redact(preview)
                    preview = preview[:log_prompt_max_chars]
                except Exception:
                    preview = ""
                # token counts
                try:
                    input_len_toks = 0
                    for m in req.messages:
                        input_len_toks += len(engine.tok(m.get("content") or "", add_special_tokens=False).input_ids)
                except Exception:
                    input_len_toks = None
                try:
                    output_len_toks = len(engine.tok(text or "", add_special_tokens=False).input_ids)
                except Exception:
                    output_len_toks = None
                log = {
                    "ts": int(time.time()*1000),
                    "request_id": rid,
                    "path": "/chat",
                    "status": status,
                    "latency_ms": int(dur*1000),
                    "ip": request.client.host if request.client else None,
                    "input_tokens_cap": input_tokens_cap,
                    "max_new_tokens_cap": max_new_tokens_cap,
                    "preview": preview,
                    "input_tokens": input_len_toks,
                    "output_tokens": output_len_toks,
                    "tokens_per_sec": (float(output_len_toks) / dur) if (output_len_toks is not None and dur > 0) else None,
                    "visible_cot": bool(req.visible_cot),
                }
                try:
                    logger.info(json.dumps(log))
                except Exception:
                    pass

        return app
        
