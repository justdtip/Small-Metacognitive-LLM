# tina/serve.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Sequence
from pathlib import Path
import json
import threading
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from .tokenizer_utils import ensure_reasoning_tokens

@dataclass
class EngineConfig:
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
    # optional calibration blob path (JSON): {"conf_temp": float}
    calibration_path: Optional[str] = None

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
        # Lazy imports to avoid heavy deps at module import time
        from .side_adapters import attach_residual_adapters, IntrospectionScaffold
        from .metacog_heads import MetacogHeads, MetacogConfig

        # Ensure CoT tokens
        self.reason_ids = ensure_reasoning_tokens(self.tok, self.model)

        # Attach residual adapters (gated; start at 0 → no behavior change)
        self.scaffold = attach_residual_adapters(
            self.model, hidden_size=hidden_size, num_layers=num_layers,
            rank=cfg.side_rank, layers=cfg.adapter_layers, init_gate=0.0
        )

        # Light metacognitive heads (you'll train later)
        self.metacog = MetacogHeads(MetacogConfig(hidden_size=hidden_size, taps=cfg.taps))

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
            if i in cfg.taps:
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
                "soft_cap_slack_ratio": float(getattr(self, "_slack_ratio", 0.2)),
            }
        except Exception:
            pass

    def _estimate_budget(self, input_ids) -> int:
        # Dry forward to populate taps, then ask metacog for budget
        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           use_cache=False, output_hidden_states=True, return_dict=True)
        out = self.metacog(B_max=self.cfg.budget_cap)
        out = self._postprocess_heads(out)
        # record metacog head outputs for observability
        try:
            plan_logits = out.get("plan_logits")
            conf_val = float(out["confidence"][0].item()) if out.get("confidence") is not None else None
            raw_budget = float(out["budget"][0].item()) if out.get("budget") is not None else None
            # Plan selection with optional thresholds
            plan_idx = None
            plan_label = None
            plan_probs = None
            if plan_logits is not None:
                probs_t = torch.softmax(plan_logits[0], dim=-1)
                plan_probs = probs_t.detach().cpu().tolist()
                labels = getattr(self, "_plan_labels", ["short", "deliberate", "verify", "stop"])
                if self._plan_thresholds:
                    chosen = None
                    best_p = -1.0
                    for i, p in enumerate(plan_probs):
                        name = labels[i] if i < len(labels) else str(i)
                        thr = self._plan_thresholds.get(name)
                        if thr is not None and p >= float(thr) and p > best_p:
                            chosen = (i, name, p)
                            best_p = p
                    if chosen is not None:
                        plan_idx, plan_label = int(chosen[0]), str(chosen[1])
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
                bc = blob.get("budget_posthoc_clip")
                if isinstance(bc, (int, float)) and bc:
                    self._budget_clip = int(bc)
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
            return answer_text.split(ans_close)[0].replace(ans_open, "").strip()
        return answer_text.strip()

    def generate_cot(self, messages: List[Dict[str, str]], max_new_tokens: int = 512,
                     temperature: float = 0.7, top_p: float = 0.95, repetition_penalty: float = 1.1,
                     ignore_eos: bool = False, stream: bool = False) -> str:
        # serialize per-engine to avoid cross-request hook state issues
        with self._gen_lock:
            # Build chat with "<think>\n" prompt (your tokenizer template already does this if add_generation_prompt=True)
            enc = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            input_ids = enc.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
            eos_id = None if ignore_eos else getattr(self.tok, "eos_token_id", None)

            # Decide reasoning budget
            think_budget = self._estimate_budget(input_ids) if self.cfg.use_dynamic_budget else self.cfg.max_think_tokens
            think_budget = min(think_budget, self.cfg.budget_cap)

            # Step 1: THINK — stop at </think> OR budget
            stop_think = StopOnTags(self.tok, tuple(getattr(self, "_stop_think_tags", ("</think>",))), max_new=None)
            soft_cap = SlackStop(base_len=input_ids.shape[1], budget=think_budget, slack_ratio=float(getattr(self, "_slack_ratio", 0.2)))
            text1 = ""
            out1 = None
            if stream and self.cfg.visible_cot:
                streamer1 = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
                gen_kwargs1 = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
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
                    with self.scaffold.think():
                        out1 = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,  # upper bound; stopping criteria will cut earlier
                        do_sample=temperature > 0,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
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

            # Count think tokens actually used (tokens before closing tag if present)
            try:
                close_ids = self.tok.encode("</think>", add_special_tokens=False)
                g = gen1.tolist()
                used = len(g)
                k = len(close_ids)
                if k > 0:
                    for i in range(max(0, len(g) - k), -1, -1):
                        if g[i:i + k] == close_ids:
                            used = i
                            break
            except Exception:
                used = int(gen1.shape[0])

            # If we didn't get </think>, forcibly close it
            if "</think>" not in text1:
                text1 = text1 + "</think>\n"

            # Adapters auto-disabled when leaving think() context

            # Compose for ANSWER: append the think text (even if hidden, model needs context)
            ans_tok = self.tok("<answer>", add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.device)
            full_prompt_ids = torch.cat([out1, ans_tok], dim=1)
            full_attn = torch.ones_like(full_prompt_ids)

            # Step 2: ANSWER — stop at </answer> or max_new_tokens
            stop_answer = StopOnTags(self.tok, tuple(getattr(self, "_stop_answer_tags", ("</answer>",))), max_new=max_new_tokens)
            text2 = ""
            if stream:
                streamer2 = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
                gen_kwargs2 = dict(
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
        self.last_stats.update({
            "answer_tokens_max": max_new_tokens,
            "gate_activity_mean": gate_mean,
            "think_tokens_used": locals().get("used", None),
            "visible_cot": bool(self.cfg.visible_cot),
        })
        if self.cfg.visible_cot:
            return (text1 + text2).strip()
        return _extract_answer((text1 or "") + (text2 or ""), include_think=False)
    
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
        
