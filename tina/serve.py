# tina/serve.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Sequence
import threading
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from .tokenizer_utils import ensure_reasoning_tokens

@dataclass
class EngineConfig:
    # reasoning behavior
    visible_cot: bool = False
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

    def _estimate_budget(self, input_ids) -> int:
        # Dry forward to populate taps, then ask metacog for budget
        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           use_cache=False, output_hidden_states=True, return_dict=True)
        out = self.metacog(B_max=self.cfg.budget_cap)
        # record metacog head outputs for observability
        try:
            plan_idx = int(torch.argmax(out["plan_logits"], dim=-1)[0].item())
            conf_val = float(out["confidence"][0].item())
            raw_budget = float(out["budget"][0].item())
        except Exception:
            plan_idx, conf_val, raw_budget = None, None, None
        # pick budget
        b = int(raw_budget) if self.cfg.use_dynamic_budget and raw_budget is not None else self.cfg.budget_cap
        b = max(self.cfg.min_think_tokens, min(b, self.cfg.budget_cap))
        # clear taps for next pass
        self.metacog.clear_cache()
        # save stats
        self.last_stats.update({
            "plan": plan_idx,
            "confidence": conf_val,
            "think_budget": b,
        })
        return b

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
            stop_think = StopOnTags(self.tok, ("</think>",), max_new=think_budget)
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
                    stopping_criteria=StoppingCriteriaList([stop_think]),
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
                        stopping_criteria=StoppingCriteriaList([stop_think]),
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
            
            # If we didn't get </think>, forcibly close it
            if "</think>" not in text1:
                text1 = text1 + "</think>\n"

            # Adapters auto-disabled when leaving think() context

            # Compose for ANSWER: append the think text (even if hidden, model needs context)
            ans_tok = self.tok("<answer>", add_special_tokens=False, return_tensors="pt").input_ids.to(self.model.device)
            full_prompt_ids = torch.cat([out1, ans_tok], dim=1)
            full_attn = torch.ones_like(full_prompt_ids)

            # Step 2: ANSWER — stop at </answer> or max_new_tokens
            stop_answer = StopOnTags(self.tok, ("</answer>",), max_new=max_new_tokens)
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
        
