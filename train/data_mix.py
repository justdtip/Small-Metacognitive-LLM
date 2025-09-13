from __future__ import annotations

import random
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type

from train.data_schema import ReasoningSample, ensure_reasoning_sample


# Adapter registry --------------------------------------------------------------
ADAPTERS: Dict[str, Any] = {}


def register_adapter(name: str, cls: Any) -> None:
    ADAPTERS[str(name)] = cls


def _try_import_and_register_defaults() -> None:
    try:
        from train.adapters.gsm8k_adapter import GSM8KAdapter as _G
        register_adapter("gsm8k", _G)
    except Exception:
        pass
    try:
        from train.adapters.math_adapter import MATHAdapter as _M
        register_adapter("math", _M)
    except Exception:
        pass
    try:
        from train.adapters.mbpp_adapter import MBPPAdapter as _B
        register_adapter("mbpp", _B)
    except Exception:
        pass
    try:
        from train.adapters.humaneval_adapter import HumanEvalAdapter as _H
        register_adapter("humaneval", _H)
    except Exception:
        pass
    try:
        from train.adapters.glaive_adapter import GlaiveAdapter as _GL
        register_adapter("glaive", _GL)
    except Exception:
        pass
    try:
        from train.adapters.logic_adapter import LogicAdapter as _L
        register_adapter("logic", _L)
    except Exception:
        pass


_try_import_and_register_defaults()


# Token length utilities --------------------------------------------------------
def _token_len(text: str, tokenizer: Any = None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            return int(len(ids)) if isinstance(ids, (list, tuple)) else int(getattr(ids, "shape", [0])[0])
        except Exception:
            pass
        # tokenizer might support call semantics
        try:
            enc = tokenizer(text, add_special_tokens=False)
            ids = getattr(enc, "input_ids", enc.get("input_ids"))
            if isinstance(ids, (list, tuple)):
                return int(len(ids))
        except Exception:
            pass
    # Fallback: whitespace token count
    return len([t for t in str(text).split() if t])


def _quantiles(values: List[int], qs: List[float]) -> List[int]:
    if not values:
        return [0 for _ in qs]
    xs = sorted(int(v) for v in values)
    out = []
    for q in qs:
        i = int(max(0, min(len(xs) - 1, round(q * (len(xs) - 1)))))
        out.append(xs[i])
    return out


def probe_token_lengths(samples: Iterable[ReasoningSample], tokenizer: Any = None, n: int = 1000) -> Dict[str, Any]:
    max_in = 0
    max_out = 0
    outs: List[int] = []
    for i, s in enumerate(samples):
        if i >= int(n):
            break
        li = _token_len(s.prompt, tokenizer)
        lo = _token_len(s.target or "", tokenizer)
        max_in = max(max_in, li)
        max_out = max(max_out, lo)
        outs.append(lo)
    p50, p90 = _quantiles(outs, [0.5, 0.9]) if outs else (0, 0)
    return {"max_in": int(max_in), "max_out": int(max_out), "p50_out": int(p50 if isinstance(p50, int) else p50[0]), "p90_out": int(p90 if isinstance(p90, int) else p90[0])}


# Mixture builder ---------------------------------------------------------------
def build_mixture(
    cfg: Dict[str, Any],
    *,
    sources: Optional[Dict[str, Iterable[Dict[str, Any]]]] = None,
    tokenizer: Any = None,
) -> Iterator[ReasoningSample]:
    """
    Build a mixed iterable of ReasoningSample from multiple adapters according to cfg.

    cfg example:
      data_mixture:
        datasets:
          - {name: gsm8k, adapter: gsm8k_adapter, weight: 1.0}
          - {name: math,  adapter: math_adapter,  weight: 0.7}
        temperature_sampling: 0.7
        probe_first_n: 10000
        target_depth_mix: {short: 0.35, mid: 0.45, long: 0.20}
    """
    dm = cfg.get("data_mixture") or {}
    ds_cfg = dm.get("datasets") or []
    temperature = float(dm.get("temperature_sampling", 1.0) or 1.0)
    probe_first_n = int(dm.get("probe_first_n", 1000) or 1000)

    # Build adapter instances
    adapters: List[Tuple[str, List[ReasoningSample], float]] = []
    total_weight = 0.0
    for d in ds_cfg:
        name = str(d.get("name") or "")
        weight = float(d.get("weight", 1.0) or 1.0)
        total_weight += max(0.0, weight)
        rows = list((sources or {}).get(name, []) if sources is not None else [])
        cls = ADAPTERS.get(name)
        if cls is None:
            # try to import on demand if adapter specifies a module
            adapter_mod = d.get("adapter")
            if isinstance(adapter_mod, str):
                try:
                    mod = __import__(f"train.adapters.{adapter_mod}", fromlist=["*"])
                    # pick first class in module that has __iter__
                    for attr in dir(mod):
                        C = getattr(mod, attr)
                        if hasattr(C, "__iter__"):
                            cls = C
                            break
                except Exception:
                    cls = None
        if cls is None:
            continue
        try:
            inst = cls(rows)
            samples = [ensure_reasoning_sample(s) for s in inst]
        except Exception:
            samples = []
        adapters.append((name, samples, weight))

    # Probe sizes and set meta fields
    flat = [s for _, lst, _ in adapters for s in lst]
    lengths = probe_token_lengths(iter(flat), tokenizer=tokenizer, n=probe_first_n)
    # Assign est_tokens_in/out and est_steps heuristics
    outs = []
    for s in flat:
        ti = _token_len(s.prompt, tokenizer)
        to = _token_len(s.target or "", tokenizer)
        s.meta["est_tokens_in"] = int(ti)
        s.meta["est_tokens_out"] = int(to)
        # crude heuristic for steps: number of lines or sentences in rationale
        rat = s.rationale or ""
        est_steps = max(1, rat.count("\n") + rat.count(".") // 2)
        s.meta["est_steps"] = int(est_steps)
        outs.append(int(to))
    # Bin by quantiles of output length
    q1, q2 = _quantiles(outs, [1/3, 2/3]) if outs else (0, 0)
    for s in flat:
        to = int(s.meta.get("est_tokens_out") or 0)
        if to <= q1:
            s.meta["length_bin"] = "short"
        elif to <= q2:
            s.meta["length_bin"] = "mid"
        else:
            s.meta["length_bin"] = "long"

    # Temperature-adjusted weights
    def _adj(w: float) -> float:
        if temperature <= 0:
            return 0.0
        return float(w) ** float(temperature)

    weights = [_adj(w) for _, __, w in adapters]
    ssum = sum(weights) or 1.0
    probs = [w / ssum for w in weights]

    # Round-robin with weighted choice
    iters = [iter(lst) for _, lst, _ in adapters]
    buffers: List[List[ReasoningSample]] = [list(lst) for _, lst, _ in adapters]
    idxs = list(range(len(buffers)))
    while any(buffers):
        # pick an index according to probs but skip empty buffers
        choices = [i for i in idxs if buffers[i]]
        if not choices:
            break
        if len(choices) == 1:
            i = choices[0]
        else:
            # renormalize over available choices
            ps = [probs[i] for i in choices]
            s = sum(ps) or 1.0
            ps = [p / s for p in ps]
            r = random.random()
            acc = 0.0
            i = choices[-1]
            for j, p in zip(choices, ps):
                acc += p
                if r <= acc:
                    i = j
                    break
        s = buffers[i].pop(0)
        yield s

