from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
import torch
# Import the ContextVar directly from tina.side_adapters to avoid shim filtering
from tina.side_adapters import _ctx_think_mask

_ctx_plan_mask: ContextVar = ContextVar("_ctx_plan_mask", default=None)
_ctx_exec_mask: ContextVar = ContextVar("_ctx_exec_mask", default=None)
_ctx_eval_mask: ContextVar = ContextVar("_ctx_eval_mask", default=None)

@contextmanager
def think_mask_context(mask: torch.Tensor):
    """
    Provide a per-token think mask [B,T] to side adapters via contextvar.
    Values are cast to float and broadcast over hidden dim inside the adapter.
    """
    token_mask = mask.to(torch.float32)
    tok = _ctx_think_mask.set(token_mask)
    try:
        yield
    finally:
        _ctx_think_mask.reset(tok)


@contextmanager
def decomp_mask_context(plan_mask=None, exec_mask=None, eval_mask=None):
    """Provide optional decomposition sub-masks via context variables.
    Masks should be [B,T] tensors or None. These are observational only.
    """
    tok_p = _ctx_plan_mask.set(plan_mask)
    tok_e = _ctx_exec_mask.set(exec_mask)
    tok_v = _ctx_eval_mask.set(eval_mask)
    try:
        yield
    finally:
        _ctx_plan_mask.reset(tok_p)
        _ctx_exec_mask.reset(tok_e)
        _ctx_eval_mask.reset(tok_v)


def reset_coverage(scaffold) -> None:
    """Reset adapter coverage metrics before a new forward/generation step."""
    try:
        adapters = getattr(scaffold, 'adapters', [])
        import torch as _t
        for m in adapters:
            try:
                setattr(m, '_last_gate_coverage', _t.tensor(0.0))
                setattr(m, '_last_gate_activity', _t.tensor(0.0))
            except Exception:
                try:
                    setattr(m, '_last_gate_coverage', 0.0)
                    setattr(m, '_last_gate_activity', 0.0)
                except Exception:
                    pass
    except Exception:
        pass
