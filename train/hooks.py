from __future__ import annotations
from contextlib import contextmanager
import torch
# Import the ContextVar directly from tina.side_adapters to avoid shim filtering
from tina.side_adapters import _ctx_think_mask

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
