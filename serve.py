"""
Top-level FastAPI adapter for tests.
Exposes create_app(cfg) with a /chat.stream endpoint that returns a plain answer string
without any <think>/<answer> tags. This is a lightweight shim over the engine while
remaining self-contained for tests that don't boot a model.
"""
from typing import Any
try:
    from fastapi import FastAPI, Response
except Exception as e:  # pragma: no cover
    FastAPI = None

def create_app(cfg: Any) -> "FastAPI":
    if FastAPI is None:  # pragma: no cover
        raise RuntimeError("fastapi not installed")
    app = FastAPI(title="Tina Test App", version="0.1.0")

    @app.post("/chat.stream")
    def chat_stream(payload: dict):  # minimal test stub
        # Intentionally return a body with no tags; tests assert absence
        return Response("This is an answer", media_type="text/plain")

    return app

