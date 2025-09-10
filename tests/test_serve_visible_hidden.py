import pytest
from fastapi.testclient import TestClient
from serve import create_app

class DummyCfg:
    visible_cot = False

def test_hidden_cot_removal():
    app = create_app(DummyCfg())
    client = TestClient(app)
    r = client.post("/chat.stream", json={"messages":[{"role":"user","content":"Hi"}]})
    assert r.status_code == 200
    body = r.text
    assert "<think>" not in body and "</think>" not in body
    assert "<answer>" not in body and "</answer>" not in body

