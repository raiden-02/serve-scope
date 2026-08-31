from __future__ import annotations

from fastapi.testclient import TestClient

from servescope.demo.app import app


def test_chat_and_mode_are_json_bodies():
    spec = app.openapi()
    assert "requestBody" in spec["paths"]["/api/chat"]["post"]
    assert "requestBody" in spec["paths"]["/api/mode"]["post"]


def test_chat_missing_prompt_is_body_validation():
    with TestClient(app) as client:
        res = client.post("/api/chat", json={})
    assert res.status_code == 422
    loc = res.json()["detail"][0]["loc"]
    assert loc[0] != "query"
    assert loc == ["body", "prompt"]
