from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from servescope.demo.app import app, create_app

ROOT = Path(__file__).resolve().parents[1]


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


def test_comparison_routes_exist():
    spec = app.openapi()
    assert "/api/comparison" in spec["paths"]
    assert "/api/comparison/start" in spec["paths"]
    assert "/api/comparison/cancel" not in spec["paths"]


def test_comparison_start_requires_connected_server():
    cfg = json.loads((ROOT / "configs" / "demo.json").read_text(encoding="utf-8"))
    cfg["base_url"] = "http://127.0.0.1:9"
    with TestClient(create_app(cfg)) as client:
        res = client.post("/api/comparison/start")
    assert res.status_code == 409
    assert "disconnected" in res.text
