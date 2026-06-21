"""Smoke tests for the CubeGB Studio backend (app/server.py).

Skipped automatically when the web extra (fastapi/httpx) is not installed, so the
core test suite stays runnable without it:

    pip install -r requirements.txt -r requirements-app.txt httpx
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by fastapi.testclient.TestClient

from fastapi.testclient import TestClient  # noqa: E402

from app.server import app  # noqa: E402

client = TestClient(app)
SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "CubeGB Studio" in r.text


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["can_view"] and body["can_bake"]
    assert "recognition_available" in body


def test_bake_glb_returns_binary():
    doc = json.loads((SAMPLES / "chair.cgb").read_text())
    r = client.post("/api/bake", json={"doc": doc, "format": "glb", "filename": "chair"})
    assert r.status_code == 200
    assert r.content[:4] == b"glTF"  # valid GLB magic
    assert len(r.content) > 0


def test_bake_rejects_bad_format():
    doc = json.loads((SAMPLES / "table.cgb").read_text())
    r = client.post("/api/bake", json={"doc": doc, "format": "stl"})
    assert r.status_code == 400


def test_bake_rejects_invalid_doc():
    r = client.post("/api/bake", json={"doc": {"format": "cgb"}, "format": "glb"})
    assert r.status_code == 400


def test_generate_without_checkpoint_is_clear_error():
    r = client.post("/api/generate", files={"image": ("x.png", b"data", "image/png")},
                    data={"device": "cpu"})
    assert r.status_code == 400
    assert "checkpoint" in r.json()["detail"].lower()
