"""API round-trip tests with the stub ASR/diarizer/LLM."""
import time

import pytest
from fastapi.testclient import TestClient

from server import db
from server.app import create_app
from server.config import Config


@pytest.fixture(scope="module")
def client():
    app = create_app(Config())
    with TestClient(app) as c:
        yield c
    db.close()


def wait_ready(client, rec_id, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        rec = client.get(f"/api/recordings/{rec_id}").json()
        if rec["status"] in ("ready", "error"):
            return rec
        time.sleep(0.2)
    raise AssertionError("recording never finished processing")


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok"
    assert h["ollama"] is True  # stub LLM


def test_upload_process_and_fetch(client, make_wav):
    wav = make_wav(seconds=12.0, name="meeting.wav")
    with open(wav, "rb") as f:
        r = client.post("/api/recordings",
                        files={"file": ("team_meeting.wav", f, "audio/wav")})
    assert r.status_code == 200
    rec_id = r.json()["id"]

    rec = wait_ready(client, rec_id)
    assert rec["status"] == "ready"
    assert rec["title"] == "team meeting"
    assert len(rec["segments"]) >= 2
    assert rec["segments"][0]["ts"] == "00:00"
    assert rec["segments"][0]["color"]  # speaker chip color present

    audio = client.get(f"/api/recordings/{rec_id}/audio")
    assert audio.status_code == 200

    listed = client.get("/api/recordings").json()
    assert any(x["id"] == rec_id for x in listed)
    # full-text search over transcript (stub text mentions 'migration')
    hits = client.get("/api/recordings", params={"q": "migration"}).json()
    assert any(x["id"] == rec_id for x in hits)


def test_summarize_and_action_items(client, make_wav):
    wav = make_wav(seconds=6.0, name="s.wav")
    with open(wav, "rb") as f:
        rec_id = client.post("/api/recordings",
                             files={"file": ("standup.wav", f, "audio/wav")}).json()["id"]
    wait_ready(client, rec_id)

    assert "key-points" in client.get("/api/templates").json()

    resp = client.post(f"/api/recordings/{rec_id}/summarize", params={"template": "key-points"})
    assert resp.status_code == 200
    body = resp.text
    assert '"done": true' in body
    assert "Action Items" in body

    items = client.get("/api/action_items").json()
    assert any("report" in i["text"] for i in items)

    item_id = items[0]["id"]
    client.post(f"/api/action_items/{item_id}/toggle")
    toggled = [i for i in client.get("/api/action_items").json() if i["id"] == item_id][0]
    assert toggled["done"] == 1


def test_ask_with_citation(client, make_wav):
    wav = make_wav(seconds=6.0, name="a.wav")
    with open(wav, "rb") as f:
        rec_id = client.post("/api/recordings",
                             files={"file": ("q.wav", f, "audio/wav")}).json()["id"]
    wait_ready(client, rec_id)

    resp = client.post(f"/api/recordings/{rec_id}/ask", json={"question": "When is the report due?"})
    assert resp.status_code == 200
    assert "Friday" in resp.text and "[00:06]" in resp.text

    rec = client.get(f"/api/recordings/{rec_id}").json()
    roles = [c["role"] for c in rec["chats"]]
    assert roles[-2:] == ["user", "assistant"]


def test_dashboard(client):
    d = client.get("/api/dashboard").json()
    assert d["stats"]["recordings"] >= 1
    assert len(d["stats"]["activity_14d"]) == 14
    assert sum(d["stats"]["activity_14d"]) >= 1
    assert d["recent"]


def test_settings_roundtrip_and_token_redaction(client):
    s = client.put("/api/settings", json={"hotwords": "EchoNote, Fubuki",
                                          "hf_token": "hf_secret123"}).json()
    assert s["hotwords"] == "EchoNote, Fubuki"
    assert s["hf_token"] == "" and s["hf_token_set"] is True  # never echoed back

    # empty token on next save must NOT clear the stored one
    s2 = client.put("/api/settings", json={"hf_token": ""}).json()
    assert s2["hf_token_set"] is True

    bad = client.put("/api/settings", json={"compute_type": "banana"}).json()
    assert bad["compute_type"] == "auto"  # validator reset


def test_delete_recording(client, make_wav):
    wav = make_wav(seconds=6.0, name="d.wav")
    with open(wav, "rb") as f:
        rec_id = client.post("/api/recordings",
                             files={"file": ("temp.wav", f, "audio/wav")}).json()["id"]
    wait_ready(client, rec_id)
    assert client.delete(f"/api/recordings/{rec_id}").json()["ok"] is True
    assert client.get(f"/api/recordings/{rec_id}").status_code == 404
