"""
EchoNote FastAPI application: REST API + static SPA.
"""
import json
import logging
import os
import re
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .config import Config, ensure_dirs, AUDIO_DIR
from .diarize import speaker_color
from .llm import (create_llm, transcript_for_prompt, LLMUnavailable, INSTALL_HINT,
                  format_timestamp)
from .pipeline import Pipeline, store_summary, ffmpeg_path

logger = logging.getLogger(__name__)

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")

ASK_SYSTEM = (
    "You answer questions about recorded conversations using ONLY the supplied "
    "transcripts. Cite the supporting timestamp(s) in square brackets like [mm:ss] "
    "after each claim so the user can jump to the audio. If the transcripts don't "
    "contain the answer, say so plainly.")


def create_app(config: Config | None = None) -> FastAPI:
    ensure_dirs()
    config = config or Config()
    db.connect()

    app = FastAPI(title="EchoNote", docs_url=None, redoc_url=None)
    app.state.config = config
    app.state.pipeline = Pipeline(config)
    app.state.pipeline.start()

    # ---------- health / settings ----------

    @app.get("/api/health")
    def health():
        llm = create_llm(config)
        return {
            "status": "ok",
            "ffmpeg": bool(ffmpeg_path()),
            "ollama": llm.available(),
            "ollama_models": llm.models(),
            "diarization_configured": bool(config.hf_token or os.environ.get("HF_TOKEN")),
        }

    @app.get("/api/settings")
    def get_settings():
        return config.to_dict()

    @app.put("/api/settings")
    async def put_settings(changes: dict):
        # empty hf_token string means "leave unchanged" (the UI never sees it)
        if changes.get("hf_token") == "":
            changes.pop("hf_token")
        config.update(changes)
        return config.to_dict()

    # ---------- recordings ----------

    SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

    @app.post("/api/recordings")
    async def upload_recording(file: UploadFile = File(...), title: str = Form(None)):
        name = SAFE_NAME.sub("_", os.path.basename(file.filename or "recording"))
        if not title:
            title = os.path.splitext(name)[0].replace("_", " ") or "Recording"
        dest = os.path.join(AUDIO_DIR, f"{int(time.time() * 1000)}_{name}")
        with open(dest, "wb") as out:
            while chunk := await file.read(1 << 20):
                out.write(chunk)
        rec_id = db.create_recording(title, dest)
        app.state.pipeline.enqueue(rec_id)
        return {"id": rec_id, "status": "queued"}

    @app.get("/api/recordings")
    def list_recordings(q: str = None):
        recs = db.list_recordings(query=q)
        progress = app.state.pipeline.progress
        for r in recs:
            r["progress"] = progress.get(r["id"])
        return recs

    @app.get("/api/recordings/{rec_id}")
    def get_recording(rec_id: int):
        rec = db.get_recording(rec_id)
        if not rec:
            raise HTTPException(404, "recording not found")
        segments = db.get_segments(rec_id)
        for s in segments:
            s["color"] = speaker_color(s["speaker"]) if s["speaker"] else None
            s["ts"] = format_timestamp(s["start_s"])
        rec["segments"] = segments
        rec["summaries"] = db.get_summaries(rec_id)
        rec["chats"] = db.get_chats(rec_id)
        rec["progress"] = app.state.pipeline.progress.get(rec_id)
        return rec

    @app.delete("/api/recordings/{rec_id}")
    def delete_recording(rec_id: int):
        db.delete_recording(rec_id)
        return {"ok": True}

    @app.get("/api/recordings/{rec_id}/audio")
    def get_audio(rec_id: int):
        rec = db.get_recording(rec_id)
        if not rec:
            raise HTTPException(404, "recording not found")
        path = rec.get("wav_path") or rec.get("original_path")
        if not path or not os.path.exists(path):
            raise HTTPException(404, "audio file missing")
        media = "audio/wav" if path.endswith(".wav") else "application/octet-stream"
        return FileResponse(path, media_type=media)

    # ---------- summaries ----------

    @app.get("/api/templates")
    def list_templates():
        names = sorted(os.path.splitext(f)[0] for f in os.listdir(TEMPLATES_DIR)
                       if f.endswith(".md"))
        return names

    @app.post("/api/recordings/{rec_id}/summarize")
    def summarize(rec_id: int, template: str = "key-points"):
        rec = db.get_recording(rec_id)
        if not rec:
            raise HTTPException(404, "recording not found")
        path = os.path.join(TEMPLATES_DIR, f"{SAFE_NAME.sub('', template)}.md")
        if not os.path.exists(path):
            raise HTTPException(404, f"unknown template '{template}'")
        segments = db.get_segments(rec_id)
        if not segments:
            raise HTTPException(409, "recording has no transcript yet")
        with open(path, encoding="utf-8") as f:
            prompt = f.read().format(transcript=transcript_for_prompt(segments))

        llm = create_llm(config)

        def stream():
            parts = []
            try:
                for chunk in llm.chat_stream([{"role": "user", "content": prompt}]):
                    parts.append(chunk)
                    yield _sse({"delta": chunk})
                content = "".join(parts)
                items = store_summary(config, rec_id, template, content)
                yield _sse({"done": True, "action_items": items})
            except LLMUnavailable as e:
                yield _sse({"error": str(e)})

        return StreamingResponse(stream(), media_type="text/event-stream")

    # ---------- ask (per-recording and global) ----------

    def _ask_stream(rec_id, question, transcripts_block):
        llm = create_llm(config)
        history = db.get_chats(rec_id)[-8:]
        messages = [{"role": "system", "content": ASK_SYSTEM},
                    {"role": "user", "content": f"Transcripts:\n{transcripts_block}"}]
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": question})

        def stream():
            parts = []
            try:
                for chunk in llm.chat_stream(messages):
                    parts.append(chunk)
                    yield _sse({"delta": chunk})
                db.add_chat(rec_id, "user", question)
                db.add_chat(rec_id, "assistant", "".join(parts))
                yield _sse({"done": True})
            except LLMUnavailable as e:
                yield _sse({"error": str(e)})

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/recordings/{rec_id}/ask")
    async def ask_recording(rec_id: int, body: dict):
        question = (body.get("question") or "").strip()
        if not question:
            raise HTTPException(400, "question required")
        segments = db.get_segments(rec_id)
        if not segments:
            raise HTTPException(409, "recording has no transcript yet")
        return _ask_stream(rec_id, question, transcript_for_prompt(segments))

    @app.post("/api/ask")
    async def ask_global(body: dict):
        question = (body.get("question") or "").strip()
        if not question:
            raise HTTPException(400, "question required")
        blocks = []
        for rec in db.list_recordings(limit=5):
            segs = db.get_segments(rec["id"])
            if segs:
                blocks.append(f"### {rec['title']}\n" + transcript_for_prompt(segs, max_chars=6000))
        if not blocks:
            return JSONResponse({"error": "No transcribed recordings yet."}, status_code=409)
        return _ask_stream(None, question, "\n\n".join(blocks))

    # ---------- action items / dashboard ----------

    @app.get("/api/action_items")
    def action_items():
        return db.list_action_items()

    @app.post("/api/action_items/{item_id}/toggle")
    def toggle_item(item_id: int):
        db.toggle_action_item(item_id)
        return {"ok": True}

    @app.get("/api/dashboard")
    def dashboard():
        stats = db.dashboard_stats()
        recent = db.list_recordings(limit=6)
        for r in recent:
            r["progress"] = app.state.pipeline.progress.get(r["id"])
        return {
            "stats": stats,
            "recent": recent,
            "action_items": db.list_action_items(limit=30),
        }

    # ---------- static SPA (mounted last so /api wins) ----------

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"
