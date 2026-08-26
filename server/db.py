"""
SQLite storage for EchoNote.

sqlite3 with check_same_thread=False + a module lock: traffic is one browser
and one pipeline worker, so a single serialized connection is plenty.
"""
import json
import os
import sqlite3
import threading
import time

from .config import DB_PATH

_lock = threading.Lock()
_conn = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    duration_s REAL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'queued',  -- queued|converting|transcribing|diarizing|summarizing|ready|error
    error TEXT,
    language TEXT,
    original_path TEXT,
    wav_path TEXT,
    gist TEXT
);
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    speaker TEXT,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_segments_rec ON segments(recording_id);
CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    template TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER REFERENCES recordings(id) ON DELETE CASCADE,  -- NULL = global ask
    role TEXT NOT NULL,       -- user|assistant
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS action_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER REFERENCES recordings(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
"""


def connect(path=None):
    global _conn
    with _lock:
        if _conn is None:
            db_path = path or DB_PATH
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            _conn = sqlite3.connect(db_path, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA foreign_keys = ON")
            _conn.executescript(SCHEMA)
            _conn.commit()
    return _conn


def close():
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def _execute(sql, params=()):
    conn = connect()
    with _lock:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur


def _rows(sql, params=()):
    conn = connect()
    with _lock:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _row(sql, params=()):
    conn = connect()
    with _lock:
        r = conn.execute(sql, params).fetchone()
        return dict(r) if r else None


# ---- recordings ----

def create_recording(title, original_path):
    cur = _execute(
        "INSERT INTO recordings (title, created_at, original_path) VALUES (?, ?, ?)",
        (title, time.time(), original_path))
    return cur.lastrowid


def update_recording(rec_id, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    _execute(f"UPDATE recordings SET {cols} WHERE id = ?", (*fields.values(), rec_id))


def get_recording(rec_id):
    return _row("SELECT * FROM recordings WHERE id = ?", (rec_id,))


def list_recordings(query=None, limit=200):
    if query:
        like = f"%{query}%"
        return _rows(
            """SELECT DISTINCT r.* FROM recordings r
               LEFT JOIN segments s ON s.recording_id = r.id
               WHERE r.title LIKE ? OR s.text LIKE ?
               ORDER BY r.created_at DESC LIMIT ?""",
            (like, like, limit))
    return _rows("SELECT * FROM recordings ORDER BY created_at DESC LIMIT ?", (limit,))


def delete_recording(rec_id):
    rec = get_recording(rec_id)
    _execute("DELETE FROM recordings WHERE id = ?", (rec_id,))
    if rec:
        for key in ("original_path", "wav_path"):
            path = rec.get(key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ---- segments ----

def add_segments(rec_id, segments):
    """segments: iterable of dicts {start_s, end_s, speaker, text}"""
    conn = connect()
    with _lock:
        conn.executemany(
            "INSERT INTO segments (recording_id, start_s, end_s, speaker, text) VALUES (?, ?, ?, ?, ?)",
            [(rec_id, s["start_s"], s["end_s"], s.get("speaker"), s["text"]) for s in segments])
        conn.commit()


def get_segments(rec_id):
    return _rows("SELECT * FROM segments WHERE recording_id = ? ORDER BY start_s", (rec_id,))


def set_segment_speakers(rec_id, speaker_by_segment_id):
    conn = connect()
    with _lock:
        conn.executemany(
            "UPDATE segments SET speaker = ? WHERE id = ? AND recording_id = ?",
            [(spk, seg_id, rec_id) for seg_id, spk in speaker_by_segment_id.items()])
        conn.commit()


# ---- summaries ----

def save_summary(rec_id, template, content):
    _execute(
        "INSERT INTO summaries (recording_id, template, content, created_at) VALUES (?, ?, ?, ?)",
        (rec_id, template, content, time.time()))


def get_summaries(rec_id):
    return _rows(
        "SELECT * FROM summaries WHERE recording_id = ? ORDER BY created_at DESC", (rec_id,))


def latest_summary(rec_id, template=None):
    if template:
        return _row(
            "SELECT * FROM summaries WHERE recording_id = ? AND template = ? ORDER BY created_at DESC LIMIT 1",
            (rec_id, template))
    return _row(
        "SELECT * FROM summaries WHERE recording_id = ? ORDER BY created_at DESC LIMIT 1", (rec_id,))


# ---- chats ----

def add_chat(rec_id, role, content):
    _execute("INSERT INTO chats (recording_id, role, content, created_at) VALUES (?, ?, ?, ?)",
             (rec_id, role, content, time.time()))


def get_chats(rec_id):
    if rec_id is None:
        return _rows("SELECT * FROM chats WHERE recording_id IS NULL ORDER BY created_at")
    return _rows("SELECT * FROM chats WHERE recording_id = ? ORDER BY created_at", (rec_id,))


# ---- action items ----

def add_action_items(rec_id, texts):
    conn = connect()
    now = time.time()
    with _lock:
        conn.executemany(
            "INSERT INTO action_items (recording_id, text, created_at) VALUES (?, ?, ?)",
            [(rec_id, t, now) for t in texts])
        conn.commit()


def list_action_items(include_done=True, limit=100):
    where = "" if include_done else "WHERE a.done = 0"
    return _rows(
        f"""SELECT a.*, r.title AS recording_title FROM action_items a
            LEFT JOIN recordings r ON r.id = a.recording_id
            {where} ORDER BY a.done ASC, a.created_at DESC LIMIT ?""", (limit,))


def toggle_action_item(item_id):
    _execute("UPDATE action_items SET done = 1 - done WHERE id = ?", (item_id,))


# ---- dashboard ----

def dashboard_stats():
    totals = _row(
        """SELECT COUNT(*) AS recordings,
                  COALESCE(SUM(duration_s), 0) AS total_seconds
           FROM recordings WHERE status = 'ready'""") or {}
    week = _row(
        "SELECT COUNT(*) AS n FROM recordings WHERE created_at > ?",
        (time.time() - 7 * 86400,)) or {}
    day_rows = _rows(
        """SELECT CAST((created_at / 86400) AS INTEGER) AS day, COUNT(*) AS n
           FROM recordings WHERE created_at > ?
           GROUP BY day""", (time.time() - 14 * 86400,))
    today = int(time.time() // 86400)
    activity = [0] * 14
    for r in day_rows:
        idx = 13 - (today - r["day"])
        if 0 <= idx < 14:
            activity[idx] = r["n"]
    return {
        "recordings": totals.get("recordings", 0),
        "total_seconds": totals.get("total_seconds", 0.0),
        "this_week": week.get("n", 0),
        "activity_14d": activity,
    }
