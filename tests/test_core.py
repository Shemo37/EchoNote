"""Unit tests for the pure-logic pieces: helpers, speaker merge, pipeline."""
import shutil

from server import db
from server.diarize import assign_speakers, speaker_color, SPEAKER_COLORS
from server.llm import (extract_action_items, transcript_for_prompt,
                        format_timestamp, CITATION_RE, parse_citation)


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(75) == "01:15"
    assert format_timestamp(3671) == "1:01:11"


def test_citation_parsing():
    matches = list(CITATION_RE.finditer("see [02:15] and [1:00:05] for details"))
    assert [parse_citation(m) for m in matches] == [135, 3605]


def test_extract_action_items():
    md = ("## Summary\n- a point\n\n## Action Items\n"
          "- [ ] Naomi: send report\n- [x] done thing\n- [ ] book the room\n")
    assert extract_action_items(md) == ["Naomi: send report", "book the room"]
    assert extract_action_items("") == []


def test_transcript_for_prompt_truncates():
    segments = [{"start_s": i * 2.0, "end_s": i * 2.0 + 2, "speaker": "Speaker 1",
                 "text": "word " * 50} for i in range(200)]
    text = transcript_for_prompt(segments, max_chars=1000)
    assert len(text) < 1100
    assert text.startswith("[00:00] Speaker 1:")
    assert "truncated" in text


def test_assign_speakers_by_overlap():
    segments = [
        {"start_s": 0.0, "end_s": 4.0, "speaker": None, "text": "a"},
        {"start_s": 5.0, "end_s": 11.0, "speaker": None, "text": "b"},
        {"start_s": 30.0, "end_s": 31.0, "speaker": None, "text": "c"},
    ]
    turns = [(0.0, 5.0, "Speaker 1"), (5.0, 12.0, "Speaker 2")]
    assign_speakers(segments, turns)
    assert segments[0]["speaker"] == "Speaker 1"
    assert segments[1]["speaker"] == "Speaker 2"
    assert segments[2]["speaker"] is None  # no overlapping turn


def test_speaker_colors_stable():
    assert speaker_color("Speaker 1") == SPEAKER_COLORS[0]
    assert speaker_color("Speaker 2") == SPEAKER_COLORS[1]
    assert speaker_color("Speaker 9") == SPEAKER_COLORS[0]  # wraps


def test_pipeline_end_to_end(make_wav, tmp_path):
    from server.config import Config
    from server.pipeline import Pipeline

    db.connect()
    wav = make_wav(seconds=9.0)
    stored = str(tmp_path / "orig.wav")
    shutil.copyfile(wav, stored)

    rec_id = db.create_recording("Weekly sync", stored)
    config = Config(path=str(tmp_path / "cfg.json"))
    pipe = Pipeline(config)
    pipe._process(rec_id)  # synchronous: no worker thread in unit tests

    rec = db.get_recording(rec_id)
    assert rec["status"] == "ready"
    assert rec["duration_s"] > 8
    assert rec["language"] == "en"
    assert rec["gist"]  # stub LLM produced a one-liner

    segments = db.get_segments(rec_id)
    assert len(segments) >= 2
    assert all(s["speaker"] in ("Speaker 1", "Speaker 2") for s in segments)


def test_pipeline_error_state(tmp_path):
    from server.config import Config
    from server.pipeline import Pipeline

    db.connect()
    rec_id = db.create_recording("Broken", str(tmp_path / "missing.mp3"))
    config = Config(path=str(tmp_path / "cfg2.json"))
    pipe = Pipeline(config)
    try:
        pipe._process(rec_id)
    except Exception:
        # worker catches this in _run; mimic it
        db.update_recording(rec_id, status="error", error="conversion failed")
    rec = db.get_recording(rec_id)
    assert rec["status"] == "error"
