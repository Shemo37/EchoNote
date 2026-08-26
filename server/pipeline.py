"""
Background processing pipeline: one worker thread, one job at a time (the
GPU is a shared resource). No realtime anywhere - a job starts only after a
complete file has been uploaded or a browser recording has been stopped.

States: queued -> converting -> transcribing -> diarizing -> summarizing -> ready
                                             (any step) -> error
"""
import logging
import os
import queue
import shutil
import subprocess
import threading
import traceback
import wave

from . import db
from .asr import create_transcriber
from .diarize import create_diarizer, assign_speakers
from .llm import (create_llm, transcript_for_prompt, extract_action_items,
                  LLMUnavailable)

logger = logging.getLogger(__name__)


def ffmpeg_path():
    return shutil.which("ffmpeg")


def convert_to_wav(src, dst):
    """Convert any audio container to 16 kHz mono WAV.

    Uses ffmpeg when present; without it, 16-bit PCM WAV files are accepted
    as-is and anything else raises with a clear message.
    """
    exe = ffmpeg_path()
    if exe:
        subprocess.run(
            [exe, "-y", "-i", src, "-ac", "1", "-ar", "16000",
             "-sample_fmt", "s16", dst],
            check=True, capture_output=True)
        return dst
    if src.lower().endswith(".wav"):
        shutil.copyfile(src, dst)
        return dst
    raise RuntimeError(
        "ffmpeg is required to import this format. Install ffmpeg "
        "(https://ffmpeg.org) or upload a 16 kHz WAV file instead.")


def wav_duration(path):
    try:
        with wave.open(path, "rb") as w:
            rate = w.getframerate() or 16000
            return w.getnframes() / float(rate)
    except Exception:
        return 0.0


GIST_PROMPT = ("Describe this conversation in one short sentence (max 15 words). "
               "Reply with the sentence only.\n\nTranscript:\n{transcript}")


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.jobs = queue.Queue()
        self.progress = {}  # rec_id -> {"stage": str, "seconds_done": float}
        self._transcriber = None
        self._diarizer = None
        self._thread = None
        self._stop = threading.Event()

    # -- lifecycle --

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="echonote-pipeline")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def enqueue(self, rec_id):
        self.progress[rec_id] = {"stage": "queued", "seconds_done": 0.0}
        self.jobs.put(rec_id)

    # -- lazily loaded heavy components (shared across jobs) --

    def transcriber(self):
        if self._transcriber is None:
            t = create_transcriber(self.config)
            t.load()
            self._transcriber = t
        return self._transcriber

    def diarizer(self):
        if self._diarizer is None:
            d = create_diarizer(self.config, device=self.transcriber().device)
            if not d.available():
                return None
            try:
                d.load()
            except Exception as e:
                logger.warning("Diarization unavailable: %s", e)
                print(f"Speaker diarization unavailable ({e}); continuing without labels.")
                return None
            self._diarizer = d
        return self._diarizer

    # -- worker --

    def _run(self):
        logger.info("Pipeline worker started")
        while not self._stop.is_set():
            try:
                rec_id = self.jobs.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._process(rec_id)
            except Exception as e:
                logger.error("Job %s failed: %s", rec_id, e)
                traceback.print_exc()
                db.update_recording(rec_id, status="error", error=str(e))
            finally:
                self.progress.pop(rec_id, None)

    def _set_stage(self, rec_id, stage):
        self.progress.setdefault(rec_id, {})["stage"] = stage
        db.update_recording(rec_id, status=stage)

    def _process(self, rec_id):
        rec = db.get_recording(rec_id)
        if rec is None:
            return

        # 1. convert
        self._set_stage(rec_id, "converting")
        wav_path = os.path.splitext(rec["original_path"])[0] + "_16k.wav"
        convert_to_wav(rec["original_path"], wav_path)
        duration = wav_duration(wav_path)
        db.update_recording(rec_id, wav_path=wav_path, duration_s=duration)
        self.progress[rec_id]["duration"] = duration

        # 2. transcribe
        self._set_stage(rec_id, "transcribing")

        def on_progress(seconds_done):
            self.progress.setdefault(rec_id, {})["seconds_done"] = seconds_done

        segments, info = self.transcriber().transcribe(wav_path, progress=on_progress)
        db.update_recording(rec_id, language=info.get("language"),
                            duration_s=info.get("duration") or duration)

        # 3. diarize (optional)
        diarizer = self.diarizer()
        if diarizer is not None and segments:
            self._set_stage(rec_id, "diarizing")
            try:
                turns = diarizer.diarize(wav_path)
                assign_speakers(segments, turns)
            except Exception as e:
                logger.warning("Diarization failed for %s: %s", rec_id, e)

        db.add_segments(rec_id, segments)

        # 4. auto gist (Plaud-style "auto generation") - best effort
        if segments:
            self._set_stage(rec_id, "summarizing")
            try:
                llm = create_llm(self.config)
                transcript = transcript_for_prompt(segments, max_chars=8000)
                gist = llm.chat([{"role": "user",
                                  "content": GIST_PROMPT.format(transcript=transcript)}])
                db.update_recording(rec_id, gist=gist.strip()[:200])
            except LLMUnavailable:
                pass  # transcribe-only mode
            except Exception as e:
                logger.warning("Gist generation failed: %s", e)

        db.update_recording(rec_id, status="ready", error=None)
        logger.info("Recording %s ready (%d segments)", rec_id, len(segments))


def store_summary(config, rec_id, template, content):
    """Persist a summary and harvest its '- [ ]' action items."""
    db.save_summary(rec_id, template, content)
    items = extract_action_items(content)
    if items:
        db.add_action_items(rec_id, items)
    return items
