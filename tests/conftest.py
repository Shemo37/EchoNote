"""
Test setup: stub the heavy components (ASR, diarization, LLM) and point the
data directory at a temp location BEFORE any server module is imported.
"""
import os
import sys
import tempfile

os.environ.setdefault("ECHONOTE_STUB_ASR", "1")
os.environ.setdefault("ECHONOTE_STUB_DIARIZE", "1")
os.environ.setdefault("ECHONOTE_STUB_LLM", "1")
os.environ.setdefault("ECHONOTE_DATA", tempfile.mkdtemp(prefix="echonote_test_"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402


@pytest.fixture()
def make_wav(tmp_path):
    """Write a small 16 kHz mono WAV of the given duration."""
    import math
    import struct
    import wave

    def _make(seconds=6.0, name="clip.wav"):
        path = tmp_path / name
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            n = int(16000 * seconds)
            frames = b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * 220 * i / 16000)))
                for i in range(n))
            w.writeframes(frames)
        return str(path)

    return _make
