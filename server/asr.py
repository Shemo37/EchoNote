"""
Batch speech-to-text via faster-whisper (CTranslate2).

Ported from HoloLiveTL's AsrBackend: cheap __init__ + heavy load(), option
pruning against the installed faster-whisper signature, hotwords (custom
vocabulary). Generalized for batch files: configurable multilingual model,
language auto-detect, built-in VAD to skip silence in long recordings.

Set ECHONOTE_STUB_ASR=1 to use a deterministic fake transcriber (tests/CI).
"""
import inspect
import logging
import os

logger = logging.getLogger(__name__)

# Short names accepted in settings; anything else is used as a Hub repo id.
MODEL_ALIASES = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def _filter_kwargs(func, kwargs):
    """Drop kwargs the installed faster-whisper version doesn't accept."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    kept = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        logger.warning("faster-whisper ignores unsupported options: %s", ", ".join(dropped))
    return kept


class Transcriber:
    """Whole-file transcription. Construct cheap, load() heavy."""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_id = MODEL_ALIASES.get(config.asr_model, config.asr_model)

        device = config.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        compute_type = config.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        self.compute_type = compute_type

    def load(self):
        from faster_whisper import WhisperModel

        from .config import DATA_DIR
        self.model = WhisperModel(
            self.model_id,
            device=self.device,
            compute_type=self.compute_type,
            download_root=os.path.join(DATA_DIR, "models"),
        )
        logger.info("ASR loaded: %s (%s/%s)", self.model_id, self.device, self.compute_type)

    def transcribe(self, wav_path, progress=None):
        """Transcribe a 16 kHz mono WAV file.

        Returns (segments, info_dict): segments are dicts
        {start_s, end_s, text, speaker: None}; info has language & duration.
        `progress(seconds_done)` is called as decoding advances.
        """
        if self.model is None:
            raise RuntimeError("Transcriber not loaded - call load() first")

        language = None if self.config.language == "auto" else self.config.language
        options = {
            "language": language,
            "task": "transcribe",
            "beam_size": 5,
            # Batch files are long-form: keep context for coherence, and let
            # faster-whisper's VAD skip silence (big win on meeting audio).
            "condition_on_previous_text": True,
            "vad_filter": True,
            "temperature": [0.0, 0.2, 0.4],
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "suppress_tokens": [-1],
        }
        hotwords = (self.config.hotwords or "").strip()
        if hotwords:
            options["hotwords"] = hotwords
        options = _filter_kwargs(self.model.transcribe, options)

        seg_iter, info = self.model.transcribe(wav_path, **options)

        segments = []
        for seg in seg_iter:  # lazy generator: decoding happens here
            text = (seg.text or "").strip()
            if not text:
                continue
            segments.append({
                "start_s": float(seg.start or 0.0),
                "end_s": float(seg.end or 0.0),
                "speaker": None,
                "text": text,
            })
            if progress is not None:
                progress(segments[-1]["end_s"])

        return segments, {
            "language": getattr(info, "language", None),
            "duration": float(getattr(info, "duration", 0.0) or 0.0),
        }

    def close(self):
        self.model = None


class StubTranscriber:
    """Deterministic fake for tests and container E2E (no models/GPU)."""

    def __init__(self, config):
        self.config = config
        self.device = "cpu"
        self.compute_type = "stub"

    def load(self):
        pass

    def transcribe(self, wav_path, progress=None):
        import wave
        try:
            with wave.open(wav_path, "rb") as w:
                duration = w.getnframes() / float(w.getframerate() or 16000)
        except Exception:
            duration = 12.0
        lines = [
            "Welcome everyone, let's get started with the weekly sync.",
            "The migration to the new backend finished ahead of schedule.",
            "Naomi will send the updated report to the client by Friday.",
            "We agreed to revisit the pricing discussion next week.",
        ]
        n = max(1, min(len(lines), int(duration // 3) or 1))
        step = duration / n
        segments = [{
            "start_s": round(i * step, 2),
            "end_s": round((i + 1) * step, 2),
            "speaker": None,
            "text": lines[i % len(lines)],
        } for i in range(n)]
        if progress is not None:
            progress(duration)
        return segments, {"language": "en", "duration": duration}

    def close(self):
        pass


def create_transcriber(config):
    if os.environ.get("ECHONOTE_STUB_ASR"):
        return StubTranscriber(config)
    return Transcriber(config)
