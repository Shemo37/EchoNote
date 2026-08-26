"""
Speaker diarization (pyannote) for batch recordings.

Ported from HoloLiveTL's SpeakerDiarizer, batch-only: run the full pipeline
over the whole file once, then label each ASR segment with the speaker whose
turns overlap it most. Entirely optional - without a HuggingFace token (or
pyannote installed) recordings simply have no speaker labels.

Set ECHONOTE_STUB_DIARIZE=1 to alternate two fake speakers (tests/CI).
"""
import logging
import os

logger = logging.getLogger(__name__)

SPEAKER_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
]


def speaker_color(label):
    """Stable color per 'Speaker N' label."""
    try:
        n = int(str(label).rsplit(" ", 1)[-1])
    except (ValueError, IndexError):
        n = 0
    return SPEAKER_COLORS[(n - 1) % len(SPEAKER_COLORS)]


class Diarizer:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device
        self.pipeline = None

    def available(self):
        token = self.config.hf_token or os.environ.get("HF_TOKEN")
        return bool(self.config.diarization and token)

    def load(self):
        token = self.config.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("no HuggingFace token configured")
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token)
        if self.device == "cuda":
            import torch
            self.pipeline.to(torch.device("cuda"))
        logger.info("pyannote diarization pipeline loaded (%s)", self.device)

    def diarize(self, wav_path):
        """Return a list of (start_s, end_s, 'Speaker N') turns."""
        if self.pipeline is None:
            raise RuntimeError("Diarizer not loaded - call load() first")
        diarization = self.pipeline(wav_path)
        label_map = {}
        turns = []
        for turn, _, raw_label in diarization.itertracks(yield_label=True):
            if raw_label not in label_map:
                label_map[raw_label] = f"Speaker {len(label_map) + 1}"
            turns.append((float(turn.start), float(turn.end), label_map[raw_label]))
        return turns

    def close(self):
        self.pipeline = None


class StubDiarizer:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device

    def available(self):
        return True

    def load(self):
        pass

    def diarize(self, wav_path):
        # Two speakers alternating every 6 seconds across a long span;
        # assign_speakers() trims to actual segment overlap.
        turns = []
        for i in range(200):
            turns.append((i * 6.0, (i + 1) * 6.0, f"Speaker {(i % 2) + 1}"))
        return turns

    def close(self):
        pass


def assign_speakers(segments, turns):
    """Label each ASR segment with the speaker whose turns overlap it most.

    segments: dicts with start_s/end_s (mutated in place: sets 'speaker').
    turns: (start_s, end_s, label) from Diarizer.diarize().
    """
    for seg in segments:
        best_label, best_overlap = None, 0.0
        for t_start, t_end, label in turns:
            overlap = min(seg["end_s"], t_end) - max(seg["start_s"], t_start)
            if overlap > best_overlap:
                best_overlap, best_label = overlap, label
        if best_label is not None and best_overlap > 0:
            seg["speaker"] = best_label
    return segments


def create_diarizer(config, device="cpu"):
    if os.environ.get("ECHONOTE_STUB_DIARIZE"):
        return StubDiarizer(config, device)
    return Diarizer(config, device)
