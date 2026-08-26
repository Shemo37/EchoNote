"""
EchoNote configuration: versioned JSON file under the data directory.

Follows the same migrate-only-unchanged-defaults pattern as HoloLiveTL:
user-tuned values survive upgrades, stale defaults get bumped.
"""
import json
import os
import threading

CONFIG_VERSION = 1

DATA_DIR = os.environ.get("ECHONOTE_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
DB_PATH = os.path.join(DATA_DIR, "echonote.db")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")

DEFAULTS = {
    # ASR (faster-whisper). Model may be a size ("small", "large-v3-turbo")
    # or any CTranslate2 repo id on the Hub.
    "asr_model": "small",
    "compute_type": "auto",          # auto | float16 | int8_float16 | int8
    "device": "auto",                # auto | cuda | cpu
    "language": "auto",              # auto-detect, or a fixed code like "en"/"ja"
    # Custom vocabulary fed to the decoder (comma-separated names/terms)
    "hotwords": "",

    # Speaker diarization (pyannote) - used only when a token is present
    "hf_token": None,
    "diarization": True,

    # Local LLM (Ollama)
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "llama3.2",

    # Server
    "host": "127.0.0.1",
    "port": 8321,

    "config_version": CONFIG_VERSION,
}

VALIDATORS = {
    "compute_type": ("auto", "float16", "int8_float16", "int8"),
    "device": ("auto", "cuda", "cpu"),
}

_lock = threading.Lock()


class Config:
    def __init__(self, path=None):
        self.path = path or CONFIG_PATH
        self.load()

    def load(self):
        data = dict(DEFAULTS)
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                data.update(loaded)
                self._migrate(data, loaded)
            except Exception as e:
                print(f"Config load error ({e}); using defaults")
        self._validate(data)
        self.__dict__.update(data)

    @staticmethod
    def _migrate(data, loaded):
        if loaded.get("config_version"):
            return
        # v0 -> v1: nothing to migrate yet; stamp the version
        data["config_version"] = CONFIG_VERSION

    @staticmethod
    def _validate(data):
        for key, allowed in VALIDATORS.items():
            if data.get(key) not in allowed:
                print(f"Config: invalid {key}={data.get(key)!r}, using {DEFAULTS[key]!r}")
                data[key] = DEFAULTS[key]
        try:
            data["port"] = int(data["port"])
        except (TypeError, ValueError):
            data["port"] = DEFAULTS["port"]

    def save(self):
        with _lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            payload = {k: v for k, v in self.__dict__.items() if k != "path"}
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    def update(self, changes: dict):
        """Apply a partial update (settings API), validate, persist."""
        editable = set(DEFAULTS) - {"config_version"}
        data = {k: v for k, v in self.__dict__.items() if k != "path"}
        for key, value in changes.items():
            if key in editable:
                data[key] = value
        self._validate(data)
        self.__dict__.update(data)
        self.save()

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if k != "path"}
        # never leak the HF token back to the browser in full
        if d.get("hf_token"):
            d["hf_token_set"] = True
            d["hf_token"] = ""
        else:
            d["hf_token_set"] = False
        return d


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
