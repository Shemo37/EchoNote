"""
Local LLM access via Ollama's REST API.

Everything degrades gracefully: when Ollama isn't reachable, callers get
LLMUnavailable with install instructions instead of a traceback, and the
app keeps working as a transcribe-only tool.

Set ECHONOTE_STUB_LLM=1 for a deterministic fake (tests/CI).
"""
import json
import logging
import os
import re

import requests

logger = logging.getLogger(__name__)

INSTALL_HINT = ("Ollama is not running. Install it from https://ollama.com, then run "
                "'ollama pull llama3.2' and start the app again to enable AI features.")


class LLMUnavailable(Exception):
    pass


class OllamaClient:
    def __init__(self, config):
        self.config = config

    @property
    def base(self):
        return (self.config.ollama_url or "http://127.0.0.1:11434").rstrip("/")

    def available(self):
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def models(self):
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=3)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException:
            return []

    def chat_stream(self, messages):
        """Yield response text chunks for a chat conversation."""
        try:
            r = requests.post(
                f"{self.base}/api/chat",
                json={"model": self.config.ollama_model, "messages": messages, "stream": True},
                stream=True, timeout=(5, 600))
        except requests.RequestException as e:
            raise LLMUnavailable(INSTALL_HINT) from e
        if r.status_code == 404:
            raise LLMUnavailable(
                f"Model '{self.config.ollama_model}' not found in Ollama - run "
                f"'ollama pull {self.config.ollama_model}' or pick another model in Settings.")
        if r.status_code != 200:
            raise LLMUnavailable(f"Ollama error HTTP {r.status_code}")
        for line in r.iter_lines():
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            chunk = payload.get("message", {}).get("content", "")
            if chunk:
                yield chunk
            if payload.get("done"):
                break

    def chat(self, messages):
        return "".join(self.chat_stream(messages))


class StubLLM:
    """Deterministic fake LLM for tests and container E2E."""

    def __init__(self, config):
        self.config = config

    def available(self):
        return True

    def models(self):
        return ["stub"]

    def chat_stream(self, messages):
        prompt = messages[-1]["content"] if messages else ""
        if "one short sentence" in prompt.lower():
            yield "Weekly sync covering the backend migration and client report."
            return
        if "action item" in prompt.lower() or "ACTION_ITEMS" in prompt:
            yield ("## Summary\n- Backend migration finished ahead of schedule [00:03]\n"
                   "- Pricing discussion moved to next week [00:09]\n\n"
                   "## Action Items\n- [ ] Naomi: send updated report to client by Friday\n"
                   "- [ ] Schedule pricing discussion for next week\n")
            return
        yield ("The report is due Friday - Naomi committed to sending it to the "
               "client [00:06].")

    def chat(self, messages):
        return "".join(self.chat_stream(messages))


def create_llm(config):
    if os.environ.get("ECHONOTE_STUB_LLM"):
        return StubLLM(config)
    return OllamaClient(config)


# ---- transcript formatting + citation helpers ----

def format_timestamp(seconds):
    m, s = divmod(int(max(seconds, 0)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcript_for_prompt(segments, max_chars=24000):
    """Render segments as '[mm:ss] Speaker: text' lines, tail-truncated."""
    lines = []
    for seg in segments:
        who = f"{seg.get('speaker')}: " if seg.get("speaker") else ""
        lines.append(f"[{format_timestamp(seg['start_s'])}] {who}{seg['text']}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... transcript truncated ...]"
    return text


ACTION_ITEM_RE = re.compile(r"^\s*[-*]\s*\[\s?\]\s*(.+)$", re.MULTILINE)


def extract_action_items(markdown):
    """Pull '- [ ] item' checkboxes out of a generated summary."""
    return [m.group(1).strip() for m in ACTION_ITEM_RE.finditer(markdown or "")]


CITATION_RE = re.compile(r"\[(\d{1,2}:)?(\d{1,2}):(\d{2})\]")


def parse_citation(text_match):
    """'[mm:ss]' or '[h:mm:ss]' -> seconds."""
    h = int((text_match.group(1) or "0").rstrip(":") or 0)
    return h * 3600 + int(text_match.group(2)) * 60 + int(text_match.group(3))
