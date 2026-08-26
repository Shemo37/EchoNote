# EchoNote 🎙️

**A self-hosted, fully local AI voice notes system** — record or upload conversations,
get transcripts with speaker labels, AI summaries with action items, and ask questions
about your recordings with answers that cite the audio. Like a Plaud.ai workflow, but
everything runs on your own machine: your audio never leaves it.

Forked from the ASR/diarization stack of [HoloLiveTL](https://github.com/Shemo37/HoloLiveTL).

## What it does

- **Record in the browser or upload files** (mp3/m4a/wav/webm…). Processing starts
  after the recording is complete — nothing is streamed anywhere.
- **Transcription** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (CTranslate2), multilingual with language auto-detect. Model size is configurable
  from `tiny` to `large-v3-turbo`.
- **Speaker labels** ("who said what") via [pyannote](https://github.com/pyannote/pyannote-audio) — optional.
- **AI summaries** with templates (meeting minutes, action items, key points,
  interview) powered by a local LLM through [Ollama](https://ollama.com) — optional.
- **PDF export**: download any generated summary (e.g. meeting minutes) as a clean
  A4 PDF with embedded fonts — Japanese content included.
- **Ask your recordings**: chat about a recording (or across recent ones); answers
  cite `[mm:ss]` timestamps you can click to jump the audio player.
- **Dashboard**: recordings/hours stats, 14-day activity, aggregated checkable
  action items harvested from summaries.
- **Custom vocabulary**: names and terms fed to the decoder as hotwords.

Everything is stored locally in `data/` (SQLite + audio files).

## Quickstart

```bash
git clone https://github.com/Shemo37/EchoNote
cd EchoNote
pip install -r requirements.txt
python main.py        # opens http://127.0.0.1:8321
```

Then upload an audio file or press **Record**.

### Optional components (each unlocks a feature, nothing breaks without them)

| Component | Unlocks | Setup |
|---|---|---|
| **ffmpeg** | importing mp3/m4a/webm (WAV works without it) | [ffmpeg.org](https://ffmpeg.org) — on Windows: `winget install ffmpeg` |
| **Ollama** | summaries, action items, Ask | install from [ollama.com](https://ollama.com), then `ollama pull llama3.2` |
| **pyannote** | speaker labels | `pip install pyannote.audio`, create a [HF token](https://huggingface.co/settings/tokens), accept the [pyannote license](https://huggingface.co/pyannote/speaker-diarization-3.1), paste the token in Settings |
| **NVIDIA GPU** | much faster transcription (float16) | CUDA + cuDNN 9 (ships with recent torch wheels) |

Without a GPU everything still works — whisper `small` on CPU (int8) is fine for
meeting-length audio, just not instant.

## Project structure

```
main.py               # start server + open browser
server/
  app.py              # FastAPI: REST API + static SPA
  pipeline.py         # background job: convert → transcribe → diarize → gist
  asr.py              # faster-whisper transcriber (hotwords, auto language)
  diarize.py          # pyannote speaker turns → per-segment labels
  llm.py              # Ollama client, summary/citation helpers
  templates/          # summary prompt templates (add your own .md here)
  db.py, config.py    # SQLite storage, versioned JSON config
web/                  # vanilla JS/CSS single-page app (no build step)
tests/                # pytest suite (runs with stubbed ASR/LLM, no GPU needed)
```

## Adding a summary template

Drop a Markdown file in `server/templates/`, e.g. `sales-call.md`, containing your
prompt with a `{transcript}` placeholder. It appears in the Summary dropdown
immediately. Use `- [ ]` checkboxes for anything you want harvested into the
dashboard's Action Items.

## Development

```bash
pip install pytest
python -m pytest tests/     # uses stub ASR/diarizer/LLM - fast, no models
```

## License

MIT
