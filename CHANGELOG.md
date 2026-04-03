# Changelog

## [Unreleased]

---

## [1.1.0] - 2026-04-03

### Fixed

- **ASR stub emitted fake transcripts instead of transcribing real speech.**
  The `_run_stub_session` fallback (used when Riva is not installed) previously
  discarded all microphone audio and emitted a counter-based placeholder string
  (`"stub transcript N"`) every three seconds.  It now uses
  `faster-whisper` (tiny model, CPU, INT8) to perform real speech recognition.
  Audio is accumulated from the mic queue, an RMS energy VAD detects utterance
  boundaries, and each complete utterance is transcribed in a thread-pool
  executor so the async event loop is never blocked.  The original placeholder
  stub is preserved as `_run_original_stub_session` and is used only when
  `faster-whisper` cannot be imported.

- **TTS stub sent silent PCM (all-zero bytes) instead of audible speech.**
  The `_synthesize_stub` fallback previously packed zero-valued Int16 samples
  and streamed them to the browser, producing no audible output.  It now calls
  `edge-tts` (Microsoft Edge neural TTS, `en-US-AriaNeural` voice) to
  synthesize real speech, decodes the MP3 response to 16-bit mono PCM at the
  configured sample rate (22050 Hz) using PyAV's built-in resampler, and
  streams the result in 200 ms chunks.  A silence fallback (`_synthesize_silence`)
  is retained for environments where `edge-tts` or `av` cannot be imported.

### Added

- `asr.py` — `_whisper_transcribe`: helper coroutine that runs Whisper
  inference in a thread-pool executor and pushes a `TranscriptResult` to the
  transcript queue.
- `asr.py` — `_run_original_stub_session`: preserves legacy placeholder
  behaviour for offline/no-dependency environments.
- `tts.py` — `_synthesize_edge_tts`: full edge-tts → PyAV MP3 decode →
  PCM resample → chunk-stream pipeline.
- `tts.py` — `_synthesize_silence`: explicit silence fallback (previously
  inlined in `_synthesize_stub`).

### Dependencies added

```
faster-whisper>=1.2.1   # Whisper tiny model, CPU INT8, real ASR
edge-tts>=7.2.0         # Microsoft Edge neural TTS, no API key required
av>=12.0.0              # PyAV — MP3 decode + PCM resample for TTS output
```

Install with:

```bash
pip install faster-whisper edge-tts av
```

---

## [1.0.0] - 2026-04-03

### Added

- Initial release: full-duplex voice AI agent.
- NVIDIA Riva streaming ASR + TTS via gRPC.
- LLaMA 3 via Ollama for local LLM responses.
- FAISS RAG over telecom knowledge base (`docs/`).
- FastAPI + WebSocket real-time backend.
- Browser AudioWorklet mic capture and TTS playback.
- VAD interrupt flow with < 100 ms latency.
- Stub/dev mode: runs without GPU or Riva.
