# Changelog

## [Unreleased]

---

## [1.3.0] - 2026-04-03

### Fixed

- **Transcription latency was too high (~1.2 s after speech ended).**
  Three compounding issues:
  1. `SILENCE_FRAMES_TO_COMMIT` was 12 (1.2 s). Reduced to 6 (0.6 s) so
     utterances are committed twice as fast.
  2. `audio_buf.extend(samples.tolist())` converted each 100 ms numpy chunk
     into individual Python floats and appended them one by one — O(n) list
     growth per frame.  Changed to `audio_buf.append(samples)` (store the
     numpy array itself) and `np.concatenate(audio_buf)` at transcription
     time — O(1) per frame, single copy at commit.
  3. `np.array(audio_buf, dtype=np.float32)` in `_whisper_transcribe` had to
     rebuild the entire flat array from a Python list.  Now uses
     `np.concatenate(audio_buf)` directly on the stored arrays.

- **TTS output had audible clicks and noise at start/end of each response.**
  MP3 encoding introduces encoder-delay priming: the encoder prepends up to
  1152 near-zero samples before real audio begins.  The PyAV resampler flush
  (`resampler.resample(None)`) appends FIR filter tail samples at the end.
  Both produce clicks/pops when the browser's Web Audio scheduler plays the
  PCM.  Fixed by trimming all leading and trailing samples below a 160
  Int16-unit threshold (~0.5 % of full scale) from the decoded PCM before
  streaming.

- **`ttsAudioCtx` could be auto-suspended by the browser between turns.**
  Added `ttsAudioCtx.resume()` call in the `tts_start` handler so the
  playback context is guaranteed to be running before audio chunks arrive.

### Changed

- `backend/asr.py` — `SILENCE_FRAMES_TO_COMMIT` 12 → 6; audio buffer stores
  numpy arrays instead of flat floats.
- `backend/tts.py` — PCM silence trimming in `_synthesize_edge_tts` using
  numpy; decoded frames stored as arrays and concatenated once.
- `frontend/index.html` — `PREBUFFER_MS` 200 → 300 ms to absorb scheduling
  jitter; `ttsAudioCtx.resume()` on `tts_start`.

---

## [1.2.0] - 2026-04-03

### Fixed

- **TTS audio quality was muffled due to AudioContext sample rate mismatch.**
  The browser's mic capture `AudioContext` was created at 16 kHz and also
  used for TTS playback. Scheduling 22050 Hz `AudioBuffer`s inside a 16 kHz
  context caused the browser to resample audio down, producing muffled,
  degraded speech. A dedicated `ttsAudioCtx` is now created at exactly
  22050 Hz for playback only, eliminating all resampling.

- **Full duplex was broken: mic capture froze during bot interrupts.**
  `stopTTSPlayback()` called `audioCtx.suspend()` on the shared
  `AudioContext`. Since the same context drives the `AudioWorklet` mic
  capture, suspending it silenced the microphone at the exact moment the
  user tried to interrupt the bot. Fixed by tracking each scheduled
  `AudioBufferSourceNode` in a `ttsSourceNodes[]` array and calling
  `.stop(0)` on them directly — the mic `AudioContext` is never touched.

- **LLM stub hallucinated product details unprompted.**
  The offline stub response pool (`llm.py`) contained the hardcoded line
  `"Alright, the cheapest option is the 400MB hourly bundle for $1. Want me
  to activate that?"` which was randomly returned on every turn when Ollama
  was not running. All stub responses are now neutral acknowledgements
  (`"Got it, let me look into that for you."` etc.) with no invented
  product, pricing, or plan details.

### Changed

- `frontend/index.html` — `ttsAudioCtx` (new) handles all TTS playback at
  22050 Hz; `audioCtx` is now mic-capture only at 16 kHz.
- `frontend/index.html` — `stopTTSPlayback()` stops source nodes via
  `.stop(0)` instead of `audioCtx.suspend()/resume()`.
- `frontend/index.html` — `ttsSourceNodes[]` array tracks live
  `AudioBufferSourceNode`s and cleans up via `ended` event listeners.
- `llm.py` — stub response pool stripped of all hardcoded product/price
  references.

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
