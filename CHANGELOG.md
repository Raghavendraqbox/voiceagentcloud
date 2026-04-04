# Changelog

## [Unreleased]

---

## [1.5.0] - 2026-04-04

### Changed — Qobox rebrand & cloud GPU stack

- **Agent identity changed to Qobox Assistant.**
  System prompt rewritten: agent represents *Qobox (Quality Outside The Box)*,
  an Indian software QA company.  Greeting is now *"Welcome to Qobox. I am
  your virtual assistant."*  Out-of-scope queries (anything unrelated to
  software testing / QA) are rejected with *"I'm sorry, I can only assist
  you with issues related to Qobox."*

- **Name memory via conversation history.**
  System prompt explicitly instructs the model to read the conversation
  history and use the user's name when they have already given it.

- **NVIDIA Riva via NVCF cloud gRPC (no Docker required).**
  When `NVIDIA_API_KEY` is set the server authenticates to
  `grpc.nvcf.nvidia.com:443` using the Riva Python SDK with
  `function-id` + `authorization` gRPC metadata headers.
  NVCF function IDs configured in `config.py`:
  - ASR: Parakeet-CTC-1.1B  (`1598d209-5e27-4d3c-8079-4751568b1081`)
  - TTS: FastPitch-HiFiGAN   (`bc45d9e9-7c78-4d56-9737-e27011962ba8`)

- **Kokoro GPU TTS as primary fallback (replaces edge-tts as first fallback).**
  `tts.py` completely rewritten.  A module-level `_kokoro_pipeline` singleton
  is initialised once on first TTS call (thread-safe via `threading.Lock`).
  Synthesis streams chunk-by-chunk — first audio reaches the browser in ~50 ms.
  `cancel_event` is checked inside the generation thread and in the async
  dispatch loop for < 200 ms interrupt latency.  A background `schedule_kokoro_warmup()`
  thread pre-compiles GPU kernels after app startup.
  Fallback chain: **Riva → Kokoro → edge-tts → silence**.

- **Whisper `small` GPU (float16 CUDA) as ASR fallback.**
  When Riva is unavailable the stub session now loads `whisper-small` on CUDA
  instead of `whisper-tiny` on CPU, significantly improving transcription
  accuracy.

- **NVIDIA NIM (Nemotron-70B) as second LLM fallback.**
  `llm.py` fallback chain is now: **Ollama → NIM (if `NVIDIA_API_KEY`) → Claude API → neutral stub**.
  NIM uses `nvidia/llama-3.1-nemotron-70b-instruct` via `integrate.api.nvidia.com`.

- **RAG knowledge base updated for Qobox.**
  Seven seed documents added to `rag.py`: company identity, services catalogue,
  target industries, mission statement, tech stack, culture, and future focus.
  `RAG_SIMILARITY_THRESHOLD` lowered 0.5 → 0.3 for better recall on Qobox
  queries.

- **VAD tuning — background-noise immunity.**
  Raised `VAD_THRESHOLD` (0.010 → 0.018) and `VAD_HOLD_FRAMES` (2 → 4) in
  the frontend to prevent ambient noise from firing a premature interrupt
  while the bot is speaking.  `PREBUFFER_MS` raised 150 → 250 ms so the
  browser audio buffer is full before the VAD can fire.

- **ASR streaming fixed — no longer blocks forever.**
  `_run_riva_session()` used `list()` to materialise all Riva responses before
  processing — blocking the event loop indefinitely.  Changed to
  `_run_riva_streaming()` (thread + queue) so transcripts arrive while audio
  is still being sent.

- **`.env.example` added** — documents every supported environment variable
  with defaults and descriptions so new contributors can set up the project
  from a clone without reading the source.

- **`PLAYBACK_SAMPLE_RATE` set to 22050 Hz** in the frontend to match the
  Riva TTS native output rate (Kokoro outputs 24000 Hz but is also accepted
  at 22050 via the same WebSocket binary path).

### Files changed

- `backend/config.py` — Qobox system prompt; NVCF URIs & function IDs;
  `RAG_SIMILARITY_THRESHOLD` 0.5 → 0.3; default `OLLAMA_MODEL` → `llama3.2:1b`
- `backend/asr.py` — NVCF auth; `_run_riva_streaming()` replaces blocking path;
  Whisper small/cuda/float16
- `backend/tts.py` — full rewrite: Kokoro singleton, NVCF auth, chunk streaming,
  cancel-aware generation thread, warmup scheduler
- `backend/llm.py` — `_nim_http` client; `_stream_fragments_nim()`;
  Ollama → NIM → Claude → stub fallback chain
- `backend/rag.py` — Qobox seed documents; threshold 0.5 → 0.3
- `backend/session_manager.py` — Qobox greeting; `cancel_tts()` sets
  `interrupt_event`; calls `schedule_kokoro_warmup()` after `initialize_rag()`
- `frontend/index.html` — title/header → Qobox; `PLAYBACK_SAMPLE_RATE` 24000
  → 22050; `VAD_THRESHOLD` 0.010 → 0.018; `VAD_HOLD_FRAMES` 2 → 4;
  `PREBUFFER_MS` 150 → 250; `tts_start` flushes stale audio buffer;
  `int16ToFloat32` uses symmetric `/ 32768.0`
- `.env.example` — new file
- `CHANGELOG.md` — this entry

---

## [1.4.0] - 2026-04-03

### Added

- **Agent identity — Wesaal / Etisalat Afghanistan.**
  Agent name is now Wesaal.  System prompt updated to reflect Etisalat
  Afghanistan brand, instruct the agent never to volunteer product prices
  or bundle names unprompted, and to say "Let me connect you to a
  specialist" when it does not know an answer.

- **Automatic greeting on session start.**
  As soon as the WebSocket session is ready, Wesaal plays: *"Welcome to
  Etisalat Afghanistan. I am Wesaal, your virtual assistant. Please tell
  me your preferred language: English, Dari, or Pashto."* without waiting
  for user input.

- **Language selection flow.**
  The first user utterance is treated as a language choice.  Regardless of
  what is said, Wesaal confirms English ("I'll assist you in English
  today.") and proceeds.  All subsequent turns are handled by the LLM.

- **Claude API (Haiku) as LLM backend.**
  `llm.py` now tries Ollama first; on connection failure it streams from
  `claude-haiku-4-5-20251001` using the `ANTHROPIC_API_KEY` env variable,
  falling back to a neutral neutral stub only if neither is available.
  Responses are contextually relevant to what the user actually said.

### Fixed

- **TTS noise from resampling artifacts.**
  edge-tts outputs 24 kHz audio; previous code resampled to 22050 Hz via a
  complex FIR filter that introduced ringing artefacts.  The pipeline now
  decodes at the native 24 kHz without any sample-rate conversion.
  `PLAYBACK_SAMPLE_RATE` on the frontend updated to 24000 accordingly.
  Voice changed to `en-US-JennyNeural` (higher perceived quality for
  customer service scenarios).  A 20 ms linear fade-in/fade-out is applied
  to each synthesised segment to prevent click artefacts at boundaries.

- **Short words (e.g. "hello") silently swallowed.**
  `MIN_SPEECH_FRAMES` reduced from 3 → 1 so even a single 100 ms voice
  frame triggers a transcription commit.

- **Subsequent speech appended to previous short utterance.**
  When `MIN_SPEECH_FRAMES` wasn't met, `speech_started` stayed `True` and
  the buffer kept accumulating.  Now the buffer is always reset once
  `SILENCE_FRAMES_TO_COMMIT` is reached, regardless of whether the
  transcription threshold was met.  A `MAX_SILENCE_FRAMES = 20` (2 s)
  hard-reset also clears stale buffers unconditionally.

### Changed

- `config.py` — system prompt, agent name, Claude model setting.
- `llm.py` — Claude API streaming path + `_build_claude_user_message`.
- `session_manager.py` — `_play_hardcoded` helper; greeting + language
  phase in `_llm_tts_loop`; `language_selected` flag on `Session`.
- `asr.py` — `MIN_SPEECH_FRAMES` 3→1; always-reset on silence commit;
  `MAX_SILENCE_FRAMES` hard-reset.
- `tts.py` — 24 kHz native decode, `en-US-JennyNeural`, fade-in/out.
- `frontend/index.html` — `PLAYBACK_SAMPLE_RATE` 22050→24000; page title
  and header updated to "Wesaal / Etisalat Afghanistan".

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
