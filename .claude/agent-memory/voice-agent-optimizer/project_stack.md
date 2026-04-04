---
name: Voice Agent Stack Architecture
description: Runtime stack, file layout, and audio format contracts for this project
type: project
---

Stack in use (as of 2026-04-04):
- ASR: faster-whisper (Whisper tiny, CPU, int8) — Riva stub mode since nvidia-riva-client not installed
- TTS: edge-tts → PyAV decode → 24 kHz s16 PCM — Riva stub mode
- LLM: Ollama (llama3) with Claude Haiku 4.5 fallback via ANTHROPIC_API_KEY
- RAG: FAISS IndexFlatIP + sentence-transformers/all-MiniLM-L6-v2, cached at backend/faiss_index/
- Backend: FastAPI + asyncio WebSockets, single worker
- Frontend: AudioWorklet PCM capture (16 kHz → server), Web Audio API TTS playback (24 kHz)

Audio format contracts:
- Mic → server: PCM 16-bit mono 16 kHz, ~3200 bytes per 100 ms chunk
- Server → browser: PCM 16-bit mono 24 kHz (edge-tts native, no resampling), 60 ms chunks = 2880 bytes
- Frontend PLAYBACK_SAMPLE_RATE = 24000 (must match backend EDGE_TTS_RATE = 24000)

Key files:
- backend/session_manager.py — session lifecycle, greeting flow, LLM+TTS co-routine
- backend/tts.py — edge-tts synthesis, PyAV decode, noise reduction, chunk streaming
- backend/rag.py — TELECOM_SEED_DOCS list + Qobox docs embedded here, FAISSIndex build/load
- backend/asr.py — faster-whisper stub, VAD via RMS energy (threshold 0.008 RMS)
- backend/llm.py — Ollama streaming, Claude fallback, sentence-boundary fragment splitter
- backend/config.py — all tunable parameters, system_prompt for Wesaal persona
- frontend/index.html — single-file frontend with AudioWorklet, VAD, TTS playback queue

**Why:** Architectural snapshot taken after first major optimization pass (2026-04-04).
**How to apply:** Use this to orient quickly at the start of future sessions without re-reading all files.
