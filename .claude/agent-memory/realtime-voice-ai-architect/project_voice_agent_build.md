---
name: Voice AI Agent — initial system build
description: Full real-time voice agent built 2026-04-03 in /workspace/voiceagentcloud/. Telecom CSR persona, Riva ASR+TTS, LLaMA 3 via Ollama, FAISS RAG.
type: project
---

Full duplex voice agent built to /workspace/voiceagentcloud/ on 2026-04-03.

Key facts:
- Backend: FastAPI + WebSocket at /ws, single-worker uvicorn (asyncio state not fork-safe)
- ASR: NVIDIA Riva gRPC, PCM 16-bit mono 16kHz, 3200-byte chunks (100ms)
- TTS: NVIDIA Riva streaming synthesis, LINEAR_PCM 22050Hz output
- LLM: Ollama llama3, stream=True, sentence-fragment buffering before TTS
- RAG: FAISS IndexFlatIP (cosine via normalized embeddings), sentence-transformers/all-MiniLM-L6-v2, cached to ./faiss_index/
- Memory: deque with maxlen = max_turns * 2 half-turns
- Interrupt: tts_cancel_event (asyncio.Event) checked between every TTS chunk; VAD in JS triggers {"type":"interrupt"} WS message
- Frontend: pure HTML/JS, AudioWorklet for PCM capture, AudioBufferSourceNode chaining for gapless playback

**Why:** User wanted a production-grade system with sub-second latency, full interrupt support, and telecom domain RAG.

**How to apply:** When extending, maintain the three-loop async architecture. Never batch ASR or TTS. Always check tts_cancel_event between chunks.
