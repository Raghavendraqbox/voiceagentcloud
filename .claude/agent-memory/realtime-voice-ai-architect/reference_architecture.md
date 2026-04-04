---
name: Architecture decisions and reliable parameter values
description: Riva gRPC params, audio chunk sizes, FAISS config, and latency-critical implementation choices validated in this build.
type: reference
---

## Riva gRPC — reliable parameters

- ASR: LINEAR_PCM, 16000Hz, mono, interim_results=True, automatic_punctuation=True
- ASR chunk: 3200 bytes = 1600 samples = 100ms at 16kHz (matches frontend send interval)
- TTS: LINEAR_PCM, 22050Hz, voice=English-US.Female-1
- gRPC default port: 50051 (localhost)
- Riva SDK: nvidia-riva-client 2.14.0 — streaming_response_generator is a synchronous iterator; must be run in ThreadPoolExecutor

## FAISS

- Index type: IndexFlatIP with normalized embeddings (equivalent to cosine similarity)
- Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast)
- Persisted to: ./faiss_index/index.faiss + ./faiss_index/chunks.pkl
- top_k=3, similarity_threshold=0.5
- Chunk size: 300 words, overlap: 50 words

## Audio pipeline (frontend)

- AudioWorklet preferred over ScriptProcessor (lower latency, no main-thread blocking)
- Send interval: 100ms via setInterval
- TTS playback: AudioBufferSourceNode queue chained on audioCtx timeline — eliminates inter-chunk gaps
- Prebuffer: 200ms before starting playback to avoid underrun at slow connections
- VAD: RMS threshold 0.012, hold for 8 consecutive frames to avoid false positives from single loud transient
- Interrupt: suspend+resume AudioContext to flush scheduled buffers immediately

## LLM sentence splitting

- Split on [.!?,] followed by whitespace or EOS — yields fragments of ~1-2 sentences
- Do NOT split on period-in-number (e.g., $1.50) — the regex handles this since it requires whitespace after punctuation

## Latency optimizations

- TTS starts after first sentence fragment (~15-30 tokens), not full LLM response
- Ollama num_predict=150 caps response length and reduces tail latency
- FAISS retrieval: sub-millisecond after first search (JIT compilation on first call)
- Riva TTS thread runs in daemon thread — does not block event loop
