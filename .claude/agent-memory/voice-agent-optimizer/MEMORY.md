# Voice Agent Optimizer — Memory Index

- [Voice Agent Stack Architecture](project_stack.md) — Runtime stack, file layout, audio format contracts (24 kHz end-to-end, 60 ms TTS chunks)
- [Greeting Overlap Bug Fix](project_greeting_overlap_fix.md) — Deferred greeting to first user utterance; never fire TTS outside the turn loop
- [TTS Noise Sources and Fixes](tts_noise_fixes.md) — Symmetric int16→float32 divisor, scheduling look-ahead, float drift, chunk size reduction
- [RAG Qobox Data Added](rag_qobox.md) — Qobox company info in TELECOM_SEED_DOCS; cached FAISS index must be deleted after seed doc changes
