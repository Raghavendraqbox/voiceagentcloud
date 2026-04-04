---
name: Greeting overlap bug root cause and fix
description: The greeting TTS was fired immediately on connect; user speech caused a second TTS to start before the first finished
type: project
---

Root cause (fixed 2026-04-04):
The original `_llm_tts_loop` called `_play_hardcoded(greeting)` synchronously at startup before entering the transcript-wait loop. When the user spoke during the greeting, the loop fell into Phase 1 which called `_play_hardcoded(lang_confirm)` — this created a brand-new TTSOrchestrator and fired TTS audio while the greeting audio was still streaming, causing audible overlap.

Fix applied:
Removed the immediate greeting. The loop now waits for the first user utterance before producing any audio. On the first transcript, it combines greeting + English confirmation into a single `_play_hardcoded` call (`greeting_and_confirm`), then continues to Phase 1+ (LLM turns). This guarantees TTS output is strictly sequential — one `_play_hardcoded` or one LLM turn at a time, never concurrent.

**Why:** Any audio played before the loop entered its `await transcript_queue.get()` state could overlap with the first response because `_play_hardcoded` itself awaits TTS completion but the outer loop could re-enter before cancellation finished.
**How to apply:** Never fire TTS output outside the main `while True` transcript loop. All audio must originate from within the turn-processing block so cancel_tts() / reset_for_new_turn() always runs first.
