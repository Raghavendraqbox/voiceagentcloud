---
name: TTS audio noise sources and fixes
description: Identified noise/artifact sources in the edge-tts → browser audio pipeline and the fixes applied
type: project
---

Noise sources found and fixed (2026-04-04):

1. **int16ToFloat32 asymmetric divisor (frontend/index.html)**
   - Bug: `int16Array[i] / (int16Array[i] < 0 ? 0x8000 : 0x7FFF)` divided positive and negative samples by different constants (32767 vs 32768), creating a subtle DC-offset waveform asymmetry that manifested as harmonic distortion on voiced speech.
   - Fix: `int16Array[i] / 32768.0` — symmetric divisor for all samples.

2. **AudioBufferSourceNode scheduling look-ahead too small (frontend/index.html)**
   - Bug: `ttsScheduledUntil` was snapped to `now` exactly when behind, meaning the AudioContext had zero scheduling headroom for the first chunk after a gap. Any event-loop jitter caused the buffer to be late, producing a pop/click at the start of each TTS response.
   - Fix: Added 5 ms look-ahead (`SCHEDULE_AHEAD_S = 0.005`). If `ttsScheduledUntil < now + 0.005`, snap forward to `now + 0.005`.

3. **Floating-point drift in `ttsScheduledUntil` accumulation (frontend/index.html)**
   - Bug: `ttsScheduledUntil += buffer.duration` accumulates floating-point rounding error across many chunks. On long responses (20+ chunks), this caused micro-gaps at chunk boundaries.
   - Fix: `ttsScheduledUntil += float32.length / PLAYBACK_SAMPLE_RATE` — integer sample count division is more numerically stable than summing AudioBuffer.duration (which is already a float).

4. **TTS chunk size too large — 200 ms (backend/tts.py)**
   - Bug: 200 ms chunks at 24 kHz = 4800 samples = 9600 bytes. The browser prebuffer (300 ms) required 2 chunks before playback started, adding ~400 ms of perceived latency. Large chunks also meant long gaps in cancel responsiveness.
   - Fix: Reduced to 60 ms chunks (2880 bytes). Prebuffer fills after 5 chunks (~300 ms), first audio arrives sooner, cancel latency reduced to max 60 ms.

Known-good configuration after fixes:
- EDGE_TTS_RATE = 24000 (backend)
- PLAYBACK_SAMPLE_RATE = 24000 (frontend) — must match exactly, no browser resampling
- Trim threshold: 160 (~0.5% of Int16 full-scale) — removes MP3 encoder-delay silence
- Fade in/out: 20 ms linear ramp — prevents click at utterance boundaries
- Chunk size: 60 ms = 2880 bytes

**Why:** These noise sources were all sub-perceptual individually but compounded into audible artifacts (crackling, DC hum, click-on-start) during real usage.
**How to apply:** When adding new TTS backends, always verify: (a) symmetric int16→float32 conversion, (b) scheduling look-ahead of at least 5 ms, (c) chunk size ≤ 100 ms, (d) sample rates match end-to-end.
