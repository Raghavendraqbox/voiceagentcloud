"""
tts.py — GPU-accelerated TTS handler (Kokoro primary, edge-tts fallback).

Kokoro pipeline is a module-level singleton — loaded once at first use,
reused for every synthesis call.  Each call streams audio chunk-by-chunk
so the client receives the first audio within ~50 ms of synthesis start.

Cancel semantics:
  - cancel_event is checked INSIDE the generation thread between chunks
    AND in the async dispatch loop between send calls.
  - Setting cancel_event will abort within one Kokoro chunk (~200 ms).
"""

import asyncio
import logging
import queue as _queue
import threading
from typing import Callable, Awaitable

import numpy as np

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Riva SDK import guard
# ---------------------------------------------------------------------------
try:
    import riva.client  # type: ignore
    _RIVA_AVAILABLE = True
except ImportError:
    _RIVA_AVAILABLE = False
    logger.warning("nvidia-riva-client not installed — TTS will operate in stub mode")


# ---------------------------------------------------------------------------
# Kokoro singleton — one pipeline for the whole process lifetime
# ---------------------------------------------------------------------------
_kokoro_pipeline = None
_kokoro_init_lock = threading.Lock()


def _get_kokoro():
    """Return the process-wide Kokoro pipeline, initialising it on first call."""
    global _kokoro_pipeline
    if _kokoro_pipeline is not None:
        return _kokoro_pipeline

    with _kokoro_init_lock:
        if _kokoro_pipeline is not None:
            return _kokoro_pipeline
        try:
            import torch
            import kokoro as _kokoro_mod
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Initialising Kokoro TTS pipeline on %s …", device)
            _kokoro_pipeline = _kokoro_mod.KPipeline(lang_code="a", device=device)
            logger.info("Kokoro TTS ready on %s", device)
        except Exception as exc:
            logger.error("Failed to initialise Kokoro: %s", exc)
            _kokoro_pipeline = None

    return _kokoro_pipeline


# Kokoro is initialized lazily on first TTS call and cached after that.
# A background warmup is triggered after the FastAPI lifespan startup completes
# (called from session_manager.initialize_rag so it runs after all heavy imports
# have already resolved their transformers dependency).
def schedule_kokoro_warmup():
    """Call this AFTER app startup so all imports are settled before Kokoro loads."""
    def _warmup():
        try:
            pipe = _get_kokoro()
            if pipe is None:
                return
            for _ in pipe("hello", voice="af_heart", speed=1.0, split_pattern=None):
                break
            logger.info("Kokoro warmup complete — GPU kernels compiled")
        except Exception as exc:
            logger.warning("Kokoro warmup failed: %s", exc)

    threading.Thread(target=_warmup, daemon=True, name="kokoro-warmup").start()


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
AudioSendCallback = Callable[[bytes], Awaitable[None]]

# Sentinel used to signal an empty queue poll
_QUEUE_EMPTY = object()

# Kokoro native rate — only used when Riva is unavailable
KOKORO_RATE = 24000

# Riva TTS rate — must match frontend PLAYBACK_SAMPLE_RATE when Riva is active
RIVA_RATE = 22050


# ---------------------------------------------------------------------------
# RivaTTSHandler
# ---------------------------------------------------------------------------

class RivaTTSHandler:
    """
    Primary: Riva streaming TTS (when available).
    Fallback: Kokoro GPU TTS (chunk-streaming, cancel-aware).
    Last resort: edge-tts → silence.
    """

    def __init__(
        self,
        session_id: str,
        send_audio_cb: AudioSendCallback,
        cancel_event: asyncio.Event,
    ) -> None:
        self.session_id = session_id
        self._send_audio = send_audio_cb
        self._cancel_event = cancel_event

        if _RIVA_AVAILABLE:
            api_key = config.riva.nvidia_api_key
            if api_key:
                logger.info("TTS: connecting to NVIDIA Riva via NVCF cloud",
                            extra={"session_id": session_id})
                auth = riva.client.Auth(
                    uri=config.riva.nvcf_uri,
                    use_ssl=True,
                    metadata_args=[
                        ("function-id", config.riva.nvcf_tts_function_id),
                        ("authorization", f"Bearer {api_key}"),
                    ],
                )
            else:
                auth = riva.client.Auth(
                    uri=config.riva.server_url,
                    use_ssl=config.riva.use_ssl,
                    ssl_cert=config.riva.ssl_cert if config.riva.ssl_cert else None,
                )
            self._service = riva.client.SpeechSynthesisService(auth)
        else:
            self._service = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesize_and_stream(self, text: str) -> bool:
        if not text.strip():
            return True
        logger.info("TTS synthesize: %.60s", text, extra={"session_id": self.session_id})

        if _RIVA_AVAILABLE and self._service is not None:
            return await self._synthesize_riva(text)
        return await self._synthesize_kokoro(text)

    # ------------------------------------------------------------------
    # Riva synthesis
    # ------------------------------------------------------------------

    async def _synthesize_riva(self, text: str) -> bool:
        loop = asyncio.get_running_loop()
        response_queue: _queue.Queue = _queue.Queue()

        def riva_thread():
            try:
                responses = self._service.synthesize_online(
                    text=text,
                    voice_name=config.riva.tts_voice_name,
                    language_code=config.riva.tts_language_code,
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    sample_rate_hz=config.riva.tts_sample_rate_hz,
                )
                for resp in responses:
                    response_queue.put(resp.audio)
            except Exception as exc:
                response_queue.put(exc)
            finally:
                response_queue.put(None)

        thread = threading.Thread(target=riva_thread, daemon=True)
        thread.start()
        completed = True

        try:
            while True:
                item = await loop.run_in_executor(None, response_queue.get)
                if item is None:
                    break
                if isinstance(item, Exception):
                    logger.error("Riva TTS error: %s", item, extra={"session_id": self.session_id})
                    completed = False
                    break
                if self._cancel_event.is_set():
                    completed = False
                    break
                await self._send_audio(item)
        finally:
            thread.join(timeout=2.0)

        return completed

    # ------------------------------------------------------------------
    # Kokoro synthesis — streaming, cancel-aware
    # ------------------------------------------------------------------

    async def _synthesize_kokoro(self, text: str) -> bool:
        """
        Stream Kokoro GPU TTS chunk-by-chunk.

        The Kokoro generator yields one audio array per phoneme group (~200 ms).
        We run it in a daemon thread and forward each chunk to the client as
        soon as it arrives — first audio reaches the browser in ~50 ms.

        cancel_event is checked:
          1. Inside the generation thread (stops generator early).
          2. In the async dispatch loop between send() calls.
        """
        if self._cancel_event.is_set():
            return False

        pipeline = _get_kokoro()
        if pipeline is None:
            logger.warning("Kokoro unavailable — trying edge-tts fallback",
                           extra={"session_id": self.session_id})
            return await self._synthesize_edge_tts(text)

        cancel = self._cancel_event
        chunk_q: _queue.Queue = _queue.Queue(maxsize=32)

        def _generate():
            """Runs in a thread — feeds Kokoro output into chunk_q."""
            try:
                for _, _, audio in pipeline(text, voice="af_heart", speed=1.0,
                                             split_pattern=None):
                    if cancel.is_set():
                        break
                    if audio is not None:
                        arr = audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
                        chunk_q.put(arr.astype(np.float32))
            except Exception as exc:
                chunk_q.put(exc)
            finally:
                chunk_q.put(None)  # sentinel

        gen_thread = threading.Thread(target=_generate, daemon=True, name="kokoro-gen")
        gen_thread.start()

        loop = asyncio.get_running_loop()
        BYTES_PER_SEND = int(KOKORO_RATE * 0.060) * 2   # 60 ms chunks
        FADE_SAMPLES   = int(KOKORO_RATE * 0.005)        # 5 ms fade per chunk

        def _poll():
            """Non-blocking poll — returns _QUEUE_EMPTY if nothing ready yet."""
            try:
                return chunk_q.get(timeout=0.05)
            except _queue.Empty:
                return _QUEUE_EMPTY

        try:
            while True:
                if self._cancel_event.is_set():
                    return False

                item = await loop.run_in_executor(None, _poll)

                if item is _QUEUE_EMPTY:
                    continue

                if item is None:          # sentinel — generation complete
                    break

                if isinstance(item, Exception):
                    logger.error("Kokoro generation error: %s", item,
                                 extra={"session_id": self.session_id})
                    return False

                pcm_f32: np.ndarray = item

                # Short fade per chunk to prevent click at boundaries
                if len(pcm_f32) > FADE_SAMPLES * 2:
                    ramp = np.linspace(0.0, 1.0, FADE_SAMPLES, dtype=np.float32)
                    pcm_f32[:FADE_SAMPLES]  *= ramp
                    pcm_f32[-FADE_SAMPLES:] *= ramp[::-1]

                # Normalise peak to 92 % FS to avoid clipping
                peak = np.max(np.abs(pcm_f32))
                if peak > 0:
                    pcm_f32 = pcm_f32 * (0.92 / peak)

                pcm_int16 = np.clip(pcm_f32 * 32768.0, -32768, 32767).astype(np.int16)
                pcm_bytes  = pcm_int16.tobytes()

                # Send in small chunks so cancel latency ≤ 60 ms
                for i in range(0, len(pcm_bytes), BYTES_PER_SEND):
                    if self._cancel_event.is_set():
                        return False
                    await self._send_audio(pcm_bytes[i : i + BYTES_PER_SEND])
                    await asyncio.sleep(0)   # yield to event loop

        finally:
            gen_thread.join(timeout=2.0)

        return not self._cancel_event.is_set()

    # ------------------------------------------------------------------
    # edge-tts fallback
    # ------------------------------------------------------------------

    async def _synthesize_edge_tts(self, text: str) -> bool:
        try:
            import io, av, edge_tts  # type: ignore
        except ImportError:
            logger.warning("edge-tts / av not installed — silence",
                           extra={"session_id": self.session_id})
            return await self._synthesize_silence(text)

        try:
            communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
            audio_bytes = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_bytes += chunk["data"]

            if not audio_bytes or self._cancel_event.is_set():
                return not self._cancel_event.is_set()

            buf = io.BytesIO(audio_bytes)
            container = av.open(buf)
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=KOKORO_RATE)
            frames = []
            for frame in container.decode(audio=0):
                for r in resampler.resample(frame):
                    frames.append(np.frombuffer(bytes(r.planes[0]), dtype=np.int16))
            for r in resampler.resample(None):
                frames.append(np.frombuffer(bytes(r.planes[0]), dtype=np.int16))
            container.close()

            if not frames:
                return True

            pcm = np.concatenate(frames)
            nz = np.where(np.abs(pcm) > 160)[0]
            if not len(nz):
                return True
            pcm = pcm[nz[0] : nz[-1] + 1]

            fade = int(KOKORO_RATE * 0.020)
            if len(pcm) > fade * 2:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                f = pcm.astype(np.float32)
                f[:fade] *= ramp; f[-fade:] *= ramp[::-1]
                pcm = np.clip(f, -32768, 32767).astype(np.int16)

            pcm_bytes = pcm.tobytes()
            bpc = int(KOKORO_RATE * 0.060) * 2
            for i in range(0, len(pcm_bytes), bpc):
                if self._cancel_event.is_set():
                    return False
                await self._send_audio(pcm_bytes[i : i + bpc])
                await asyncio.sleep(0)

            return True
        except Exception as exc:
            logger.error("edge-tts error: %s", exc, extra={"session_id": self.session_id})
            return await self._synthesize_silence(text)

    # ------------------------------------------------------------------
    # Silence fallback
    # ------------------------------------------------------------------

    async def _synthesize_silence(self, text: str) -> bool:
        import struct
        words = text.split()
        spc = config.riva.tts_sample_rate_hz // 5
        silence = struct.pack(f"<{spc}h", *([0] * spc))
        for _ in words:
            if self._cancel_event.is_set():
                return False
            await self._send_audio(silence)
            await asyncio.sleep(0.2)
        return True


# ---------------------------------------------------------------------------
# TTSOrchestrator
# ---------------------------------------------------------------------------

class TTSOrchestrator:
    """
    Drains a queue of text fragments and synthesizes them in order.
    Checks cancel_event between every fragment.
    """

    def __init__(
        self,
        session_id: str,
        tts_handler: RivaTTSHandler,
        cancel_event: asyncio.Event,
    ) -> None:
        self.session_id   = session_id
        self._tts         = tts_handler
        self._cancel_event = cancel_event
        self._fragment_queue: asyncio.Queue = asyncio.Queue()
        self._active = False

    @property
    def fragment_queue(self) -> asyncio.Queue:
        return self._fragment_queue

    async def run(self) -> None:
        self._active = True
        logger.debug("TTSOrchestrator started", extra={"session_id": self.session_id})

        while True:
            if self._cancel_event.is_set():
                # Drain stale fragments so they don't play after resume
                while not self._fragment_queue.empty():
                    try:
                        self._fragment_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                break

            try:
                fragment = await asyncio.wait_for(self._fragment_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if fragment is None:
                break

            if self._cancel_event.is_set():
                break

            completed = await self._tts.synthesize_and_stream(fragment)
            if not completed:
                while not self._fragment_queue.empty():
                    try:
                        self._fragment_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                break

        self._active = False
        logger.debug("TTSOrchestrator stopped", extra={"session_id": self.session_id})

    def is_active(self) -> bool:
        return self._active
