"""
tts.py — NVIDIA Riva streaming TTS handler.

Consumes text fragments from a queue, synthesizes them via Riva,
and streams raw PCM audio bytes back to the WebSocket caller via a callback.

Every audio chunk delivery checks the `cancel_event` so that an
in-flight TTS response can be cut off within milliseconds of the user
starting to speak.
"""

import asyncio
import logging
import queue as _sync_queue
import threading
from typing import Callable, Awaitable

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
    logger.warning(
        "nvidia-riva-client not installed — TTS will operate in stub mode"
    )


# ---------------------------------------------------------------------------
# Type alias for the audio send callback
# ---------------------------------------------------------------------------
# The callback receives raw PCM bytes and returns an awaitable.
AudioSendCallback = Callable[[bytes], Awaitable[None]]


# ---------------------------------------------------------------------------
# TTS Handler
# ---------------------------------------------------------------------------

class RivaTTSHandler:
    """
    Wraps the Riva streaming TTS service for a single session.

    Usage pattern:
        handler = RivaTTSHandler(session_id, send_audio_cb, cancel_event)
        await handler.synthesize_and_stream("Alright, let me check that.")

    The `send_audio_cb` is called with each PCM chunk as it arrives from Riva.
    Playback is cancelled immediately when `cancel_event` is set.
    """

    def __init__(
        self,
        session_id: str,
        send_audio_cb: AudioSendCallback,
        cancel_event: asyncio.Event,
    ) -> None:
        """
        Initialize the TTS handler.

        Args:
            session_id:    Used in structured log messages.
            send_audio_cb: Async callable that sends a PCM bytes chunk to the
                           WebSocket client.
            cancel_event:  asyncio.Event that — when set — causes the current
                           synthesis to abort between chunks.
        """
        self.session_id = session_id
        self._send_audio = send_audio_cb
        self._cancel_event = cancel_event

        if _RIVA_AVAILABLE:
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
        """
        Synthesize `text` and stream the resulting PCM audio to the client.

        Streams in real time — each audio chunk is forwarded as soon as it
        arrives from Riva without waiting for the full synthesis.

        Args:
            text: The sentence fragment to synthesize.

        Returns:
            True  — synthesis completed normally.
            False — synthesis was cancelled mid-stream.
        """
        if not text.strip():
            return True

        logger.info(
            "TTS synthesize: %s",
            text[:60],
            extra={"session_id": self.session_id},
        )

        if _RIVA_AVAILABLE and self._service is not None:
            return await self._synthesize_riva(text)
        else:
            return await self._synthesize_stub(text)

    # ------------------------------------------------------------------
    # Riva synthesis
    # ------------------------------------------------------------------

    async def _synthesize_riva(self, text: str) -> bool:
        """
        Call Riva online synthesis and stream chunks to the client.

        The Riva SDK exposes a synchronous iterator; we run it in a
        ThreadPoolExecutor and bridge results back to the async loop via
        a sync queue so we can check `cancel_event` between each chunk.

        Args:
            text: Text to synthesize.

        Returns:
            True if completed, False if cancelled.
        """
        loop = asyncio.get_running_loop()
        response_queue: _sync_queue.Queue = _sync_queue.Queue()

        def riva_thread():
            """Runs in a thread — iterates the blocking Riva generator."""
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
                response_queue.put(None)  # sentinel

        thread = threading.Thread(target=riva_thread, daemon=True)
        thread.start()

        completed = True
        try:
            while True:
                # Poll for the next chunk without blocking the event loop
                item = await loop.run_in_executor(None, response_queue.get)

                if item is None:  # sentinel — synthesis done
                    break

                if isinstance(item, Exception):
                    logger.error(
                        "Riva TTS error: %s",
                        item,
                        extra={"session_id": self.session_id},
                    )
                    completed = False
                    break

                # Check interrupt BEFORE sending each chunk
                if self._cancel_event.is_set():
                    logger.info(
                        "TTS cancelled mid-stream",
                        extra={"session_id": self.session_id},
                    )
                    completed = False
                    break

                await self._send_audio(item)

        finally:
            thread.join(timeout=2.0)

        return completed

    # ------------------------------------------------------------------
    # Stub synthesis (fallback when Riva is not available)
    # ------------------------------------------------------------------

    async def _synthesize_stub(self, text: str) -> bool:
        """
        Synthesize real speech using edge-tts (Microsoft Edge neural TTS)
        when Riva is unavailable.  Audio is decoded from MP3 to PCM via PyAV
        and resampled to the configured TTS sample rate before streaming.

        Falls back to silence only if edge-tts or av cannot be imported.

        Args:
            text: Text to speak.

        Returns:
            True if completed, False if cancelled.
        """
        try:
            import io
            import av
            import edge_tts
        except ImportError:
            logger.warning(
                "edge-tts / av not installed — TTS produces silence",
                extra={"session_id": self.session_id},
            )
            return await self._synthesize_silence(text)

        try:
            return await self._synthesize_edge_tts(text, edge_tts, av, io)
        except Exception as exc:
            logger.error(
                "edge-tts error (%s) — falling back to silence", exc,
                extra={"session_id": self.session_id},
            )
            return await self._synthesize_silence(text)

    async def _synthesize_edge_tts(self, text: str, edge_tts, av, io) -> bool:
        """
        Stream text through Microsoft Edge TTS and forward clean PCM to the client.

        edge-tts natively outputs 24 kHz mono MP3.  We decode to 24 kHz s16 PCM
        without any resampling (integer ratio = no FIR filter artifacts).

        Noise-reduction steps applied to the decoded PCM:
          1. Trim leading/trailing near-silence to remove MP3 encoder-delay
             priming and decoder flush zero-padding.
          2. Apply a 20 ms linear fade-in and fade-out to prevent clicks at
             playback boundaries.

        The frontend AudioContext must also be at 24 kHz (PLAYBACK_SAMPLE_RATE).
        """
        import numpy as np

        EDGE_TTS_RATE = 24000   # native edge-tts output rate — no resampling needed

        # Collect the MP3 stream from edge-tts
        communicate = edge_tts.Communicate(text, voice="en-US-JennyNeural")
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        if not audio_bytes or self._cancel_event.is_set():
            return not self._cancel_event.is_set()

        # Decode MP3 → 24 kHz mono s16 PCM using PyAV (no sample rate conversion)
        buf = io.BytesIO(audio_bytes)
        container = av.open(buf)
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=EDGE_TTS_RATE,
        )
        pcm_frames: list = []
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                pcm_frames.append(np.frombuffer(bytes(resampled.planes[0]), dtype=np.int16))
        for resampled in resampler.resample(None):
            pcm_frames.append(np.frombuffer(bytes(resampled.planes[0]), dtype=np.int16))
        container.close()

        if not pcm_frames:
            return True

        pcm = np.concatenate(pcm_frames)

        # 1. Trim leading / trailing near-silence (MP3 encoder-delay priming)
        TRIM_THRESHOLD = 160  # ~0.5 % of Int16 full scale
        nonzero = np.where(np.abs(pcm) > TRIM_THRESHOLD)[0]
        if len(nonzero) == 0:
            return True
        pcm = pcm[nonzero[0] : nonzero[-1] + 1]

        # 2. 20 ms linear fade-in / fade-out to prevent click at boundaries
        fade_samples = int(EDGE_TTS_RATE * 0.020)
        if len(pcm) > fade_samples * 2:
            ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            pcm_f = pcm.astype(np.float32)
            pcm_f[:fade_samples] *= ramp
            pcm_f[-fade_samples:] *= ramp[::-1]
            pcm = np.clip(pcm_f, -32768, 32767).astype(np.int16)

        pcm_bytes = pcm.tobytes()

        # Stream PCM in 200 ms chunks to the browser
        bytes_per_chunk = EDGE_TTS_RATE * 2 // 5  # 200 ms of 16-bit mono
        for i in range(0, len(pcm_bytes), bytes_per_chunk):
            if self._cancel_event.is_set():
                logger.info(
                    "TTS edge-tts cancelled mid-stream",
                    extra={"session_id": self.session_id},
                )
                return False
            await self._send_audio(pcm_bytes[i : i + bytes_per_chunk])
            await asyncio.sleep(0)

        return True

    async def _synthesize_silence(self, text: str) -> bool:
        """Last-resort fallback: send silence so the pipeline keeps moving."""
        import struct

        words = text.split()
        samples_per_word = config.riva.tts_sample_rate_hz // 5
        silence_chunk = struct.pack(f"<{samples_per_word}h", *([0] * samples_per_word))

        for _ in words:
            if self._cancel_event.is_set():
                return False
            await self._send_audio(silence_chunk)
            await asyncio.sleep(0.2)

        return True


# ---------------------------------------------------------------------------
# TTS Orchestrator — processes a queue of text fragments
# ---------------------------------------------------------------------------

class TTSOrchestrator:
    """
    Drains a queue of text fragments and synthesizes them in order,
    checking the cancel event between each fragment.

    This sits between the LLM streaming client and the TTS handler,
    allowing the LLM to keep producing tokens while TTS works on the
    first sentence fragment.
    """

    def __init__(
        self,
        session_id: str,
        tts_handler: RivaTTSHandler,
        cancel_event: asyncio.Event,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            session_id:   Used in log messages.
            tts_handler:  RivaTTSHandler instance for this session.
            cancel_event: Shared cancel/interrupt event.
        """
        self.session_id = session_id
        self._tts = tts_handler
        self._cancel_event = cancel_event
        self._fragment_queue: asyncio.Queue = asyncio.Queue()
        self._active = False

    @property
    def fragment_queue(self) -> asyncio.Queue:
        """The queue into which the LLM client pushes text fragments."""
        return self._fragment_queue

    async def run(self) -> None:
        """
        Continuously drain the fragment queue and synthesize each fragment.

        Exits when a sentinel `None` is dequeued or `cancel_event` is set.
        """
        self._active = True
        logger.debug(
            "TTSOrchestrator started",
            extra={"session_id": self.session_id},
        )

        while True:
            # Check for cancellation between fragments
            if self._cancel_event.is_set():
                logger.info(
                    "TTSOrchestrator: cancel event — draining queue",
                    extra={"session_id": self.session_id},
                )
                # Drain the queue so stale fragments don't play after resume
                while not self._fragment_queue.empty():
                    try:
                        self._fragment_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                break

            try:
                fragment = await asyncio.wait_for(
                    self._fragment_queue.get(), timeout=0.2
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if fragment is None:  # sentinel — LLM finished
                break

            if self._cancel_event.is_set():
                break

            completed = await self._tts.synthesize_and_stream(fragment)
            if not completed:
                # Drain remaining fragments — they belong to a cancelled response
                while not self._fragment_queue.empty():
                    try:
                        self._fragment_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                break

        self._active = False
        logger.debug(
            "TTSOrchestrator stopped",
            extra={"session_id": self.session_id},
        )

    def is_active(self) -> bool:
        """Return True while the orchestrator loop is running."""
        return self._active
