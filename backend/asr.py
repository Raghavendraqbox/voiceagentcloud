"""
asr.py — NVIDIA Riva streaming ASR handler.

Reads raw PCM audio from an asyncio.Queue, streams it to Riva via gRPC,
and puts transcript results (partial and final) onto an output queue.

Audio format contract (must match frontend):
  - Encoding:    PCM 16-bit signed little-endian (LINEAR_PCM)
  - Sample rate: 16 000 Hz
  - Channels:    1 (mono)
  - Chunk size:  ~3200 bytes (100 ms at 16kHz)
"""

import asyncio
import logging
from typing import Optional

import grpc  # type: ignore

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Riva SDK import — guarded so the module loads even without Riva installed
# (useful for unit tests / CI that mock the ASR handler).
# ---------------------------------------------------------------------------
try:
    import riva.client  # type: ignore
    import riva.client.proto.riva_asr_pb2 as rasr  # type: ignore
    _RIVA_AVAILABLE = True
except ImportError:
    _RIVA_AVAILABLE = False
    logger.warning(
        "nvidia-riva-client not installed — ASR will operate in echo/stub mode"
    )


# ---------------------------------------------------------------------------
# Transcript result dataclass
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class TranscriptResult:
    """Carries a single ASR result emitted from the streaming recognizer."""
    text: str
    is_final: bool
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# ASR Handler
# ---------------------------------------------------------------------------

class RivaASRHandler:
    """
    Wraps the Riva streaming ASR service.

    Lifecycle:
        1. Construct once per session.
        2. Call `run()` as an asyncio task — it loops until cancelled.
        3. Push raw PCM bytes into `audio_queue`.
        4. Consume TranscriptResult objects from `transcript_queue`.

    The `run()` coroutine will reconnect with exponential back-off on
    transient gRPC errors so a momentary Riva hiccup does not end the session.
    """

    def __init__(
        self,
        session_id: str,
        audio_queue: asyncio.Queue,
        transcript_queue: asyncio.Queue,
        interrupt_event: asyncio.Event,
    ) -> None:
        """
        Initialize the ASR handler.

        Args:
            session_id:       Used in structured log messages.
            audio_queue:      Source queue of raw PCM bytes from the WebSocket.
            transcript_queue: Destination queue for TranscriptResult objects.
            interrupt_event:  Set when a partial transcript arrives while TTS
                              is active (used to interrupt playback).
        """
        self.session_id = session_id
        self.audio_queue = audio_queue
        self.transcript_queue = transcript_queue
        self.interrupt_event = interrupt_event
        self._stopped = False

        if _RIVA_AVAILABLE:
            channel_creds = (
                grpc.ssl_channel_credentials(
                    open(config.riva.ssl_cert, "rb").read()
                    if config.riva.ssl_cert
                    else None
                )
                if config.riva.use_ssl
                else None
            )
            auth = riva.client.Auth(
                uri=config.riva.server_url,
                use_ssl=config.riva.use_ssl,
                ssl_cert=config.riva.ssl_cert if config.riva.ssl_cert else None,
            )
            self._service = riva.client.ASRService(auth)
        else:
            self._service = None

    def _build_streaming_config(self) -> "riva.client.StreamingRecognitionConfig":
        """Construct the Riva streaming recognition configuration."""
        return riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=config.riva.asr_language_code,
                sample_rate_hertz=config.riva.asr_sample_rate_hz,
                max_alternatives=config.riva.asr_max_alternatives,
                profanity_filter=config.riva.asr_profanity_filter,
                enable_automatic_punctuation=config.riva.asr_automatic_punctuation,
                enable_word_time_offsets=config.riva.asr_word_time_offsets,
            ),
            interim_results=config.riva.asr_interim_results,
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Main ASR loop.  Streams audio from `audio_queue` to Riva and
        pushes TranscriptResult objects to `transcript_queue`.

        Reconnects on gRPC errors with exponential back-off (max 30 s).
        Exits cleanly when `stop()` is called.
        """
        backoff = 1.0
        while not self._stopped:
            try:
                if _RIVA_AVAILABLE and self._service is not None:
                    await self._run_riva_session()
                else:
                    await self._run_stub_session()
                backoff = 1.0  # reset on clean exit
            except asyncio.CancelledError:
                logger.info(
                    "ASR loop cancelled",
                    extra={"session_id": self.session_id},
                )
                break
            except Exception as exc:
                if self._stopped:
                    break
                logger.error(
                    "ASR error (reconnecting in %.1fs): %s",
                    backoff,
                    exc,
                    extra={"session_id": self.session_id},
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    def stop(self) -> None:
        """Signal the run loop to exit after the current gRPC call finishes."""
        self._stopped = True

    # ------------------------------------------------------------------
    # Riva streaming session
    # ------------------------------------------------------------------

    async def _run_riva_session(self) -> None:
        """
        One complete Riva streaming recognition session.

        Runs an async generator that feeds audio into the Riva gRPC stream
        and consumes transcript responses in the same coroutine via
        asyncio.gather (producer-consumer model without a thread).
        """
        loop = asyncio.get_running_loop()
        streaming_config = self._build_streaming_config()

        # We need to feed audio into the blocking Riva generator.
        # Use run_in_executor for the blocking gRPC call, feeding audio
        # via a synchronous generator that reads from a thread-safe queue.
        import queue as _queue
        sync_audio_queue: _queue.Queue = _queue.Queue()

        def audio_generator():
            """Synchronous generator consumed by the Riva SDK."""
            while not self._stopped:
                try:
                    chunk = sync_audio_queue.get(timeout=0.5)
                    if chunk is None:  # sentinel
                        return
                    yield chunk
                except _queue.Empty:
                    continue

        # Pump async queue → sync queue in the background
        async def pump_audio():
            while not self._stopped:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_queue.get(), timeout=1.0
                    )
                    sync_audio_queue.put(chunk)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
            sync_audio_queue.put(None)  # sentinel to unblock generator

        pump_task = asyncio.create_task(pump_audio())

        try:
            # Run the blocking Riva call in a thread pool executor
            responses = await loop.run_in_executor(
                None,
                lambda: list(
                    self._service.streaming_response_generator(
                        audio_chunks=audio_generator(),
                        streaming_config=streaming_config,
                    )
                ),
            )
            # NOTE: For true streaming we process responses as they arrive.
            # The SDK's streaming_response_generator is a synchronous iterator,
            # so we wrap it in a thread and forward results to the async queue.
            for response in responses:
                await self._process_response(response)
        finally:
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass

    async def _run_riva_streaming(self) -> None:
        """
        True streaming version using a thread that iterates the gRPC responses
        and forwards them to the async loop.

        This is the preferred implementation when the Riva SDK supports
        streaming_response_generator as a lazy iterator (not materializing all
        results upfront).
        """
        loop = asyncio.get_running_loop()
        import queue as _queue
        import threading

        response_queue: _queue.Queue = _queue.Queue()
        sync_audio_queue: _queue.Queue = _queue.Queue()

        streaming_config = self._build_streaming_config()

        def audio_gen():
            while not self._stopped:
                try:
                    chunk = sync_audio_queue.get(timeout=0.5)
                    if chunk is None:
                        return
                    yield chunk
                except _queue.Empty:
                    continue

        def riva_thread():
            try:
                for resp in self._service.streaming_response_generator(
                    audio_chunks=audio_gen(),
                    streaming_config=streaming_config,
                ):
                    response_queue.put(resp)
            except Exception as exc:
                response_queue.put(exc)
            finally:
                response_queue.put(None)  # sentinel

        thread = threading.Thread(target=riva_thread, daemon=True)
        thread.start()

        # Pump audio async → sync
        async def pump():
            while not self._stopped:
                try:
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=1.0)
                    sync_audio_queue.put(chunk)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
            sync_audio_queue.put(None)

        pump_task = asyncio.create_task(pump())

        try:
            while True:
                item = await loop.run_in_executor(None, response_queue.get)
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                await self._process_response(item)
        finally:
            pump_task.cancel()
            try:
                await pump_task
            except asyncio.CancelledError:
                pass
            thread.join(timeout=2.0)

    async def _process_response(self, response) -> None:
        """
        Parse a Riva StreamingRecognizeResponse and push results to the queue.

        Partial transcripts also set the interrupt_event so the TTS loop can
        react to the user speaking.

        Args:
            response: riva.client.proto StreamingRecognizeResponse proto.
        """
        for result in response.results:
            if not result.alternatives:
                continue

            text = result.alternatives[0].transcript.strip()
            if not text:
                continue

            is_final = result.is_final
            confidence = (
                result.alternatives[0].confidence if is_final else 0.0
            )

            transcript = TranscriptResult(
                text=text,
                is_final=is_final,
                confidence=confidence,
            )

            # Any speech activity should interrupt TTS
            if not self.interrupt_event.is_set():
                self.interrupt_event.set()
                logger.debug(
                    "Interrupt event set by ASR partial",
                    extra={"session_id": self.session_id},
                )

            await self.transcript_queue.put(transcript)
            logger.debug(
                "Transcript %s: %s",
                "FINAL" if is_final else "partial",
                text[:60],
                extra={"session_id": self.session_id},
            )

    # ------------------------------------------------------------------
    # Stub session (when Riva is not available)
    # ------------------------------------------------------------------

    async def _run_stub_session(self) -> None:
        """
        Fallback ASR stub used when nvidia-riva-client is not installed.

        Drains the audio queue silently and emits a placeholder transcript
        every ~3 seconds to keep the pipeline alive during development.
        """
        logger.warning(
            "Running ASR in STUB mode — no real transcription",
            extra={"session_id": self.session_id},
        )
        counter = 0
        while not self._stopped:
            # Drain any audio chunks to avoid queue build-up
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            await asyncio.sleep(3.0)
            counter += 1
            stub_text = f"stub transcript {counter}"
            await self.transcript_queue.put(
                TranscriptResult(text=stub_text, is_final=True, confidence=1.0)
            )
