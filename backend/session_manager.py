"""
session_manager.py — Per-session state, interrupt logic, and task lifecycle.

Each WebSocket connection creates one Session.  The Session owns:
  - The three async pipeline tasks (ASR, LLM, TTS)
  - The shared queues and events that connect them
  - The conversation memory
  - Interrupt handling logic

The SessionManager is a process-wide registry that maps session IDs to
Session objects and provides clean teardown on WebSocket disconnect.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, Optional

from config import config
from memory import ConversationMemory
from rag import RAGRetriever
from asr import RivaASRHandler, TranscriptResult
from tts import RivaTTSHandler, TTSOrchestrator
from llm import OllamaClient

logger = logging.getLogger(__name__)

# Type alias
AudioSendCallback = Callable[[bytes], Awaitable[None]]
JsonSendCallback = Callable[[dict], Awaitable[None]]


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """
    All state for a single connected client session.

    Attributes:
        session_id:        UUID string identifying this connection.
        memory:            Sliding-window conversation history.
        audio_queue:       PCM chunks arriving from the WebSocket.
        transcript_queue:  TranscriptResult objects from the ASR loop.
        interrupt_event:   Set when user speech is detected mid-response.
        tts_cancel_event:  Watched by the TTS orchestrator to abort playback.
        asr_task:          asyncio.Task running the ASR loop.
        llm_tts_task:      asyncio.Task running the LLM+TTS pipeline.
    """
    session_id: str
    memory: ConversationMemory
    audio_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    transcript_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)
    tts_cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    asr_task: Optional[asyncio.Task] = field(default=None, init=False)
    llm_tts_task: Optional[asyncio.Task] = field(default=None, init=False)

    # References to handler objects (set by SessionManager.create_session)
    asr_handler: Optional[RivaASRHandler] = field(default=None, init=False)
    tts_handler: Optional[RivaTTSHandler] = field(default=None, init=False)
    tts_orchestrator: Optional[TTSOrchestrator] = field(default=None, init=False)
    llm_client: Optional[OllamaClient] = field(default=None, init=False)

    def cancel_tts(self) -> None:
        """Signal TTS to stop and clear the interrupt event for next turn."""
        self.tts_cancel_event.set()
        logger.info(
            "TTS cancel signalled",
            extra={"session_id": self.session_id},
        )

    def reset_for_new_turn(self) -> None:
        """Reset interrupt/cancel events after a new utterance starts."""
        self.interrupt_event.clear()
        self.tts_cancel_event.clear()
        logger.debug(
            "Session events reset for new turn",
            extra={"session_id": self.session_id},
        )

    async def cleanup(self) -> None:
        """Cancel all running tasks and close the LLM HTTP client."""
        logger.info(
            "Cleaning up session",
            extra={"session_id": self.session_id},
        )
        for task in (self.asr_task, self.llm_tts_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self.asr_handler is not None:
            self.asr_handler.stop()

        if self.llm_client is not None:
            await self.llm_client.close()

        logger.info(
            "Session cleanup complete",
            extra={"session_id": self.session_id},
        )


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Process-wide registry of active sessions.

    Also owns the shared RAGRetriever (built once at startup).
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._retriever: Optional[RAGRetriever] = None

    def initialize_rag(self) -> None:
        """
        Build or reload the FAISS index.  Call once at application startup.
        This is synchronous and may take a few seconds on first run.
        """
        self._retriever = RAGRetriever()
        self._retriever.initialize()
        logger.info("RAG retriever initialized")

    def create_session(
        self,
        send_audio_cb: AudioSendCallback,
        send_json_cb: JsonSendCallback,
    ) -> Session:
        """
        Allocate a new Session, wire up all handlers, and start async loops.

        Args:
            send_audio_cb: Coroutine that sends raw PCM bytes to the client.
            send_json_cb:  Coroutine that sends a JSON dict to the client.

        Returns:
            The newly created Session object.
        """
        session_id = str(uuid.uuid4())
        memory = ConversationMemory(session_id=session_id)

        session = Session(
            session_id=session_id,
            memory=memory,
        )

        # Wire up ASR
        session.asr_handler = RivaASRHandler(
            session_id=session_id,
            audio_queue=session.audio_queue,
            transcript_queue=session.transcript_queue,
            interrupt_event=session.interrupt_event,
        )

        # Wire up TTS
        session.tts_handler = RivaTTSHandler(
            session_id=session_id,
            send_audio_cb=send_audio_cb,
            cancel_event=session.tts_cancel_event,
        )

        session.tts_orchestrator = TTSOrchestrator(
            session_id=session_id,
            tts_handler=session.tts_handler,
            cancel_event=session.tts_cancel_event,
        )

        # Wire up LLM
        session.llm_client = OllamaClient(retriever=self._retriever)

        # Start the two background loops
        session.asr_task = asyncio.create_task(
            session.asr_handler.run(),
            name=f"asr-{session_id}",
        )

        session.llm_tts_task = asyncio.create_task(
            self._llm_tts_loop(session, send_json_cb),
            name=f"llm_tts-{session_id}",
        )

        self._sessions[session_id] = session
        logger.info(
            "Session created",
            extra={"session_id": session_id},
        )
        return session

    async def destroy_session(self, session_id: str) -> None:
        """
        Tear down a session and remove it from the registry.

        Args:
            session_id: The session to destroy.
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        await session.cleanup()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Return the session or None if it does not exist."""
        return self._sessions.get(session_id)

    # ------------------------------------------------------------------
    # LLM + TTS co-routine (runs as background task per session)
    # ------------------------------------------------------------------

    async def _llm_tts_loop(
        self,
        session: Session,
        send_json_cb: JsonSendCallback,
    ) -> None:
        """
        Waits for final transcripts from ASR, runs the LLM, streams TTS.

        This is the "brain" of the pipeline:
        1. Block on `transcript_queue` for a final transcript.
        2. Reset interrupt/cancel events.
        3. Notify frontend that LLM is thinking.
        4. Stream LLM tokens → sentence fragments → TTS fragment queue.
        5. When LLM is done, sentinel TTS orchestrator, wait for it to finish.
        6. Record full bot response in memory.
        7. Notify frontend that TTS ended.
        8. Go back to step 1.

        Args:
            session:      The current session.
            send_json_cb: Sends control messages (tts_start, tts_end, etc.)
                          to the frontend.
        """
        logger.info(
            "LLM/TTS loop started",
            extra={"session_id": session.session_id},
        )

        while True:
            # ----------------------------------------------------------
            # 1. Wait for a final transcript
            # ----------------------------------------------------------
            try:
                transcript: TranscriptResult = await session.transcript_queue.get()
            except asyncio.CancelledError:
                break

            if not transcript.is_final:
                # Partial — already handled by ASR (interrupt event set)
                continue

            user_text = transcript.text.strip()
            if not user_text:
                continue

            logger.info(
                "Processing transcript: %s",
                user_text[:80],
                extra={"session_id": session.session_id},
            )

            # ----------------------------------------------------------
            # 2. Stop any ongoing TTS
            # ----------------------------------------------------------
            session.cancel_tts()
            # Give the orchestrator a moment to drain
            await asyncio.sleep(0.05)
            session.reset_for_new_turn()

            # ----------------------------------------------------------
            # 3. Record user turn
            # ----------------------------------------------------------
            session.memory.add_user_turn(user_text)

            # Notify frontend: user transcript confirmed
            await send_json_cb(
                {"type": "transcript_final", "text": user_text}
            )

            # ----------------------------------------------------------
            # 4. Stream LLM → TTS fragment queue
            # ----------------------------------------------------------
            await send_json_cb({"type": "tts_start"})

            # Create a fresh orchestrator for this response turn
            session.tts_orchestrator = TTSOrchestrator(
                session_id=session.session_id,
                tts_handler=session.tts_handler,
                cancel_event=session.tts_cancel_event,
            )

            orchestrator_task = asyncio.create_task(
                session.tts_orchestrator.run(),
                name=f"tts-orch-{session.session_id}",
            )

            full_bot_response = ""

            try:
                async for fragment in session.llm_client.stream_response(
                    user_query=user_text,
                    memory=session.memory,
                    session_id=session.session_id,
                ):
                    if session.tts_cancel_event.is_set():
                        # User interrupted — stop feeding TTS
                        break

                    full_bot_response += fragment + " "

                    # Notify frontend of streaming bot text
                    await send_json_cb(
                        {"type": "bot_text_fragment", "text": fragment}
                    )

                    # Push fragment to TTS orchestrator
                    await session.tts_orchestrator.fragment_queue.put(fragment)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(
                    "LLM error: %s",
                    exc,
                    extra={"session_id": session.session_id},
                )
                await send_json_cb(
                    {"type": "error", "message": "LLM processing failed"}
                )

            # Sentinel to tell TTS orchestrator the stream is done
            await session.tts_orchestrator.fragment_queue.put(None)

            # Wait for TTS to finish (or be cancelled)
            try:
                await asyncio.wait_for(orchestrator_task, timeout=30.0)
            except asyncio.TimeoutError:
                orchestrator_task.cancel()
            except asyncio.CancelledError:
                orchestrator_task.cancel()
                break

            # ----------------------------------------------------------
            # 5. Record bot turn in memory
            # ----------------------------------------------------------
            bot_text = full_bot_response.strip()
            if bot_text:
                session.memory.add_bot_turn(bot_text)

            await send_json_cb({"type": "tts_end"})

            logger.info(
                "Turn complete",
                extra={
                    "session_id": session.session_id,
                    "bot_preview": bot_text[:60],
                },
            )

        logger.info(
            "LLM/TTS loop exiting",
            extra={"session_id": session.session_id},
        )


# ---------------------------------------------------------------------------
# Module-level singleton — imported by main.py
# ---------------------------------------------------------------------------
session_manager = SessionManager()
