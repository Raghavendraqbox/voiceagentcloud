"""
main.py — FastAPI application with WebSocket endpoint for the Voice AI agent.

WebSocket message protocol:
  CLIENT → SERVER (binary):
    Raw PCM audio bytes (16-bit, mono, 16kHz, ~3200 bytes per 100ms chunk)

  CLIENT → SERVER (JSON text):
    {"type": "interrupt"}                — user VAD detected speech mid-response
    {"type": "ping"}                     — keepalive

  SERVER → CLIENT (binary):
    Raw PCM audio bytes (synthesized TTS, 22050 Hz, 16-bit, mono)

  SERVER → CLIENT (JSON text):
    {"type": "tts_start"}               — first TTS chunk about to arrive
    {"type": "tts_end"}                 — TTS fully delivered (or cancelled)
    {"type": "tts_stopped"}             — TTS interrupted mid-stream
    {"type": "transcript_partial", "text": "..."}  — ASR interim result
    {"type": "transcript_final", "text": "..."}    — ASR final result
    {"type": "bot_text_fragment", "text": "..."}   — streamed LLM token group
    {"type": "error", "message": "..."}            — pipeline error
    {"type": "pong"}                               — keepalive reply
    {"type": "session_ready"}                      — handshake complete
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import config
from session_manager import session_manager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, config.server.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources at startup; clean up at shutdown."""
    logger.info("Starting Voice AI Agent server…")
    session_manager.initialize_rag()
    logger.info("Server ready.")
    yield
    logger.info("Shutting down…")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Voice AI Agent",
    description="Real-time full-duplex conversational voice agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend static files
import os
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to the frontend HTML."""
    return HTMLResponse(
        content='<meta http-equiv="refresh" content="0; url=/static/index.html">',
        status_code=200,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model": config.ollama.model}


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket handler.

    Accepts the connection, creates a session, forwards audio to the
    session's ASR queue, and dispatches JSON control messages.
    """
    await websocket.accept()
    logger.info("WebSocket connected from %s", websocket.client)

    # ------------------------------------------------------------------
    # Callbacks injected into the session
    # ------------------------------------------------------------------

    async def send_audio(pcm_bytes: bytes) -> None:
        """Forward a PCM chunk to the connected client."""
        try:
            await websocket.send_bytes(pcm_bytes)
        except Exception as exc:
            logger.debug("send_audio error (client likely disconnected): %s", exc)

    async def send_json(payload: dict) -> None:
        """Forward a JSON control message to the connected client."""
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception as exc:
            logger.debug("send_json error (client likely disconnected): %s", exc)

    # ------------------------------------------------------------------
    # Create session
    # ------------------------------------------------------------------
    session = session_manager.create_session(
        send_audio_cb=send_audio,
        send_json_cb=send_json,
    )

    await send_json({"type": "session_ready", "session_id": session.session_id})
    logger.info("Session %s is ready", session.session_id)

    # ------------------------------------------------------------------
    # Message receive loop
    # ------------------------------------------------------------------
    try:
        while True:
            message = await websocket.receive()

            # Binary frame — raw PCM audio from microphone
            if "bytes" in message and message["bytes"]:
                pcm_chunk: bytes = message["bytes"]

                # Validate basic size constraint (must be even for 16-bit PCM)
                if len(pcm_chunk) % 2 != 0:
                    await send_json(
                        {"type": "error", "message": "PCM chunk has odd byte count"}
                    )
                    continue

                await session.audio_queue.put(pcm_chunk)

            # Text frame — JSON control message
            elif "text" in message and message["text"]:
                try:
                    msg = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning(
                        "Malformed JSON from client: %s",
                        message["text"][:80],
                        extra={"session_id": session.session_id},
                    )
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "interrupt":
                    # User started speaking — cancel TTS immediately
                    logger.info(
                        "Interrupt received from client",
                        extra={"session_id": session.session_id},
                    )
                    session.cancel_tts()
                    await send_json({"type": "tts_stopped"})

                elif msg_type == "transcript_partial":
                    # Frontend VAD sent a partial — forward to ASR queue
                    # as an event so the interrupt logic is triggered
                    session.interrupt_event.set()

                elif msg_type == "ping":
                    await send_json({"type": "pong"})

                else:
                    logger.debug(
                        "Unknown message type: %s",
                        msg_type,
                        extra={"session_id": session.session_id},
                    )

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected (session %s)",
            session.session_id,
        )
    except Exception as exc:
        logger.error(
            "WebSocket error for session %s: %s",
            session.session_id,
            exc,
        )
    finally:
        await session_manager.destroy_session(session.session_id)
        logger.info("Session %s destroyed", session.session_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        workers=1,  # Must be 1 — asyncio state is not fork-safe
    )
