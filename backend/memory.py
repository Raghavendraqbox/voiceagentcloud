"""
memory.py — Per-session conversation memory module.

Stores the last N turns of dialogue as (role, text) pairs and
formats them into a string suitable for LLM prompt injection.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

from config import config

logger = logging.getLogger(__name__)


@dataclass
class Turn:
    """A single dialogue turn."""
    role: str   # "User" or "Bot"
    text: str


class ConversationMemory:
    """
    Manages the sliding-window conversation history for a single session.

    Keeps at most `max_turns` complete turns (one User + one Bot = one turn).
    Each half-turn (either a user utterance or a bot reply) counts as one entry.
    """

    def __init__(self, session_id: str, max_turns: int = None) -> None:
        """
        Initialize conversation memory.

        Args:
            session_id: Unique identifier for the session (used in logs).
            max_turns: Maximum half-turns to retain. Defaults to config value * 2
                       so that config.memory.max_turns full exchanges are preserved.
        """
        self.session_id = session_id
        # max_turns in config means full exchanges; store 2x half-turns
        capacity = (max_turns or config.memory.max_turns) * 2
        self._history: Deque[Turn] = deque(maxlen=capacity)
        logger.info(
            "ConversationMemory initialized",
            extra={"session_id": session_id, "capacity": capacity},
        )

    def add_user_turn(self, text: str) -> None:
        """
        Record a user utterance.

        Args:
            text: The transcribed user speech.
        """
        text = text.strip()
        if not text:
            return
        self._history.append(Turn(role="User", text=text))
        logger.debug(
            "User turn added",
            extra={"session_id": self.session_id, "text": text[:60]},
        )

    def add_bot_turn(self, text: str) -> None:
        """
        Record a bot response.

        Args:
            text: The bot's synthesized reply text.
        """
        text = text.strip()
        if not text:
            return
        self._history.append(Turn(role="Bot", text=text))
        logger.debug(
            "Bot turn added",
            extra={"session_id": self.session_id, "text": text[:60]},
        )

    def format_history(self) -> str:
        """
        Return the conversation history as a newline-delimited string.

        Returns:
            Multi-line string with alternating 'User: ...' and 'Bot: ...' lines,
            or an empty string if there is no history yet.

        Example:
            "User: Hello, I need help.\\nBot: Sure, how can I assist?"
        """
        if not self._history:
            return ""
        return "\n".join(f"{turn.role}: {turn.text}" for turn in self._history)

    def get_turns(self) -> List[Tuple[str, str]]:
        """
        Return the raw turn list as (role, text) tuples.

        Returns:
            List of (role, text) pairs in chronological order.
        """
        return [(t.role, t.text) for t in self._history]

    def clear(self) -> None:
        """Wipe all history for this session."""
        self._history.clear()
        logger.info(
            "ConversationMemory cleared",
            extra={"session_id": self.session_id},
        )

    @property
    def turn_count(self) -> int:
        """Number of half-turns currently stored."""
        return len(self._history)

    def __repr__(self) -> str:
        return (
            f"ConversationMemory(session_id={self.session_id!r}, "
            f"turns={self.turn_count})"
        )
