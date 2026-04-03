"""
llm.py — Ollama streaming LLM client for the Voice AI agent.

Builds prompts from conversation memory + RAG context and streams
token-by-token responses.  Sentence fragments are yielded to the caller
so that TTS can begin speaking after the first complete sentence rather
than waiting for the full response.
"""

import asyncio
import logging
import re
from typing import AsyncIterator, Optional

import httpx

from config import config
from memory import ConversationMemory
from rag import RAGRetriever

logger = logging.getLogger(__name__)

# Regex that detects sentence-boundary punctuation followed by whitespace or EOS.
_SENTENCE_BOUNDARY = re.compile(r"([.!?,])(?:\s|$)")


class OllamaClient:
    """
    Async streaming client for the Ollama API.

    Each call to `stream_response` is fully non-blocking and yields
    sentence fragments as soon as they are complete, enabling the TTS
    pipeline to start speaking after the first ~15–30 tokens.
    """

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
    ) -> None:
        """
        Initialize the Ollama client.

        Args:
            retriever: Optional RAGRetriever for context injection.
                       Pass None to disable RAG (e.g., in unit tests).
        """
        self._retriever = retriever
        self._http = httpx.AsyncClient(
            base_url=config.ollama.base_url,
            timeout=httpx.Timeout(30.0, connect=5.0),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        user_query: str,
        memory: ConversationMemory,
        session_id: str = "unknown",
    ) -> AsyncIterator[str]:
        """
        Stream sentence fragments for the given user query.

        The generator yields one *complete sentence fragment* at a time
        (terminated by `.`, `!`, `?`, or `,`).  The final fragment is
        yielded even if it does not end with punctuation.

        Args:
            user_query:  The latest user utterance.
            memory:      Session conversation memory (read-only here).
            session_id:  Used for structured log context.

        Yields:
            Sentence fragments suitable for direct TTS synthesis.
        """
        prompt = self._build_prompt(user_query, memory)
        logger.debug(
            "LLM prompt built",
            extra={"session_id": session_id, "prompt_len": len(prompt)},
        )

        async for fragment in self._stream_fragments(prompt, session_id):
            yield fragment

    async def close(self) -> None:
        """Cleanly close the underlying HTTP client."""
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, user_query: str, memory: ConversationMemory) -> str:
        """
        Assemble the full prompt string from system persona, RAG context,
        conversation history, and the current user query.

        Args:
            user_query: Current user utterance.
            memory:     Session memory object.

        Returns:
            Complete prompt string for Ollama.
        """
        parts: list[str] = [config.system_prompt]

        # RAG context block (may be empty if no relevant docs)
        if self._retriever is not None:
            rag_context = self._retriever.format_context(user_query)
            if rag_context:
                parts.append(rag_context)

        # Conversation history
        history = memory.format_history()
        if history:
            parts.append(f"Conversation so far:\n{history}")

        # Current turn — the "Bot:" suffix primes the model to complete it
        parts.append(f"User: {user_query}\nBot:")

        return "\n\n".join(parts)

    async def _stream_fragments(
        self, prompt: str, session_id: str
    ) -> AsyncIterator[str]:
        """
        Open a streaming HTTP request to Ollama and yield sentence fragments.

        Args:
            prompt:     The assembled prompt string.
            session_id: Used in log context.

        Yields:
            Sentence fragments (strings).
        """
        payload = {
            "model": config.ollama.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.ollama.temperature,
                "top_p": config.ollama.top_p,
                "num_predict": config.ollama.max_tokens,
            },
        }

        buffer = ""
        full_response = ""

        try:
            async with self._http.stream(
                "POST", "/api/generate", json=payload
            ) as response:
                response.raise_for_status()

                async for raw_line in response.aiter_lines():
                    if not raw_line:
                        continue

                    import json as _json
                    try:
                        data = _json.loads(raw_line)
                    except _json.JSONDecodeError:
                        continue

                    token = data.get("response", "")
                    done = data.get("done", False)

                    buffer += token
                    full_response += token

                    # Yield every complete sentence fragment
                    fragment, buffer = self._split_fragment(buffer)
                    if fragment:
                        logger.debug(
                            "LLM fragment ready",
                            extra={"session_id": session_id, "fragment": fragment[:40]},
                        )
                        yield fragment

                    if done:
                        break

                # Flush any remaining text
                if buffer.strip():
                    yield buffer.strip()

        except httpx.HTTPStatusError as exc:
            logger.error(
                "Ollama HTTP error %s: %s",
                exc.response.status_code,
                exc.response.text,
                extra={"session_id": session_id},
            )
            async for fragment in self._stub_response(prompt, session_id):
                yield fragment
        except httpx.RequestError as exc:
            logger.warning(
                "Ollama not reachable (%s) — using stub LLM",
                exc,
                extra={"session_id": session_id},
            )
            async for fragment in self._stub_response(prompt, session_id):
                yield fragment

        logger.info(
            "LLM response complete",
            extra={
                "session_id": session_id,
                "response_len": len(full_response),
                "preview": full_response[:80],
            },
        )

    async def _stub_response(
        self, prompt: str, session_id: str
    ) -> AsyncIterator[str]:
        """Stub LLM used when Ollama is not reachable (for testing).

        Returns only neutral, non-specific replies that don't hallucinate
        product details or pricing.
        """
        import random
        stubs = [
            "Got it, let me look into that for you.",
            "Sure thing! Could you confirm your account details so I can help?",
            "Of course, I can help with that. Could you tell me a bit more?",
            "Absolutely. Let me pull that up for you right now.",
            "I understand. Could you give me just a moment while I check?",
        ]
        response = random.choice(stubs)
        logger.warning(
            "LLM STUB response: %s",
            response,
            extra={"session_id": session_id},
        )
        for word in response.split():
            await asyncio.sleep(0.05)
        yield response

    @staticmethod
    def _split_fragment(buffer: str) -> tuple[str, str]:
        """
        Extract the longest complete sentence fragment from the buffer.

        A fragment ends at `.`, `!`, `?`, or `,` followed by whitespace or EOS.

        Args:
            buffer: Accumulated token text.

        Returns:
            (fragment, remaining_buffer) tuple.
            fragment is empty string if no boundary has been reached.
        """
        match = None
        for m in _SENTENCE_BOUNDARY.finditer(buffer):
            match = m  # take the last boundary found

        if match is None:
            return "", buffer

        split_pos = match.end()
        fragment = buffer[:split_pos].strip()
        remaining = buffer[split_pos:]
        return fragment, remaining
