"""
config.py — Centralized configuration for the Voice AI Agent system.

All tunable parameters are defined here. Override via environment variables.
"""

import os
from dataclasses import dataclass, field


@dataclass
class RivaConfig:
    """NVIDIA Riva gRPC service configuration."""
    server_url: str = os.getenv("RIVA_SERVER_URL", "localhost:50051")
    use_ssl: bool = os.getenv("RIVA_USE_SSL", "false").lower() == "true"
    ssl_cert: str = os.getenv("RIVA_SSL_CERT", "")

    # ASR settings
    asr_language_code: str = os.getenv("RIVA_ASR_LANGUAGE", "en-US")
    asr_sample_rate_hz: int = int(os.getenv("RIVA_ASR_SAMPLE_RATE", "16000"))
    asr_encoding: str = "LINEAR_PCM"  # PCM 16-bit signed little-endian
    asr_max_alternatives: int = 1
    asr_interim_results: bool = True
    asr_profanity_filter: bool = False
    asr_automatic_punctuation: bool = True
    asr_word_time_offsets: bool = False

    # TTS settings
    tts_voice_name: str = os.getenv("RIVA_TTS_VOICE", "English-US.Female-1")
    tts_language_code: str = os.getenv("RIVA_TTS_LANGUAGE", "en-US")
    tts_sample_rate_hz: int = int(os.getenv("RIVA_TTS_SAMPLE_RATE", "22050"))
    tts_encoding: str = "LINEAR_PCM"


@dataclass
class OllamaConfig:
    """Ollama LLM service configuration."""
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "llama3")
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("OLLAMA_TOP_P", "0.9"))
    max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "150"))
    # Sentence-boundary tokens that flush a TTS segment
    sentence_delimiters: tuple = (".", "!", "?", ",")


@dataclass
class RAGConfig:
    """FAISS RAG configuration."""
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    docs_directory: str = os.getenv("RAG_DOCS_DIR", "./docs")
    index_path: str = os.getenv("RAG_INDEX_PATH", "./faiss_index")
    top_k: int = int(os.getenv("RAG_TOP_K", "3"))
    similarity_threshold: float = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5"))
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "300"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))


@dataclass
class MemoryConfig:
    """Conversation memory configuration."""
    max_turns: int = int(os.getenv("MEMORY_MAX_TURNS", "8"))


@dataclass
class AudioConfig:
    """Audio pipeline configuration."""
    # PCM chunk size sent from frontend every 100ms at 16kHz mono 16-bit
    # 16000 samples/sec * 0.1 sec * 2 bytes = 3200 bytes
    input_chunk_bytes: int = 3200
    input_sample_rate: int = 16000
    input_channels: int = 1
    input_bit_depth: int = 16

    # VAD interrupt threshold — RMS energy (applied in frontend)
    vad_rms_threshold: float = float(os.getenv("VAD_RMS_THRESHOLD", "0.01"))

    # Frontend pre-buffer before starting playback (ms)
    playback_prebuffer_ms: int = 200


@dataclass
class ServerConfig:
    """FastAPI server configuration."""
    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "info")
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class AppConfig:
    """Root application config — single source of truth."""
    riva: RivaConfig = field(default_factory=RivaConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    # System persona injected into every LLM prompt
    system_prompt: str = (
        "You are Wesaal, a friendly and professional customer service voice agent for Etisalat Afghanistan. "
        "You are on a live phone call. Speak naturally and conversationally in English only. "
        "Keep every response to 1-2 short sentences maximum — this is a voice call, not a chat. "
        "Use warm, natural phrases like 'Sure thing', 'Got it', 'Let me check that for you', 'Of course'. "
        "Only answer what the customer actually asked. Do NOT volunteer product details, prices, or "
        "bundle names unless the customer specifically asks about them. "
        "Never use bullet points, markdown, asterisks, lists, or emojis. "
        "Never make up prices, plan names, or account details. "
        "If you do not know something, say 'Let me connect you to a specialist for that'."
    )

    # Claude model used when Ollama is unavailable
    claude_model: str = "claude-haiku-4-5-20251001"


# Module-level singleton — import this everywhere
config = AppConfig()
