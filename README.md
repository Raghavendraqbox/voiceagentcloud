# VoiceAgentCloud — Qobox Assistant

A **near full-duplex conversational voice AI agent** that simulates a live phone call for **Qobox (Quality Outside The Box)**, an Indian software QA company.  The user speaks freely; the bot responds in real time with no push-to-talk.

## Stack

| Layer | Primary | Fallback |
|---|---|---|
| ASR | NVIDIA Riva via NVCF cloud (Parakeet-CTC-1.1B) | Whisper small (GPU/CPU) |
| LLM | Ollama llama3.2:1b (local) | NVIDIA NIM Nemotron-70B → Claude API → stub |
| TTS | NVIDIA Riva via NVCF cloud (FastPitch-HiFiGAN) | Kokoro GPU → edge-tts → silence |
| RAG | FAISS + sentence-transformers | — |
| Transport | FastAPI + WebSocket | — |
| Frontend | Browser AudioWorklet (no build step) | — |

> **No NVIDIA API key?**  Set `ANTHROPIC_API_KEY` and the server falls back to Whisper ASR + Kokoro TTS + Claude Haiku.
> **No GPU?**  Whisper runs on CPU (INT8) and Kokoro falls back to edge-tts.

---

## Architecture

```
Browser (Chrome / Firefox)
    │
    │  binary frames: PCM 16-bit mono 16kHz (microphone chunks)
    │  JSON text:     {"type":"interrupt"} | {"type":"ping"}
    │
    ▼
FastAPI WebSocket /ws  ──── GET /  ──── GET /health
    │
    ├─ [Loop 1] ASR  (asr.py)
    │      Riva NVCF gRPC  →  TranscriptResult queue
    │      Fallback: Whisper small on GPU
    │
    ├─ [Loop 2] LLM  (llm.py)
    │      Reads final transcripts  →  streams sentence fragments
    │      Injects: RAG context (Qobox KB) + conversation memory
    │      Fallback chain: Ollama → NIM → Claude → stub
    │
    └─ [Loop 3] TTS  (tts.py)
           Consumes sentence fragments  →  Riva NVCF gRPC TTS
           Fallback: Kokoro GPU chunk-streaming (cancel-aware)
           →  binary PCM audio back to browser

Browser AudioWorklet chains PCM chunks into seamless playback.
VAD (RMS energy) detects user speech → sends interrupt → TTS stops < 200ms.
```

---

## Repository Structure

```
voiceagentcloud/
├── backend/
│   ├── main.py              # FastAPI app + WebSocket handler + HTTP routes
│   ├── session_manager.py   # Per-session state, 3-loop pipeline, interrupt logic
│   ├── asr.py               # Riva NVCF streaming ASR (Whisper fallback)
│   ├── tts.py               # Riva NVCF streaming TTS (Kokoro GPU fallback)
│   ├── llm.py               # Ollama / NIM / Claude streaming LLM client
│   ├── rag.py               # FAISS vector DB + sentence-transformer embeddings
│   ├── memory.py            # Sliding-window conversation memory (last 8 turns)
│   └── config.py            # All settings, overridable via env vars
├── frontend/
│   └── index.html           # Single-page browser UI (no build step)
├── docs/
│   ├── telecom_products.txt # Additional RAG documents (optional)
│   └── troubleshooting.txt
├── .env.example             # All environment variables with descriptions
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

---

## Quick Start

### Option A — NVIDIA Cloud (recommended, no local GPU services needed)

You need a free [NVIDIA API key](https://build.nvidia.com) and any Python 3.11 environment.

```bash
# 1. Clone
git clone https://github.com/Raghavendraqbox/voiceagentcloud.git
cd voiceagentcloud

# 2. Python environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set your NVIDIA API key
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# (or copy .env.example → .env and fill in NVIDIA_API_KEY)

# 5. Start the server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

# 6. Open browser
# http://localhost:8000
# Click "Start Talking" and speak — Riva ASR + Riva TTS + NIM LLM
```

### Option B — Local GPU only (no NVIDIA cloud key)

Requires a machine with a CUDA GPU (RTX 3090 / A100 / H100).

```bash
# Same steps 1-3 above, then:

# Optional: pull a local LLM via Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b

# Set Anthropic key as LLM fallback (if Ollama is not running)
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxx

# Start server — will use Whisper small (GPU) + Kokoro (GPU) + Ollama/Claude
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Option C — Dev / stub mode (no GPU, no API keys)

```bash
# Same steps 1-3 above, then:
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# ASR: Whisper tiny (CPU)
# TTS: edge-tts (free Microsoft neural TTS, internet required)
# LLM: neutral stub responses (no API needed)
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in the values you need.

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_API_KEY` | _(unset)_ | NVIDIA API key — enables NVCF Riva ASR+TTS and NIM LLM |
| `ANTHROPIC_API_KEY` | _(unset)_ | Claude API key — LLM fallback when Ollama unavailable |
| `RIVA_SERVER_URL` | `localhost:50051` | Local Riva gRPC endpoint (only used when no NVIDIA_API_KEY) |
| `RIVA_TTS_VOICE` | `English-US.Female-1` | Riva TTS voice name |
| `RIVA_ASR_LANGUAGE` | `en-US` | ASR language code |
| `RIVA_TTS_SAMPLE_RATE` | `22050` | TTS output sample rate |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP endpoint |
| `OLLAMA_MODEL` | `llama3.2:1b` | Ollama model name |
| `OLLAMA_MAX_TOKENS` | `150` | Max LLM response tokens |
| `OLLAMA_TEMPERATURE` | `0.7` | LLM sampling temperature |
| `RAG_DOCS_DIR` | `./docs` | Directory of `.txt` knowledge files |
| `RAG_INDEX_PATH` | `./faiss_index` | FAISS index cache path |
| `RAG_TOP_K` | `3` | Number of RAG chunks to inject |
| `RAG_SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity for RAG match |
| `VAD_RMS_THRESHOLD` | `0.018` | Mic RMS energy to trigger interrupt (raise to reduce false triggers) |
| `MEMORY_MAX_TURNS` | `8` | Conversation turns retained in memory |
| `SERVER_PORT` | `8000` | HTTP/WebSocket listen port |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warning` |

---

## Starting with a `.env` file

```bash
# Create .env from the example
cp .env.example .env
# Edit .env with your keys

# Start server — reads .env automatically via python-dotenv
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

Or pass variables inline without a file:

```bash
NVIDIA_API_KEY=nvapi-xxx ANTHROPIC_API_KEY=sk-ant-xxx \
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## HTTPS for Remote Access

Browsers require HTTPS (or `localhost`) for microphone access.

### Self-signed certificate (RunPod / dev)

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
  -days 365 -nodes -subj "/CN=voiceagent"

uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 \
  --ssl-keyfile ../key.pem --ssl-certfile ../cert.pem
```

Open `https://YOUR_SERVER_IP:8000` (accept the browser warning once).

### Nginx reverse proxy (production)

```nginx
server {
    listen 443 ssl;
    server_name voiceagent.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/voiceagent.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voiceagent.yourdomain.com/privkey.pem;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
        proxy_buffering    off;
    }
}
```

### RunPod notes

1. Expose port 8000 in the pod configuration.
2. RunPod assigns `https://<pod-id>-8000.proxy.runpod.net` — HTTPS is already handled; no self-signed cert needed.
3. Set `NVIDIA_API_KEY` in the pod's environment variables tab.

---

## Adding Knowledge Base Documents

1. Add `.txt` files to the `docs/` directory.
2. Delete the cached FAISS index:
   ```bash
   rm -rf backend/faiss_index/
   ```
3. Restart the server — the index rebuilds automatically.

---

## WebSocket Protocol

| Direction | Frame type | Payload |
|---|---|---|
| Client → Server | binary | PCM 16-bit mono 16kHz (~3200 bytes / 100ms) |
| Client → Server | JSON | `{"type":"interrupt"}` — user started speaking |
| Client → Server | JSON | `{"type":"ping"}` — keepalive |
| Server → Client | binary | PCM 16-bit mono 22050Hz — TTS audio |
| Server → Client | JSON | `{"type":"session_ready","session_id":"..."}` |
| Server → Client | JSON | `{"type":"transcript_partial","text":"..."}` |
| Server → Client | JSON | `{"type":"transcript_final","text":"..."}` |
| Server → Client | JSON | `{"type":"bot_text_fragment","text":"..."}` |
| Server → Client | JSON | `{"type":"tts_start"}` |
| Server → Client | JSON | `{"type":"tts_end"}` |
| Server → Client | JSON | `{"type":"tts_stopped"}` — TTS interrupted |
| Server → Client | JSON | `{"type":"error","message":"..."}` |
| Server → Client | JSON | `{"type":"pong"}` |

---

## Interrupt Flow

```
1. Browser VAD: RMS energy > VAD_THRESHOLD for VAD_HOLD_FRAMES (4×100ms)
2. Browser sends: {"type":"interrupt"}
3. Server: session.cancel_tts() → sets tts_cancel_event + interrupt_event
4. TTS handler: checks cancel_event before each PCM chunk → breaks
5. TTS orchestrator: drains fragment queue (discards stale LLM output)
6. Server sends: {"type":"tts_stopped"}
7. session.reset_for_new_turn() → clears events
8. New user utterance processed from clean state
```

---

## Troubleshooting

**Nothing happens when I speak**
Check that `VAD_THRESHOLD` is not too high. Open browser DevTools (Console) and watch for `[VAD]` log lines. If none appear, the mic may not have permission.

**Bot starts talking then immediately stops**
`VAD_THRESHOLD` is too low — background noise is triggering an interrupt. Raise `VAD_RMS_THRESHOLD` to `0.025` or higher.

**Riva: "Connection refused" / gRPC error**
When `NVIDIA_API_KEY` is set, the server uses `grpc.nvcf.nvidia.com:443` — no local Riva needed. If you see gRPC errors, verify the key is valid at [build.nvidia.com](https://build.nvidia.com).

**Microphone blocked in browser**
Browsers require HTTPS (or `localhost`) for `getUserMedia`. Use the self-signed cert approach above.

**First response is very slow**
Ollama loads model weights into GPU on first inference. Pre-warm it:
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.2:1b","prompt":"hello","stream":false}'
```

**FAISS index rebuilt every restart**
The `faiss_index/` directory is missing or the process lacks write access to the `backend/` directory.

**`AlbertModel` import error from transformers**
`sentence-transformers 5.x` requires `transformers >= 5.0` which breaks Kokoro. Pin versions:
```bash
pip install "sentence-transformers==3.0.1" "transformers>=4.40,<5.0"
```

---

## License

MIT
