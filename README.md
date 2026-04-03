# VoiceAgentCloud

A **near full-duplex conversational voice AI agent** that simulates a live phone call. The user speaks freely, the bot interrupts and responds in real time — no push-to-talk.

Built with:
- **NVIDIA Riva** — streaming ASR (speech-to-text) + TTS (text-to-speech) via gRPC
- **LLaMA 3 via Ollama** — local LLM for natural language responses
- **FAISS RAG** — telecom knowledge base for grounded answers
- **FastAPI + WebSockets** — async real-time backend
- **Browser AudioWorklet** — low-latency mic capture and audio playback

> **No Riva or Ollama?** The server boots in **stub mode** — ASR emits fake transcripts, TTS sends silence, and the LLM returns pre-canned responses. You can see the full pipeline working immediately without any GPU services.

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
    │      Riva gRPC streaming ASR  →  TranscriptResult queue
    │
    ├─ [Loop 2] LLM  (llm.py)
    │      Reads final transcripts  →  streams sentence fragments
    │      Injects: RAG context + conversation memory
    │
    └─ [Loop 3] TTS  (tts.py)
           Consumes sentence fragments  →  Riva streaming TTS
           Checks cancel_event before every PCM chunk
           →  binary audio back to browser

Browser AudioContext chains PCM chunks into seamless playback.
VAD (RMS energy) detects user speech → sends interrupt → TTS stops < 100ms.
```

---

## Repository Structure

```
voiceagentcloud/
├── backend/
│   ├── main.py              # FastAPI app + WebSocket handler + HTTP routes
│   ├── session_manager.py   # Per-session state, 3-loop pipeline, interrupt logic
│   ├── asr.py               # NVIDIA Riva streaming ASR (with stub fallback)
│   ├── tts.py               # NVIDIA Riva streaming TTS (with stub fallback)
│   ├── llm.py               # Ollama LLaMA 3 async streaming client (with stub fallback)
│   ├── rag.py               # FAISS vector DB + sentence-transformer embeddings
│   ├── memory.py            # Sliding-window conversation memory (last 8 turns)
│   └── config.py            # All settings, overridable via env vars
├── frontend/
│   └── index.html           # Single-page browser UI (no build step needed)
├── docs/
│   ├── telecom_products.txt # RAG knowledge base — telecom product catalog
│   └── troubleshooting.txt  # RAG knowledge base — troubleshooting guide
├── requirements.txt
└── README.md
```

---

## Quick Start (Stub / Dev Mode — no GPU required)

This gets the server running immediately. ASR and TTS operate in silent stub mode; LLM returns pre-canned responses.

```bash
# 1. Clone
git clone https://github.com/Raghavendraqbox/voiceagentcloud.git
cd voiceagentcloud

# 2. Python environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies (no Riva, no GPU needed)
pip install --upgrade pip
pip install fastapi "uvicorn[standard]" websockets httpx python-multipart \
            sentence-transformers faiss-cpu numpy aiofiles structlog \
            grpcio grpcio-tools protobuf

# 4. Start server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000

# 5. Open browser
# http://localhost:8000
# Click "Start Talking" — you will see transcripts + bot text flowing
```

---

## Full Production Setup (GPU Server / RunPod)

### Prerequisites

| Component | Recommended |
|---|---|
| OS | Ubuntu 22.04 LTS |
| GPU | RTX 3090 / A100 / H100 |
| CUDA | 12.x |
| cuDNN | 8.x |
| Python | 3.11 |
| Docker | 24+ |
| NVIDIA Container Toolkit | latest |

---

### Step 1 — Verify GPU

```bash
nvidia-smi        # must show your GPU
nvcc --version    # must show CUDA version
```

---

### Step 2 — NVIDIA Riva (ASR + TTS)

#### 2a. Install NGC CLI

```bash
wget --content-disposition \
  https://ngc.nvidia.com/downloads/ngccli_linux.zip -O ngccli_linux.zip
unzip ngccli_linux.zip
chmod u+x ngc-cli/ngc
sudo mv ngc-cli/ngc /usr/local/bin/ngc

# Sign in with a free NGC account
ngc config set
```

#### 2b. Download Riva Quickstart

```bash
ngc registry resource download-version \
  "nvidia/riva/riva_quickstart:2.14.0"
cd riva_quickstart_v2.14.0
```

#### 2c. Configure Riva (enable ASR + TTS only)

Edit `config.sh`:

```bash
service_enabled_asr=true
service_enabled_nlp=false
service_enabled_nmt=false
service_enabled_tts=true

asr_acoustic_model=("en-US-conformer-ctc-l")
tts_model=("English-US.Female-1")
```

#### 2d. Initialize and Start Riva

```bash
bash riva_init.sh     # Downloads models (~8 GB, one-time)
bash riva_start.sh    # Starts gRPC server on port 50051

# Verify
docker ps | grep riva
# Should show riva-speech running
```

---

### Step 3 — Ollama + LLaMA 3

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull LLaMA 3 8B (~5 GB)
ollama pull llama3

# Verify
ollama list
curl http://localhost:11434/api/tags

# Optional: pre-warm the model (first inference is slow)
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3","prompt":"hi","stream":false}'
```

For high-end GPUs (A100/H100), use the 70B model for much better quality:

```bash
ollama pull llama3:70b
export OLLAMA_MODEL=llama3:70b
```

---

### Step 4 — Python Dependencies

```bash
cd voiceagentcloud

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU FAISS** (optional, faster RAG retrieval):
> ```bash
> pip uninstall faiss-cpu
> pip install faiss-gpu==1.7.4   # requires CUDA 11.x compatible wheels
> ```

---

### Step 5 — Configuration

All settings in `backend/config.py` can be overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `RIVA_SERVER_URL` | `localhost:50051` | Riva gRPC endpoint |
| `RIVA_TTS_VOICE` | `English-US.Female-1` | Riva TTS voice name |
| `RIVA_ASR_LANGUAGE` | `en-US` | ASR language code |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP endpoint |
| `OLLAMA_MODEL` | `llama3` | Model name (e.g. `llama3:70b`) |
| `OLLAMA_MAX_TOKENS` | `150` | Max LLM output tokens (lower = faster) |
| `OLLAMA_TEMPERATURE` | `0.7` | LLM temperature |
| `RAG_DOCS_DIR` | `./docs` | Directory of `.txt` knowledge files |
| `RAG_INDEX_PATH` | `./faiss_index` | FAISS index cache location |
| `RAG_TOP_K` | `3` | Number of RAG results to inject |
| `VAD_RMS_THRESHOLD` | `0.01` | Mic RMS energy to trigger interrupt |
| `MEMORY_MAX_TURNS` | `8` | Conversation turns to keep in memory |
| `SERVER_PORT` | `8000` | HTTP/WebSocket port |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warning` |

Create a `.env` file or export before starting:

```bash
export RIVA_SERVER_URL=localhost:50051
export OLLAMA_MODEL=llama3
export SERVER_PORT=8000
```

---

### Step 6 — Start the Server

```bash
cd voiceagentcloud/backend

uvicorn main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --log-level info
```

- On first start, the FAISS index builds from `docs/*.txt` (takes ~10s).
- Subsequent starts load the index cache instantly.
- Open **http://localhost:8000** in Chrome.
- Click **Start Talking** and speak.

---

### Step 7 — HTTPS for Remote Access

Browsers block microphone access on non-`localhost` HTTP origins. Use HTTPS.

#### Self-signed (dev / RunPod)

```bash
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
  -days 365 -nodes -subj "/CN=voiceagent"

uvicorn main:app \
  --host 0.0.0.0 --port 8000 --workers 1 \
  --ssl-keyfile ../key.pem \
  --ssl-certfile ../cert.pem
```

Then open **https://YOUR_SERVER_IP:8000** (accept the browser warning).

#### Nginx reverse proxy (production)

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
        proxy_buffering    off;   # critical for binary WebSocket audio frames
    }
}
```

---

### RunPod Specific Notes

1. When creating a pod, **expose port 8000** (or whichever `SERVER_PORT` you set) in the pod configuration.
2. RunPod assigns a public URL like `https://<pod-id>-8000.proxy.runpod.net` — use that URL in your browser.
3. Because RunPod proxies are HTTPS, microphone access works without self-signed certs.
4. Install Riva inside the pod or point `RIVA_SERVER_URL` to an external Riva host.
5. If GPU memory is limited (< 16 GB), use `llama3` (8B) and Riva ASR-only (skip TTS model to save VRAM).

---

## Adding Knowledge Base Documents

1. Add `.txt` files to the `docs/` directory.
2. Delete the cached index:
   ```bash
   rm -rf backend/faiss_index/
   ```
3. Restart the server — the index rebuilds automatically.

---

## WebSocket Protocol Reference

| Direction | Frame type | Payload |
|---|---|---|
| Client → Server | binary | Raw PCM audio (16-bit, mono, 16kHz, ~3200 bytes / 100ms) |
| Client → Server | JSON text | `{"type":"interrupt"}` — user started speaking |
| Client → Server | JSON text | `{"type":"ping"}` — keepalive |
| Server → Client | binary | Raw PCM audio (22050 Hz, 16-bit, mono) — TTS output |
| Server → Client | JSON text | `{"type":"session_ready","session_id":"..."}` |
| Server → Client | JSON text | `{"type":"transcript_partial","text":"..."}` |
| Server → Client | JSON text | `{"type":"transcript_final","text":"..."}` |
| Server → Client | JSON text | `{"type":"bot_text_fragment","text":"..."}` |
| Server → Client | JSON text | `{"type":"tts_start"}` |
| Server → Client | JSON text | `{"type":"tts_end"}` |
| Server → Client | JSON text | `{"type":"tts_stopped"}` — TTS interrupted |
| Server → Client | JSON text | `{"type":"error","message":"..."}` |
| Server → Client | JSON text | `{"type":"pong"}` |

---

## Interrupt Flow (< 100ms)

```
1. Browser VAD: RMS energy > threshold for 8 consecutive 100ms frames
2. Browser sends: {"type":"interrupt"}
3. Server: session.cancel_tts() → sets tts_cancel_event
4. TTS handler: checks cancel_event BEFORE each PCM chunk → breaks
5. TTS orchestrator: drains fragment queue (discards stale LLM output)
6. Server sends: {"type":"tts_stopped"}
7. session.reset_for_new_turn() → clears both events
8. New user utterance processed from clean state
```

---

## Latency Tuning

| Lever | Effect |
|---|---|
| Use LLaMA 3 8B (not 70B) | Fastest TTFT on single GPU |
| `OLLAMA_MAX_TOKENS=80` | Caps response, forces shorter bot turns |
| Run Riva on same host | Eliminates gRPC network RTT |
| `faiss-gpu` instead of `faiss-cpu` | Sub-ms RAG retrieval |
| Reduce `RAG_TOP_K` to 2 | Less context injection = faster tokenization |
| Lower `VAD_RMS_THRESHOLD` | Detect speech earlier, interrupt sooner |

---

## Troubleshooting

**"nvidia-riva-client not installed" warning**
The server runs in stub mode. Install `nvidia-riva-client==2.14.0` and start Riva to get real speech I/O.

**Riva "Connection refused" on port 50051**
```bash
docker ps | grep riva         # check container is running
netstat -tlnp | grep 50051    # check port is bound
```

**Microphone blocked in browser**
Browsers require HTTPS (or `localhost`) for `getUserMedia`. Use self-signed cert or Nginx + Let's Encrypt.

**First response is very slow**
Ollama loads model weights into GPU on first inference. Pre-warm:
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"llama3","prompt":"hello","stream":false}'
```

**FAISS index rebuilt every restart**
The `faiss_index/` directory is missing or write-protected. Ensure the process has write access to the `backend/` working directory.

**High interrupt latency**
Lower `VAD_RMS_THRESHOLD` in `.env` so VAD fires faster. Also check browser frame size — AudioWorklet processes 128-sample frames at 48kHz; the JS downsamples to 16kHz before sending.

---

## License

MIT
