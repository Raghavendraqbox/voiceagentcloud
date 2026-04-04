# VoiceAgentCloud — Qobox Assistant

A **near full-duplex conversational voice AI agent** simulating a live phone call for **Qobox (Quality Outside The Box)**, an Indian software QA company. The user speaks freely; the bot responds in real time — no push-to-talk required.

---

## How It Works

```
Browser mic (16kHz PCM) ──► WebSocket ──► FastAPI backend
                                               │
                              ┌────────────────┼────────────────────┐
                              ▼                ▼                    ▼
                           ASR loop        LLM loop            TTS loop
                        (Riva NVCF)    (Ollama / NIM /      (Riva NVCF /
                       [Parakeet-1.1B]   Claude API)        Kokoro GPU)
                              │                │                    │
                              └────────────────┴────────────────────┘
                                               │
                                     Binary PCM audio ──► Browser playback
```

VAD in the browser detects speech → sends interrupt → TTS stops within 200 ms.

---

## Stack

| Layer | Primary | Fallback 1 | Fallback 2 |
|---|---|---|---|
| **ASR** | Riva NVCF (Parakeet-CTC-1.1B) | Whisper small (GPU) | Whisper tiny (CPU) |
| **LLM** | Ollama llama3.1:8b (local) | NVIDIA NIM Nemotron-70B | Claude Haiku / stub |
| **TTS** | Riva NVCF (FastPitch-HiFiGAN) | Kokoro GPU (24 kHz) | edge-tts / silence |
| **RAG** | FAISS + sentence-transformers | — | — |

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11 | `python3.11 --version` |
| CUDA 12.x + GPU | A100 / H100 / RTX 3090 recommended. CPU-only works but is slow. |
| Git | `git --version` |
| NVIDIA API key | Free at [build.nvidia.com](https://build.nvidia.com) — enables cloud Riva ASR+TTS+NIM |
| Anthropic API key | Optional. Fallback LLM if Ollama is down. Get at [console.anthropic.com](https://console.anthropic.com) |

---

## Clone & Run — Step by Step

### Step 1 — Clone the repository

```bash
git clone https://github.com/Raghavendraqbox/voiceagentcloud.git
cd voiceagentcloud
```

---

### Step 2 — Create Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

---

### Step 3 — Install PyTorch with CUDA (do this BEFORE requirements.txt)

Check your CUDA version first:
```bash
nvcc --version    # e.g. "release 12.4"
nvidia-smi        # confirms GPU is visible
```

Install the matching PyTorch wheel:
```bash
# CUDA 12.4 (recommended — matches this project)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (no GPU — TTS/ASR fallbacks will use CPU)
# pip install torch torchaudio
```

Verify CUDA is available:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
# Expected: CUDA: True | GPU: NVIDIA A100 ...
```

---

### Step 4 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If you see an `AlbertModel` import error later, run:
> ```bash
> pip install "sentence-transformers==3.0.1" "transformers>=4.40,<5.0"
> ```
> This happens when `sentence-transformers` 5.x pulls in `transformers` 5.x, which breaks Kokoro.

---

### Step 5 — Install Ollama + pull the LLM model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service (runs in background on port 11434)
ollama serve &

# Pull the 8B model (~4.7 GB, one-time download)
ollama pull llama3.1:8b

# Verify
ollama list
# Should show: llama3.1:8b
```

> **No GPU / low VRAM?** Use the 1B model instead (much less accurate for memory):
> ```bash
> ollama pull llama3.2:1b
> export OLLAMA_MODEL=llama3.2:1b
> ```

---

### Step 6 — Create your .env file

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```bash
# Required for NVIDIA cloud Riva ASR + TTS + NIM LLM
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional — Claude Haiku fallback LLM when Ollama is unreachable
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxx
```

Everything else in `.env` has sensible defaults and does not need changing for a first run.

---

### Step 7 — Start the server

```bash
cd backend

# Simple start (reads .env automatically if python-dotenv is installed)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info

# OR — pass keys inline without a .env file
NVIDIA_API_KEY=nvapi-xxx ANTHROPIC_API_KEY=sk-ant-xxx \
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info
```

Wait for these lines in the log — the server is ready when you see all three:
```
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
Kokoro warmup complete — GPU kernels compiled
```

The first startup takes ~15 seconds (sentence-transformers loads, FAISS index builds).
Subsequent starts are ~5 seconds (FAISS loads from cache).

---

### Step 8 — Open in browser

```
http://localhost:8000
```

Click **Start Talking** and speak. You should hear the Qobox greeting within 1–2 seconds.

> **Remote server / RunPod?** See the [HTTPS section](#https-for-remote-access) below.
> Browsers block microphone access on non-`localhost` HTTP origins.

---

### Step 9 — Verify everything is working

In the server log you should see, in order:
```
ASR: connecting to NVIDIA Riva via NVCF cloud      ← Riva ASR active
TTS: connecting to NVIDIA Riva via NVCF cloud      ← Riva TTS active
...
[on first user speech]
Processing transcript: <what you said>
HTTP Request: POST http://localhost:11434/api/generate  ← Ollama LLM responding
TTS synthesize: <bot reply text>
```

---

## HTTPS for Remote Access

Browsers require HTTPS (or `localhost`) for microphone access. Two options:

### Option A — Self-signed certificate (RunPod / dev server)

```bash
# Generate cert (one-time)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
  -days 365 -nodes -subj "/CN=voiceagent"

# Start with SSL
cd backend
NVIDIA_API_KEY=nvapi-xxx \
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 \
  --ssl-keyfile ../key.pem --ssl-certfile ../cert.pem
```

Open `https://YOUR_SERVER_IP:8000` in Chrome — click **Advanced → Proceed** on the cert warning.

### Option B — Nginx reverse proxy (production)

```bash
sudo apt install nginx certbot python3-certbot-nginx
sudo certbot --nginx -d voiceagent.yourdomain.com
```

Add to `/etc/nginx/sites-available/voiceagent`:
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

### RunPod

1. In pod settings → expose port `8000`.
2. RunPod provides `https://<pod-id>-8000.proxy.runpod.net` — HTTPS is handled for you; no self-signed cert needed.
3. Set `NVIDIA_API_KEY` in the pod's **Environment Variables** tab before starting.

---

## Environment Variables Reference

All settings have defaults and can be overridden in `.env` or as shell exports. See `.env.example` for the full list with descriptions.

| Variable | Default | What it controls |
|---|---|---|
| `NVIDIA_API_KEY` | _(unset)_ | Enables NVCF cloud Riva ASR+TTS and NIM LLM |
| `ANTHROPIC_API_KEY` | _(unset)_ | Claude Haiku fallback LLM |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `RIVA_SERVER_URL` | `localhost:50051` | Local Riva gRPC (only if no NVIDIA_API_KEY) |
| `RAG_DOCS_DIR` | `./docs` | Directory of `.txt` knowledge files |
| `VAD_RMS_THRESHOLD` | `0.018` | Raise to reduce false interrupts from background noise |
| `LOG_LEVEL` | `info` | `debug` for detailed pipeline tracing |
| `SERVER_PORT` | `8000` | HTTP/WebSocket port |

---

## Adding Knowledge Base Documents

1. Add `.txt` files to the `docs/` directory.
2. Delete the cached FAISS index:
   ```bash
   rm -rf backend/faiss_index/
   ```
3. Restart the server — the index rebuilds automatically.

---

## Troubleshooting

**Nothing happens when I speak / bot never responds**
- Open browser DevTools (F12) → Console tab → look for `[VAD]` lines.
- If VAD lines appear but no response: check server log for `Processing transcript:`.
- If no VAD lines: mic permission may be blocked. Check browser padlock icon → Microphone.

**Bot starts talking then immediately stops**
- Background noise is above `VAD_THRESHOLD`. Open `.env`, set `VAD_RMS_THRESHOLD=0.025` and restart the server.

**"CUDA: False" after installing PyTorch**
- Wrong PyTorch wheel was installed. Reinstall with the correct CUDA version:
  ```bash
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```

**`AlbertModel` import error**
- `transformers 5.x` was pulled in. Fix:
  ```bash
  pip install "sentence-transformers==3.0.1" "transformers>=4.40,<5.0"
  ```

**Agent ignores the user's name / poor memory**
- Make sure `llama3.1:8b` is being used, not `llama3.2:1b`. Check with `ollama list`.
- The 1B model is too small to reliably follow conversation history instructions.

**Riva gRPC errors (StatusCode.UNAUTHENTICATED)**
- `NVIDIA_API_KEY` is missing or expired. Verify at [build.nvidia.com](https://build.nvidia.com).

**Ollama "connection refused"**
- Ollama service is not running. Start it: `ollama serve &`
- The server will fall back to NIM (if `NVIDIA_API_KEY` set) or Claude API automatically.

**First response is very slow (> 5 s)**
- Ollama loads model weights into GPU on first inference. Pre-warm it:
  ```bash
  curl http://localhost:11434/api/generate \
    -d '{"model":"llama3.1:8b","prompt":"hello","stream":false}'
  ```

**FAISS index rebuilds every restart**
- The `backend/faiss_index/` directory is missing write access. Ensure the process runs from inside `backend/`.

---

## Repository Structure

```
voiceagentcloud/
├── backend/
│   ├── main.py              # FastAPI app, WebSocket handler, HTTP routes
│   ├── session_manager.py   # Per-session pipeline, 3 async loops, interrupt logic
│   ├── asr.py               # Riva NVCF streaming ASR (Whisper fallback)
│   ├── tts.py               # Riva NVCF streaming TTS (Kokoro GPU fallback)
│   ├── llm.py               # Ollama / NIM / Claude streaming LLM client
│   ├── rag.py               # FAISS vector DB + Qobox knowledge documents
│   ├── memory.py            # Sliding-window conversation memory (last 8 turns)
│   └── config.py            # All settings, overridable via env vars
├── frontend/
│   └── index.html           # Single-page browser UI (no build step)
├── docs/
│   └── *.txt                # Additional RAG knowledge files (add your own)
├── .env.example             # All environment variables documented
├── requirements.txt         # Python dependencies (pinned)
├── CHANGELOG.md
└── README.md
```

---

## License

MIT
