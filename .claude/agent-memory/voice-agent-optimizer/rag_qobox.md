---
name: RAG index — Qobox company data added
description: Qobox company info added to TELECOM_SEED_DOCS; cached FAISS index deleted for rebuild
type: project
---

Action taken (2026-04-04):
Added 7 new seed document strings for Quality Outside The Box (Qobox) to the `TELECOM_SEED_DOCS` list in `backend/rag.py`. The strings cover: company overview, founding/HQ/directors, industries served, core services (testing, automation, performance, security, API, QA consulting), technology stack, mission/vision, approach, culture, and future focus.

The stale cached index at `backend/faiss_index/index.faiss` and `backend/faiss_index/chunks.pkl` was deleted. On next server startup, `RAGRetriever.initialize()` will rebuild the FAISS index from the updated `TELECOM_SEED_DOCS` list and re-persist it.

Important: Any time seed documents are changed, the cached index must be deleted or the changes will not take effect. The RAGRetriever loads from cache first; it only builds from source if no cache exists.

**Why:** The FAISS index caches embeddings to avoid recomputing on every startup. The cache is not automatically invalidated when source docs change.
**How to apply:** After modifying TELECOM_SEED_DOCS, always run `rm backend/faiss_index/index.faiss backend/faiss_index/chunks.pkl` before restarting the server.
