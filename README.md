# Qwen MemoryAgent

**Track 1: MemoryAgent — Global AI Hackathon with Qwen Cloud**

A persistent memory layer where Qwen actively scores, recalls, and forgets memories across sessions. Unlike stateless chat, MemoryAgent remembers what users tell it — preferences, goals, project details, personal context — and injects that memory into every response, even days later.

---

## What It Does

Most AI chat is stateless. Every session starts from zero. MemoryAgent fixes that.

Every conversation turn runs through a full memory pipeline:

1. **Recall** — semantic search finds relevant past memories for the current query
2. **Retrieve** — recalled memories are formatted into a context block ordered by importance and relevance
3. **Extract** — after responding, Qwen extracts key facts from the turn as structured memories
4. **Score** — each memory gets an importance score (0.0–1.0) from Qwen
5. **TTL** — low-importance memories expire in 24h, medium in 7 days, high-importance are permanent
6. **Forget** — a smart forget endpoint lets Qwen review and delete truly useless memories

The result: an agent that gets more useful over time, not less.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User (Browser)                        │
│              frontend/index.html  (port 8001/app)            │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    agent.py  (port 8001)                     │
│  POST /chat        → recall → Qwen chat → extract → store   │
│  GET  /chat/memories/{user_id}                               │
└──────────┬──────────────────────────────┬───────────────────┘
           │ HTTP                          │ OpenAI-compat SDK
           ▼                              ▼
┌──────────────────────┐      ┌───────────────────────────────┐
│   memory_api.py      │      │       Qwen Cloud API           │
│   (port 8000)        │      │  qwen-plus  (chat + scoring)  │
│                      │      │  text-embedding-v3 (1024-dim) │
│  POST /memory        │      └───────────────────────────────┘
│  POST /recall        │
│  DELETE /memory/{id} │
│  DELETE /memories/   │
│  DELETE /forget      │
│  GET  /memories/     │
└──────┬───────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐    ┌───────────────┐
│  Neon DB    │    │  Upstash Redis │
│  Postgres   │    │  (cache 1hr)   │
│  + pgvector │    └───────────────┘
│  1024-dim   │
└─────────────┘
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM + Scoring | Qwen Cloud (`qwen-plus`) |
| Embeddings | Qwen Cloud (`text-embedding-v3`, 1024-dim) |
| Vector DB | Neon PostgreSQL + pgvector |
| Cache | Upstash Redis |
| Backend | FastAPI + asyncpg |
| Frontend | Vanilla HTML/CSS/JS |
| Deploy | Alibaba Cloud ECS |

---

## Memory Intelligence

**Importance scoring** — Qwen rates every memory 0.0–1.0 based on content type:
- `>= 0.6` → permanent (goals, preferences, key facts)
- `0.3–0.6` → expires in 7 days
- `< 0.3` → expires in 24 hours (greetings, filler)

The scoring prompt is explicitly calibrated against two failure modes found during testing:
- **Specific beats vague** — a named, concrete fact (e.g. "grows cherry tomatoes and basil") must score at least as high as the general category it belongs to (e.g. "is an indoor gardener"). Without this rule, vague statements were outscoring the specific facts that actually make recall useful.
- **Names have a floor** — a person's own name is treated as a 0.6+ identity fact, never scored as casual/trivial, so it can't accidentally expire within a day of being mentioned.

**Importance-first recall** — pgvector cosine similarity, ranked `ORDER BY importance_score DESC, similarity DESC`. This guarantees a user's core profile facts (name, project, deadlines) surface on the very first turn of a new session, even before enough conversation exists for a strong semantic match.

**Weighted duplicate/conflict detection** — before storing, candidate memories are ranked by a blended score (`60% similarity + 40% importance`) and checked against the new content:
- **similarity > 0.96** or exact text match → rejected as duplicate (`409`)
- **similarity > 0.82** → Qwen arbitrates: does the new fact `UPDATE` (correct/supersede) the old one, or is it a `NEW` independent fact? If `UPDATE`, the old row is overwritten in place rather than creating a redundant entry.

**Negative-fact filtering** — the extraction prompt explicitly forbids generating memories from absence-of-information statements (e.g. "user doesn't have a car"), preventing the memory store from filling with noise.

**Structured context retrieval** — recalled memories aren't dumped into the prompt as raw rows; they're formatted into a clean, dated bullet list ordered by importance and relevance before being injected into the system prompt.

**Smart extraction** — after each turn, a second Qwen call extracts structured facts from the conversation (up to 3 per turn) rather than storing raw message text.

**Smart forget** — memories with `expires_at` in the past (low/medium importance only — permanent memories with `expires_at = NULL` are never touched) are batch-reviewed via `DELETE /forget`. For each candidate, Qwen weighs the content, its original importance score, and its age, then votes `DELETE` or `KEEP`. Deleted memories are hard-removed from Neon and purged from the Redis cache; kept memories get their TTL renewed by 7 days rather than being re-flagged every cycle.

**Prompt injection hardening** — since stored memories get re-injected into the system prompt of *future*, unrelated sessions, a malicious message stored as a "memory" could otherwise function as a persistent, cross-session jailbreak. Both the system prompt (recall) and the extraction prompt (storage) explicitly frame all user/memory content as untrusted data to read, never instructions to follow — including text that impersonates system/admin commands. The conflict-arbitration prompt has the same framing, so a crafted memory can't manipulate the UPDATE/NEW verdict into overwriting unrelated memories. This is prompt-level defense-in-depth, not a hard guarantee — LLM-based defenses reduce but don't eliminate injection risk.

---

## Project Structure

```
qwen-memory-agent/
├── memory_api.py        # Core memory CRUD, scoring, recall, forget
├── agent.py             # Chat layer — memory injection + extraction
├── seed_memories.py     # Test data seeder
├── test_memory_agent.py # End-to-end test suite (API + frontend file validation)
├── frontend/
│   └── index.html       # Chat UI + live memory panel + intro/how-to-use guide
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Hereforlolz/qwen-memory-agent
cd qwen-memory-agent
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in `.env`:

```
QWEN_API_KEY=sk-...
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
QWEN_EMBEDDING_MODEL=text-embedding-v3
DATABASE_URL=postgresql://...neon.tech/neondb?sslmode=require
REDIS_URL=rediss://default:...@...upstash.io:6379
MEMORY_API_URL=http://localhost:8000
```

> **Note:** Upstash requires `rediss://` (double s) for TLS. Neon requires `?sslmode=require`.

### 3. Run

Terminal 1 — memory API:
```bash
python memory_api.py
# running on http://localhost:8000
```

Terminal 2 — chat agent + frontend:
```bash
python agent.py
# API on http://localhost:8001
# Frontend at http://localhost:8001/app
```

### 4. Optional — seed test memories

```bash
python seed_memories.py
```

---

## API Reference

### memory_api.py (port 8000)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/memory` | Store a memory — Qwen scores + embeds |
| POST | `/recall` | Semantic search + context synthesis |
| GET | `/memories/{user_id}` | List all memories, sorted by importance |
| DELETE | `/memory/{memory_id}` | Hard delete a single memory |
| DELETE | `/memories/{user_id}` | Delete all memories for a user |
| DELETE | `/forget` | Qwen-arbitrated smart forget — reviews expired, low-importance memories and deletes or renews each (body: `{user_id, batch_size}`) |
| GET | `/health` | Health check (DB + Redis ping) |

### agent.py (port 8001)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat` | Full memory-injected chat turn — returns `memories_stored: list[dict]` |
| GET | `/chat/memories/{user_id}` | Memory panel data for frontend |
| DELETE | `/forget/{user_id}` | Proxies to memory_api.py's smart forget |
| GET | `/health` | Health check |

---

## Frontend

Open `http://localhost:8001/app` after starting `agent.py`.

- **Chat panel** — standard chat, with badges showing how many memories were recalled and whether the turn was stored
- **Memory panel** — live view of all stored memories, sorted by importance score, with color-coded TTL bars
- **🧹 Smart Forget button** — manually triggers a review of expired memories for the current user via `DELETE /forget/{user_id}`, and shows the reviewed/deleted/kept counts inline
- **New Session** — starts a fresh session ID while keeping all memories intact (tests cross-session recall)
- **Clear Chat** — wipes the UI conversation history, memory unaffected
- **🗑 per-card delete** — remove individual memories
- **✕ Clear All** — nuke all memories for the current user ID
- **"How this works" intro panel** — shown on first load, walks new users/judges through what the app does and a step-by-step script to test cross-session recall in under a minute. Dismissible via the close button, reopenable via the header link.

---

## Cross-Session Recall Demo

1. Start a session, tell the agent your name, your project, your preferences
2. Click **New Session** (or restart the server entirely)
3. Ask the agent something related — it will recall and reference what you told it
4. The memory panel shows which memories were injected into that response

---

## Testing

`test_memory_agent.py` is a real end-to-end suite — it makes live HTTP calls against a running instance (local or deployed) rather than mocking anything, so a passing run is genuine proof the system behaves as documented.

```bash
# against local servers (memory_api.py on :8000, agent.py on :8001)
python test_memory_agent.py

# against the live Alibaba Cloud deployment
python test_memory_agent.py --remote <ECS-public-IP>
```

Covers: health checks, store/recall, importance scoring calibration (including the specific-vs-vague and name-floor rules above), deduplication and conflict arbitration, negative-fact filtering, cross-session recall, smart forget, manual delete, and a structural validation pass over `frontend/index.html` (catches regressions like a hardcoded `API_BASE` or a leaked default user ID before they reach a live deployment).

---

## Services Used

- [Qwen Cloud / DashScope](https://dashscope-intl.aliyuncs.com) — LLM + embeddings
- [Neon](https://neon.tech) — serverless Postgres with pgvector
- [Upstash](https://upstash.com) — serverless Redis
- [Alibaba Cloud ECS](https://www.alibabacloud.com) — deployment

---
##Link to demo
DEMO: https://vimeo.com/1204609875?fl=tl&fe=ec

## License

MIT
