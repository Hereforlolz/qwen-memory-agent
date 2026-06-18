# Qwen MemoryAgent

**Track 1: MemoryAgent — Global AI Hackathon with Qwen Cloud**

A persistent memory layer where Qwen actively scores, recalls, synthesizes, and forgets memories across sessions. Unlike stateless chat, MemoryAgent remembers what users tell it — preferences, goals, project details, personal context — and injects that memory into every response, even days later.

---

## What It Does

Most AI chat is stateless. Every session starts from zero. MemoryAgent fixes that.

Every conversation turn runs through a full memory pipeline:

1. **Recall** — semantic search finds relevant past memories for the current query
2. **Synthesize** — Qwen builds a context window from recalled memories before responding
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

**Semantic recall** — pgvector cosine similarity, ranked by `similarity × importance_score` so high-importance memories surface first even with moderate similarity.

**Context synthesis** — recalled memories aren't dumped raw into the prompt. Qwen synthesizes them into a 3-sentence context paragraph tuned to the current query.

**Smart extraction** — after each turn, a second Qwen call extracts structured facts from the conversation (up to 3 per turn) rather than storing raw message text.

**Smart forget** — Qwen reviews low-importance expired memories and decides delete vs keep, preventing both memory bloat and accidental loss of edge-case useful context.

---

## Project Structure

```
qwen-memory-agent/
├── memory_api.py        # Core memory CRUD, scoring, recall, forget
├── agent.py             # Chat layer — memory injection + extraction
├── seed_memories.py     # Test data seeder
├── frontend/
│   └── index.html       # Chat UI + live memory panel
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
| DELETE | `/forget` | Trigger Qwen-powered smart forget |
| GET | `/health` | Health check (DB + Redis ping) |

### agent.py (port 8001)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat` | Full memory-injected chat turn |
| GET | `/chat/memories/{user_id}` | Memory panel data for frontend |
| GET | `/health` | Health check |

---

## Frontend

Open `http://localhost:8001/app` after starting `agent.py`.

- **Chat panel** — standard chat, with badges showing how many memories were recalled and whether the turn was stored
- **Memory panel** — live view of all stored memories, sorted by importance score, with color-coded TTL bars
- **New Session** — starts a fresh session ID while keeping all memories intact (tests cross-session recall)
- **Clear Chat** — wipes the UI conversation history, memory unaffected
- **🗑 per-card delete** — remove individual memories
- **✕ Clear All** — nuke all memories for the current user ID

---

## Cross-Session Recall Demo

1. Start a session, tell the agent your name, your project, your preferences
2. Click **New Session** (or restart the server entirely)
3. Ask the agent something related — it will recall and reference what you told it
4. The memory panel shows which memories were injected into that response

---

## Services Used

- [Qwen Cloud / DashScope](https://dashscope-intl.aliyuncs.com) — LLM + embeddings
- [Neon](https://neon.tech) — serverless Postgres with pgvector
- [Upstash](https://upstash.com) — serverless Redis
- [Alibaba Cloud ECS](https://www.alibabacloud.com) — deployment

---

## License

MIT