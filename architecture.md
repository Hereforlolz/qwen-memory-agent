# Architecture

System architecture for Qwen MemoryAgent — Track 1: MemoryAgent, Global AI Hackathon with Qwen Cloud.

## Diagram

```mermaid
graph TB
    User["User<br/>(Browser)"]

    subgraph ECS["Alibaba Cloud ECS — Ubuntu 22.04"]
        Frontend["frontend/index.html<br/>Chat + Memory Panel"]
        Agent["agent.py — port 8001<br/>Chat orchestration layer"]
        MemoryAPI["memory_api.py — port 8000<br/>Memory CRUD + intelligence"]
    end

    subgraph Qwen["Qwen Cloud API"]
        QwenChat["qwen-plus<br/>chat completions"]
        QwenEmbed["text-embedding-v3<br/>1024-dim embeddings"]
    end

    subgraph Data["Data Layer"]
        Neon[("Neon PostgreSQL<br/>+ pgvector")]
        Redis[("Upstash Redis<br/>1hr cache")]
    end

    User -->|HTTP| Frontend
    Frontend -->|POST /chat| Agent
    Frontend -->|GET /chat/memories/user_id| Agent
    Frontend -->|DELETE /memory/id| Agent

    Agent -->|POST /recall| MemoryAPI
    Agent -->|POST /memory| MemoryAPI
    Agent -->|GET /memories/user_id| MemoryAPI
    Agent -->|proxy DELETE| MemoryAPI
    Agent -->|chat completion<br/>extract memories| QwenChat

    MemoryAPI -->|score importance<br/>synthesize context| QwenChat
    MemoryAPI -->|generate embeddings| QwenEmbed
    MemoryAPI -->|store/query vectors| Neon
    MemoryAPI -->|cache 1hr TTL| Redis

    style ECS fill:#1a1a2e,stroke:#7c3aed,color:#fff
    style Qwen fill:#1a2e1a,stroke:#34d399,color:#fff
    style Data fill:#2e1a1a,stroke:#f87171,color:#fff
    style User fill:#16161e,stroke:#a78bfa,color:#fff
```

## Request Flow — Chat Turn

```mermaid
sequenceDiagram
    participant U as User
    participant A as agent.py
    participant M as memory_api.py
    participant Q as Qwen Cloud
    participant N as Neon (pgvector)
    participant R as Upstash Redis

    U->>A: POST /chat {message}
    A->>M: POST /recall {query, top_k: 10}
    M->>Q: embed(query)
    Q-->>M: embedding vector
    M->>N: ORDER BY importance_score DESC,<br/>similarity DESC
    N-->>M: top-k memories<br/>(importance-first)
    M-->>A: {memories, context_window}

    A->>Q: chat completion<br/>(system prompt + memory context + history)
    Q-->>A: assistant reply

    A->>Q: extract structured memories<br/>(excludes negative/absence facts)
    Q-->>A: JSON array of facts

    loop for each extracted memory (max 3)
        A->>M: POST /memory {content}
        M->>Q: embed(content)
        Q-->>M: embedding vector
        M->>N: rank candidates by<br/>60% similarity + 40% importance
        N-->>M: top-3 candidates

        alt similarity > 0.96 or exact match
            M-->>A: 409 — SKIP (duplicate)
        else similarity > 0.82
            M->>Q: arbitrate: UPDATE or NEW?
            Q-->>M: verdict
            alt verdict = UPDATE
                M->>N: UPDATE existing row<br/>(content, embedding, score)
                M->>R: invalidate old cache entry
            else verdict = NEW
                M->>Q: score importance (0–1)
                Q-->>M: score
                M->>N: INSERT new row
                M->>R: cache memory (1hr TTL)
            end
        else no close match
            M->>Q: score importance (0–1)
            Q-->>M: score
            M->>N: INSERT new row
            M->>R: cache memory (1hr TTL)
        end
    end

    A-->>U: {reply, memories_used,<br/>memories_stored, context_injected}
```

## Memory Lifecycle

```mermaid
flowchart LR
    A[New memory content] --> B[Embed content<br/>+ rank top-3 candidates<br/>60% similarity + 40% importance]
    B --> C{Top candidate<br/>similarity?}
    C -->|> 0.96 or exact match| D[Reject — 409 SKIP]
    C -->|0.82 – 0.96| E[Qwen arbitrates:<br/>UPDATE or NEW?]
    C -->|< 0.82| F[Treat as NEW]

    E -->|UPDATE| G[Overwrite existing row<br/>content, embedding, score]
    E -->|NEW| F

    F --> H[Qwen scores importance<br/>0.0 – 1.0]
    H --> I{Score range}
    I -->|< 0.3| J[TTL: 24 hours]
    I -->|0.3 – 0.6| K[TTL: 7 days]
    I -->|≥ 0.6| L[Permanent]

    J --> M[(Stored in Neon<br/>+ pgvector embedding)]
    K --> M
    L --> M
    G --> M
    M --> N[Cached in Redis<br/>1hr TTL]

    O[DELETE /forget<br/>user_id, batch_size] --> P[Fetch up to batch_size memories<br/>WHERE expires_at IS NOT NULL<br/>AND expires_at <= NOW]
    P --> Q{For each candidate,<br/>Qwen reviews:<br/>content + importance + age}
    Q -->|DELETE| R[Hard delete from Neon<br/>+ purge Redis cache entry]
    Q -->|KEEP| S[Renew expires_at<br/>+7 days from now]
    S --> M
```

## Component Responsibilities

| Component | Responsibility |
|---|---|
| **frontend/index.html** | Chat UI, live memory panel, session management, delete controls |
| **agent.py** | Orchestrates chat turns: recall → prompt assembly → Qwen call → memory extraction → storage. Proxies delete requests. |
| **memory_api.py** | Owns all memory intelligence: importance scoring, embedding generation, semantic recall, context synthesis, deduplication, smart forgetting, CRUD |
| **Qwen Cloud (qwen-plus)** | Chat completions for: user-facing replies, importance scoring, context synthesis, memory extraction |
| **Qwen Cloud (text-embedding-v3)** | 1024-dimension embeddings for semantic search and duplicate detection |
| **Neon PostgreSQL + pgvector** | Persistent storage for memories and their vector embeddings; cosine similarity search |
| **Upstash Redis** | Short-term cache (1hr TTL) for recently stored memories |
| **Alibaba Cloud ECS** | Hosts both backend services (`memory_api.py`, `agent.py`) — see [`DEPLOYMENT.md`](./DEPLOYMENT.md) |

## Why This Design

**Two services instead of one** — `memory_api.py` is a standalone, reusable memory layer with its own API contract. `agent.py` is a thin orchestration layer on top. This separation means the memory system could plug into a different agent/chat layer without modification, and is independently testable.

**Recall ranks by `similarity × importance`** — rather than pure semantic similarity, so a highly important memory with moderate relevance can still outrank a trivial memory with high relevance.

**Two-step memory write (extract, then store)** — instead of storing the raw user message, `agent.py` asks Qwen to extract structured facts first. This produces cleaner, more reusable memories ("User prefers concise explanations") instead of noisy raw text ("yeah I guess I'd rather you keep it short tbh").

**Deduplication and conflict resolution at write time** — rather than blind dedup, candidate memories are ranked by a blended similarity+importance score and Qwen arbitrates whether a close match should be treated as a duplicate (reject), an update (overwrite the stale fact), or a genuinely new independent memory. This handles the common case where a user's stated preference changes over time ("I prefer Python" → "actually I've switched to Rust") without leaving stale, contradictory memories in the store.