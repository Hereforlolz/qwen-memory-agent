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
    A->>M: POST /recall {query}
    M->>Q: embed(query)
    Q-->>M: embedding vector
    M->>N: cosine similarity search<br/>ranked by similarity × importance
    N-->>M: top-k memories
    M->>Q: synthesize context window
    Q-->>M: context summary
    M-->>A: {memories, context_window}

    A->>Q: chat completion<br/>(system prompt + memory context + history)
    Q-->>A: assistant reply

    A->>Q: extract structured memories<br/>from this turn
    Q-->>A: JSON array of facts

    loop for each extracted memory
        A->>M: POST /memory {content}
        M->>N: check duplicate (cosine similarity ≥ 0.92)
        alt not duplicate
            M->>Q: score importance (0–1)
            Q-->>M: score
            M->>Q: embed(content)
            Q-->>M: embedding vector
            M->>N: INSERT memory + embedding
            M->>R: cache memory (1hr TTL)
        else duplicate
            M-->>A: 409 Conflict (skipped)
        end
    end

    A-->>U: {reply, memories_used, context_injected}
```

## Memory Lifecycle

```mermaid
flowchart LR
    A[New memory content] --> B{Duplicate check<br/>cosine similarity ≥ 0.92?}
    B -->|Yes| C[Reject — 409]
    B -->|No| D[Qwen scores importance<br/>0.0 – 1.0]
    D --> E{Score range}
    E -->|< 0.3| F[TTL: 24 hours]
    E -->|0.3 – 0.6| G[TTL: 7 days]
    E -->|≥ 0.6| H[Permanent]
    F --> I[(Stored in Neon<br/>+ pgvector embedding)]
    G --> I
    H --> I
    I --> J[Cached in Redis<br/>1hr TTL]

    K[Smart Forget<br/>triggered] --> L[Fetch expired<br/>low-importance memories]
    L --> M[Qwen reviews:<br/>delete or keep?]
    M -->|delete| N[Removed from Neon]
    M -->|keep| I
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

**Deduplication at write time** — checks cosine similarity against existing memories before storing, preventing the same fact from being re-stored every session (a common failure mode in naive memory systems).
