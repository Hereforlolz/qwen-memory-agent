"""
Qwen MemoryAgent - Core Memory API
Track 1: MemoryAgent - Global AI Hackathon with Qwen Cloud
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncpg
import redis.asyncio as aioredis
import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# QWEN CLIENT
# =============================================================================

qwen = AsyncOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
)

QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.7-plus")
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

# =============================================================================
# MODELS
# =============================================================================

class MemoryCreate(BaseModel):
    content: str = Field(..., description="The memory content in natural language")
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(default="default_user")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MemoryResponse(BaseModel):
    id: str
    content: str
    session_id: str
    user_id: str
    importance_score: float
    embedding_preview: List[float]
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any]

class RecallRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    top_k: int = Field(default=5, le=20)

class RecallResponse(BaseModel):
    memories: List[Dict[str, Any]]
    context_window: str

# =============================================================================
# DATABASE
# =============================================================================

class DB:
    pool: asyncpg.Pool = None
    redis: aioredis.Redis = None

db = DB()

async def init_db():
    db.pool = await asyncpg.create_pool(
        os.getenv("DATABASE_URL"),
        min_size=2,
        max_size=10
    )
    db.redis = aioredis.from_url(
        os.getenv("REDIS_URL"),
        decode_responses=True
    )
    async with db.pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1024),
                importance_score FLOAT DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE,
                is_compressed BOOLEAN DEFAULT FALSE,
                metadata JSONB DEFAULT '{}'
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_user
            ON memories(user_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_importance
            ON memories(importance_score DESC)
        """)
    logger.info("DB initialized")

async def close_db():
    if db.pool:
        await db.pool.close()
    if db.redis:
        await db.redis.aclose()

# =============================================================================
# QWEN INTELLIGENCE LAYER
# =============================================================================

async def score_importance(content: str) -> float:
    """Ask Qwen to rate how important this memory is (0.0 - 1.0)"""
    try:
        resp = await qwen.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a memory importance scorer. "
                        "Rate the importance of this memory for future recall on a scale of 0.0 to 1.0. "
                        "High importance: preferences, facts about the user, decisions, goals, problems. "
                        "Low importance: greetings, filler, redundant info. "
                        "Reply with ONLY a float like 0.7 — nothing else."
                    )
                },
                {"role": "user", "content": f"Memory to score: {content}"}
            ],
            max_tokens=10,
            extra_body={"enable_thinking": False}
        )
        score = float(resp.choices[0].message.content.strip())
        return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning(f"Importance scoring failed: {e}, defaulting to 0.5")
        return 0.5

async def get_embedding(text: str) -> List[float]:
    """Get text embedding from Qwen"""
    resp = await qwen.embeddings.create(
        model=QWEN_EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding

async def build_context_window(memories: List[Dict], query: str) -> str:
    """Ask Qwen to synthesize recalled memories into coherent context"""
    if not memories:
        return "No relevant memories found."

    memory_text = "\n".join([
        f"- [{m['created_at']}] (importance: {m['importance_score']:.2f}): {m['content']}"
        for m in memories
    ])

    resp = await qwen.chat.completions.create(
        model=QWEN_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You synthesize past memories into a concise context summary. "
                    "Given a query and relevant memories, write a brief paragraph "
                    "that captures what's most relevant for answering the query. "
                    "Be concise — max 3 sentences."
                )
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nRelevant memories:\n{memory_text}"
            }
        ],
        max_tokens=200,
        extra_body={"enable_thinking": False}
    )
    return resp.choices[0].message.content.strip()

# =============================================================================
# SMART FORGET
# =============================================================================

async def smart_forget():
    """Qwen decides which low-importance memories to compress or delete"""
    async with db.pool.acquire() as conn:
        old_memories = await conn.fetch("""
            SELECT id, content, importance_score, access_count, created_at
            FROM memories
            WHERE importance_score < 0.3
            AND created_at < NOW() - INTERVAL '1 hour'
            AND is_compressed = FALSE
            ORDER BY importance_score ASC
            LIMIT 20
        """)

        if not old_memories:
            return {"deleted": 0, "message": "No memories to forget"}

        memory_list = [dict(m) for m in old_memories]
        memory_text = "\n".join([
            f"ID:{m['id']} SCORE:{m['importance_score']:.2f} CONTENT:{m['content']}"
            for m in memory_list
        ])

        resp = await qwen.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a memory manager. Review these low-importance memories "
                        "and decide: DELETE (truly useless) or KEEP (might still matter). "
                        "Reply with JSON only: {\"delete\": [\"id1\", \"id2\"], \"keep\": [\"id3\"]}"
                    )
                },
                {"role": "user", "content": memory_text}
            ],
            max_tokens=500,
            extra_body={"enable_thinking": False}
        )

        try:
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            decision = json.loads(raw)
            delete_ids = decision.get("delete", [])

            deleted = 0
            for mid in delete_ids:
                await conn.execute(
                    "DELETE FROM memories WHERE id = $1",
                    uuid.UUID(str(mid))
                )
                deleted += 1

            logger.info(f"Smart forget: deleted {deleted} memories")
            return {"deleted": deleted}
        except Exception as e:
            logger.warning(f"Smart forget parse error: {e}")
            return {"deleted": 0}

# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()

app = FastAPI(
    title="Qwen MemoryAgent API",
    description="Persistent memory layer powered by Qwen Cloud — Track 1 submission",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {
        "service": "Qwen MemoryAgent",
        "track": "Track 1 - MemoryAgent",
        "hackathon": "Global AI Hackathon with Qwen Cloud",
        "status": "running"
    }

@app.get("/health")
async def health():
    async with db.pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    await db.redis.ping()
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/memory", response_model=MemoryResponse)
async def store_memory(entry: MemoryCreate):
    """Store a memory — Qwen scores importance + generates embedding"""
    importance, embedding = await asyncio.gather(
        score_importance(entry.content),
        get_embedding(entry.content)
    )

    if importance < 0.3:
        expires_at = datetime.utcnow() + timedelta(hours=24)
    elif importance < 0.6:
        expires_at = datetime.utcnow() + timedelta(days=7)
    else:
        expires_at = None

    embedding_str = f"[{','.join(str(x) for x in embedding)}]"

    async with db.pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO memories
                (user_id, session_id, content, embedding, importance_score, expires_at, metadata)
            VALUES ($1, $2, $3, $4::vector, $5, $6, $7)
            RETURNING *
        """,
        entry.user_id,
        entry.session_id,
        entry.content,
        embedding_str,
        importance,
        expires_at,
        json.dumps(entry.metadata)
        )

    await db.redis.setex(
        f"memory:{row['id']}",
        3600,
        json.dumps({"content": entry.content, "importance": importance})
    )

    return MemoryResponse(
        id=str(row['id']),
        content=row['content'],
        session_id=row['session_id'],
        user_id=row['user_id'],
        importance_score=row['importance_score'],
        embedding_preview=embedding[:5],
        created_at=row['created_at'],
        expires_at=row['expires_at'],
        metadata=json.loads(row['metadata'])
    )

@app.post("/recall", response_model=RecallResponse)
async def recall_memories(request: RecallRequest):
    """Semantic search — find relevant memories + build context window"""
    query_embedding = await get_embedding(request.query)
    embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

    async with db.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, importance_score, created_at, session_id,
                   1 - (embedding <=> $1::vector) as similarity
            FROM memories
            WHERE user_id = $2
            AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY (1 - (embedding <=> $1::vector)) * importance_score DESC
            LIMIT $3
        """, embedding_str, request.user_id, request.top_k)

    memories = []
    for row in rows:
        memories.append({
            "id": str(row['id']),
            "content": row['content'],
            "importance_score": row['importance_score'],
            "similarity": float(row['similarity']),
            "session_id": row['session_id'],
            "created_at": row['created_at'].isoformat()
        })
        async with db.pool.acquire() as conn:
            await conn.execute("""
                UPDATE memories SET access_count = access_count + 1,
                last_accessed = NOW() WHERE id = $1
            """, row['id'])

    context = await build_context_window(memories, request.query)
    return RecallResponse(memories=memories, context_window=context)

@app.delete("/forget")
async def trigger_smart_forget():
    """Manually trigger Qwen-powered smart forgetting"""
    result = await smart_forget()
    return result

@app.get("/memories/{user_id}")
async def list_memories(user_id: str, limit: int = 20):
    """List all memories for a user, sorted by importance"""
    async with db.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, importance_score, session_id,
                   created_at, expires_at, access_count
            FROM memories
            WHERE user_id = $1
            AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY importance_score DESC
            LIMIT $2
        """, user_id, limit)

    return [dict(r) for r in rows]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("memory_api:app", host="0.0.0.0", port=8000, reload=True)