"""
memory_api.py — Core Memory Storage & Retrieval API
Track 1: MemoryAgent — Global AI Hackathon with Qwen Cloud

Manages the PostgreSQL Vector Database connection pool, 
handles similarity recall scoring, and resolves duplicate conflicts.
"""
import os
import logging
import json
import uuid
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Setup logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory_api")

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
DATABASE_URL = os.getenv("DATABASE_URL")

try:
    import asyncpg
except ImportError:
    logger.error("asyncpg is required for this backend database. Run: pip install asyncpg")

try:
    import redis.asyncio as aioredis
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
except ImportError:
    aioredis = None
    REDIS_URL = None

qwen = AsyncOpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
app = FastAPI(title="MemoryAgent Vector Core", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DatabaseManager:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.redis: Optional[aioredis.Redis] = None

    async def initialize(self):
        try:
            self.pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=10,
                timeout=30.0
            )
            logger.info("[DB] Connected to PostgreSQL Connection Pool successfully.")

            async with self.pool.acquire() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id VARCHAR(100) NOT NULL,
                        session_id VARCHAR(100) NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector(1024), 
                        importance_score NUMERIC(3,2) DEFAULT 0.5,
                        created_at TIMESTAMP DEFAULT NOW(),
                        expires_at TIMESTAMP NULL,
                        metadata JSONB DEFAULT '{}'::jsonb
                    );
                    CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
                """)
                logger.info("[DB] Table structure verified.")
        except Exception as e:
            logger.critical(f"[DB] Initialization Error: {e}")
            raise e

        if aioredis and REDIS_URL:
            try:
                self.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
                logger.info("[Redis] Memory cache layer connected.")
            except Exception as e:
                logger.warning(f"[Redis] Skipping cache layer initialization: {e}")

    async def close(self):
        if self.pool:
            await self.pool.close()
        if self.redis:
            await self.redis.close()

db = DatabaseManager()

@app.on_event("startup")
async def startup():
    await db.initialize()

@app.on_event("shutdown")
async def shutdown():
    await db.close()

class MemoryCreate(BaseModel):
    user_id: str
    session_id: str
    content: str
    metadata: dict = Field(default_factory=dict)

class MemoryResponse(BaseModel):
    id: str
    content: str
    session_id: str
    user_id: str
    importance_score: float
    embedding_preview: List[float]
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: dict

class RecallRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 5

class RecallResponse(BaseModel):
    context_window: str
    memories: List[dict]


@app.get("/health")
async def health():
    """Basic liveness check — confirms DB pool and Redis are reachable."""
    async with db.pool.acquire() as conn:
        await conn.fetchval("SELECT 1")
    if db.redis:
        await db.redis.ping()
    return {"status": "healthy"}


async def get_embedding(text: str) -> List[float]:
    """Fetch raw embeddings matrix dimension map from Qwen Cloud layer."""
    try:
        resp = await qwen.embeddings.create(
            model="text-embedding-v3",
            input=text
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"[Embedding Error] Failed parsing text coordinates: {e}")
        raise HTTPException(status_code=502, detail="Upstream embedding generation failure.")


async def score_importance(content: str) -> float:
    """Asks Qwen to assign an analytical importance weight score between 0.0 and 1.0."""
    try:
        prompt = f"""Rate the long-term importance of this user memory statement on a scale from 0.00 to 1.00.
0.0 = completely trivial greeting or boilerplate conversation ("Hello", "Goodbye").
0.5 = minor context ("User prefers Python over JavaScript").
1.0 = absolute core profile defining trait, critical project detail, or deadline constraint.

Statement: "{content}"
Return only a floating point value."""

        resp = await qwen.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            extra_body={"enable_thinking": False}
        )
        val = float(resp.choices[0].message.content.strip())
        return min(max(val, 0.0), 1.0)
    except Exception:
        return 0.5


async def manage_duplicates_and_conflicts(content: str, user_id: str, embedding: List[float]) -> Optional[str]:
    """
    Checks the user's historical graph for direct matches or conflict states.
    Returns:
       - "SKIP" if it's an identical redundant fact.
       - A valid string `UUID` of an old row if it needs an explicit overwrite update.
       - None if this is a fresh unique statement.
    """
    embedding_str = f"[{','.join(str(x) for x in embedding)}]"
    async with db.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, 1 - (embedding <=> $1::vector) as similarity
            FROM memories
            WHERE user_id = $2
            ORDER BY ((1 - (embedding <=> $1::vector)) * 0.60) + (importance_score * 0.40) DESC
            LIMIT 3
        """, embedding_str, user_id)

        for row in rows:
            sim = row['similarity']
            if sim > 0.96 or row['content'].strip().lower() == content.strip().lower():
                return "SKIP"

            if sim > 0.82:
                try:
                    prompt = f"""Compare these statements from the same user:
Old Saved Memory: "{row['content']}"
New Input Fact: "{content}"

Does the New Input explicitly correct, modify, update, or contradict the Old Saved Memory? 
Reply 'UPDATE' if the old memory is stale or overwritten. 
Reply 'NEW' if both statements are distinctly valid and independent context notes.
Return ONLY 'UPDATE' or 'NEW'."""

                    res = await qwen.chat.completions.create(
                        model="qwen-plus",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=10,
                        extra_body={"enable_thinking": False}
                    )
                    verdict = res.choices[0].message.content.strip().upper()
                    if "UPDATE" in verdict:
                        return str(row['id'])
                except Exception as e:
                    logger.error(f"[Triage System Error] Resolution skipped: {e}")

        return None


@app.post("/memory", response_model=MemoryResponse)
async def store_memory(entry: MemoryCreate):
    """Store a memory — Qwen scores importance + generates embedding. Skips duplicates cleanly."""
    try:
        embedding = await get_embedding(entry.content)
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        status = await manage_duplicates_and_conflicts(entry.content, entry.user_id, embedding)

        if status == "SKIP":
            logger.info(f"Rejecting redundant memory: {entry.content[:60]}")
            raise HTTPException(status_code=409, detail="Duplicate memory exists")

        importance = await score_importance(entry.content)

        if importance < 0.3:
            expires_at = datetime.utcnow() + timedelta(hours=24)
        elif importance < 0.6:
            expires_at = datetime.utcnow() + timedelta(days=7)
        else:
            expires_at = None

        async with db.pool.acquire() as conn:
            if status and status != "SKIP":
                logger.info(f"Overwriting stale memory row: {status}")
                row = await conn.fetchrow("""
                    UPDATE memories
                    SET content = $1, embedding = $2::vector, importance_score = $3, expires_at = $4, session_id = $5, created_at = NOW()
                    WHERE id = $6
                    RETURNING *
                """, entry.content, embedding_str, importance, expires_at, entry.session_id, uuid.UUID(status))
                if db.redis:
                    await db.redis.delete(f"memory:{status}")
            else:
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

        if db.redis and row:
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
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"CRITICAL PIPELINE ERROR IN STORE_MEMORY: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recall", response_model=RecallResponse)
async def recall_memories(request: RecallRequest):
    """Retrieve chronologically weighted and semantically close memories isolated strictly by user_id."""
    try:
        embedding = await get_embedding(request.query)
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        async with db.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, content, importance_score, created_at, session_id,
                       1 - (embedding <=> $1::vector) as similarity
                FROM memories
                WHERE user_id = $2
                AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY importance_score DESC, (1 - (embedding <=> $1::vector)) DESC
                LIMIT $3
            """, embedding_str, request.user_id, request.top_k)

        memories_output = []
        context_blocks = []
        for r in rows:
            m = {
                "id": str(r['id']),
                "content": r['content'],
                "importance_score": float(r['importance_score']),
                "similarity": float(r['similarity']),
                "created_at": r['created_at'].isoformat()
            }
            memories_output.append(m)
            context_blocks.append(f"- {r['content']} (Recorded: {r['created_at'].strftime('%Y-%m-%d')})")

        return RecallResponse(
            context_window="\n".join(context_blocks),
            memories=memories_output
        )
    except Exception as e:
        logger.error(f"Recall engine error: {e}")
        return RecallResponse(context_window="", memories=[])


@app.get("/memories/{user_id}")
async def list_user_memories(user_id: str, limit: int = 30):
    """Exposes all active memory schemas bound to a specific user context."""
    async with db.pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, content, importance_score, created_at, expires_at, metadata
            FROM memories
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """, user_id, limit)

        return [
            {
                "id": str(r['id']),
                "content": r['content'],
                "importance_score": float(r['importance_score']),
                "created_at": r['created_at'].isoformat(),
                "expires_at": r['expires_at'].isoformat() if r['expires_at'] else None,
                "metadata": json.loads(r['metadata']) if isinstance(r['metadata'], str) else r['metadata']
            }
            for r in rows
        ]


@app.delete("/memory/{memory_id}")
async def delete_single_memory(memory_id: str):
    """Remove a specific targeted fact vector across storage partitions."""
    try:
        target_uuid = uuid.UUID(memory_id)
        async with db.pool.acquire() as conn:
            res = await conn.execute("DELETE FROM memories WHERE id = $1", target_uuid)
            if "DELETE 0" in res:
                raise HTTPException(status_code=404, detail="Memory key not found.")
        if db.redis:
            await db.redis.delete(f"memory:{memory_id}")
        return {"status": "deleted", "id": memory_id}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid UUID token string format.")


@app.delete("/memories/{user_id}")
async def flush_user_vault(user_id: str):
    """Full hard drop of every data vector entry assigned to a single clean context user partition."""
    async with db.pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE user_id = $1", user_id)
    return {"status": "cleared", "user_id": user_id}

if __name__ == "__main__":
    import uvicorn
    print("[Vector Core] Spawning server layer on port :8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)