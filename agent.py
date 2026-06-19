"""
agent.py — MemoryAgent chat layer
Track 1: MemoryAgent — Global AI Hackathon with Qwen Cloud

- Injects recalled memory context into every Qwen call
- Stores each user turn as a new memory after responding
- Exposes /chat + /chat/memories/{user_id} for the frontend
- Also runs as a CLI loop: python agent.py cli
"""
import json
import os
import uuid
import asyncio
import httpx
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")
MEMORY_API_URL = os.getenv("MEMORY_API_URL", "http://localhost:8000")

qwen = AsyncOpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

app = FastAPI(title="MemoryAgent Chat", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── memory API calls ──────────────────────────────────────────────────────────

async def recall_context(user_id: str, query: str) -> tuple[str, list[dict]]:
    """Returns (context_window string, raw memories list)"""
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{MEMORY_API_URL}/recall",
                json={"user_id": user_id, "query": query, "top_k": 10},
            )
            if resp.status_code != 200:
                return "", []
            data = resp.json()
            return data.get("context_window", ""), data.get("memories", [])
        except Exception as e:
            print(f"[ERROR] Recall communication failed: {e}")
            return "", []


async def store_memory(user_id: str, session_id: str, content: str, metadata: dict = None) -> dict:
    """Store a memory — handles 409 duplicate gracefully"""
    if metadata is None:
        metadata = {"source": "chat_agent"}

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(
                f"{MEMORY_API_URL}/memory",
                json={
                    "user_id": user_id,
                    "session_id": session_id,
                    "content": content,
                    "metadata": metadata,
                },
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 409:
                print(f"[INFO] Duplicate skipped: {content[:60]}")
                return {"status": "duplicate", "skipped": True}
            else:
                print(f"[WARN] Store failed: {resp.status_code}")
                return {}
        except Exception as e:
            print(f"[ERROR] Store memory failed: {e}")
            return {}


async def list_memories(user_id: str, limit: int = 20) -> list[dict]:
    """API returns a plain list"""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{MEMORY_API_URL}/memories/{user_id}",
                params={"limit": limit},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            print(f"[ERROR] List memories failed: {e}")
            return []


# ── core chat turn ────────────────────────────────────────────────────────────

async def chat_turn(
    user_id: str,
    session_id: str,
    user_message: str,
    conversation_history: list[dict],
) -> dict:
    """Full turn: recall → prompt → Qwen → extract → store"""

    # 1. recall
    context_str, memories_used = await recall_context(user_id, user_message)

    # 2. system prompt
    if context_str.strip():
        memory_block = f"=== RELEVANT MEMORIES FROM PAST SESSIONS ===\n{context_str}\n============================================="
    else:
        memory_block = "(No relevant memories found for this query.)"

    system_prompt = f"""You are a helpful AI assistant with persistent memory across sessions.
You remember what users tell you — their preferences, goals, problems, and facts about their life.
Use the memory context below to give personalized, context-aware responses.
Reference memories naturally when relevant. Don't force it.

{memory_block}"""

    # 3. messages
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history[-10:])
    messages.append({"role": "user", "content": user_message})

    # 4. Qwen chat call
    try:
        response = await qwen.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            extra_body={"enable_thinking": False},
        )
        assistant_reply = response.choices[0].message.content
    except Exception as e:
        assistant_reply = "Sorry, I encountered an error while thinking."
        print(f"[ERROR] Qwen call failed: {e}")

    # 5. extract structured memories from this turn
    stored_pipeline_results = []
    try:
        extract_prompt = f"""Extract important memories from this conversation turn as a list of concise statements.
        Focus on facts, user preferences, goals, project details, personal info, and action items.
        CRITICAL: Never extract negative facts or the absence of information (e.g., "User does not have a car" or "User has no recorded history of X"). 
        If the user doesn't state a fact, do not generate a memory row for it.

User: {user_message}
Assistant: {assistant_reply}

Return only a JSON array of strings. Example: ["Nidhi is working on Qwen MemoryAgent hackathon", "She prefers async FastAPI", "Project deadline is July 9 2026"]"""

        extract_resp = await qwen.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{"role": "user", "content": extract_prompt}],
            extra_body={"enable_thinking": False},
            max_tokens=300,
        )

        extracted_text = extract_resp.choices[0].message.content.strip()
        # strip markdown fences if Qwen wraps in ```json
        extracted_text = extracted_text.replace("```json", "").replace("```", "").strip()

        try:
            memories_list = json.loads(extracted_text)
            if isinstance(memories_list, list):
                for mem in memories_list[:3]:
                    save_res = await store_memory(
                        user_id,
                        session_id,
                        str(mem),
                        metadata={"source": "chat_agent", "type": "extracted"}
                    )
                    if save_res and "id" in save_res:
                        stored_pipeline_results.append(save_res)
        except Exception:
            # fallback: store summary
            summary = f"User: {user_message[:200]} | Assistant: {assistant_reply[:200]}"
            save_res = await store_memory(user_id, session_id, summary)
            if save_res and "id" in save_res:
                stored_pipeline_results.append(save_res)

    except Exception as e:
        # fallback: store raw user message
        save_res = await store_memory(user_id, session_id, f"User said: {user_message}")
        if save_res and "id" in save_res:
            stored_pipeline_results.append(save_res)
        print(f"[WARN] Memory extraction failed, stored basic memory: {e}")

    return {
        "reply": assistant_reply,
        "memories_used": memories_used,
        "memories_stored": stored_pipeline_results,
        "context_injected": bool(context_str.strip()),
    }


# ── FastAPI endpoints ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    session_id: str = ""
    message: str
    conversation_history: list[dict] = []


class ChatResponse(BaseModel):
    reply: str
    memories_used: list
    memories_stored: list
    context_injected: bool
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    session_id = req.session_id or str(uuid.uuid4())
    result = await chat_turn(req.user_id, session_id, req.message, req.conversation_history)
    return {**result, "session_id": session_id}


@app.get("/chat/memories/{user_id}")
async def get_user_memories(user_id: str, limit: int = 20):
    memories = await list_memories(user_id, limit)
    return {"user_id": user_id, "memories": memories}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "MemoryAgent Chat"}


# ── proxy delete endpoints to memory_api ─────────────────────────────────────

@app.delete("/memory/{memory_id}")
async def delete_memory_proxy(memory_id: str):
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.delete(f"{MEMORY_API_URL}/memory/{memory_id}")
            if resp.status_code == 404:
                raise HTTPException(status_code=404, detail="Memory not found")
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail="Memory service error")


@app.delete("/memories/{user_id}")
async def delete_all_memories_proxy(user_id: str):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.delete(f"{MEMORY_API_URL}/memories/{user_id}")
        return resp.json()


# ── serve frontend ────────────────────────────────────────────────────────────

frontend_path = os.path.join(os.path.dirname(__file__), "frontend")
print(f"[DEBUG] Frontend path: {frontend_path}")
print(f"[DEBUG] Frontend exists: {os.path.exists(frontend_path)}")

try:
    app.mount("/app", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print("[DEBUG] Frontend mounted at /app")
except Exception as e:
    print(f"[ERROR] Frontend mount failed: {e}")


# ── CLI loop ──────────────────────────────────────────────────────────────────

async def cli_loop():
    user_id = input("Enter user ID (e.g. 'nidhi'): ").strip() or "default_user"
    session_id = str(uuid.uuid4())
    print(f"\n[MemoryAgent] Session '{session_id[:8]}...' for '{user_id}'")
    print("[MemoryAgent] Type 'quit' to exit, 'memories' to list stored memories.\n")

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[MemoryAgent] Session ended.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("[MemoryAgent] Goodbye!")
            break

        if user_input.lower() == "memories":
            mems = await list_memories(user_id)
            if not mems:
                print("  (no memories stored yet)\n")
            else:
                for m in mems:
                    exp = m.get("expires_at") or "permanent"
                    print(f"  [{m['importance_score']:.2f}] {m['content'][:80]}  ({exp})")
            print()
            continue

        if not user_input:
            continue

        result = await chat_turn(user_id, session_id, user_input, conversation_history)

        print(f"\nAssistant: {result['reply']}")
        if result["context_injected"]:
            print(f"  [memory] injected {len(result['memories_used'])} memories")
        print("  [memory] new memories extracted and stored")
        print()

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": result["reply"]})


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        asyncio.run(cli_loop())
    else:
        import uvicorn
        print("[MemoryAgent] API on :8001 — frontend at http://localhost:8001/app")
        print("[MemoryAgent] Use 'python agent.py cli' for terminal chat")
        uvicorn.run(app, host="0.0.0.0", port=8001)