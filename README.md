# Qwen MemoryAgent

Track 1: MemoryAgent — Global AI Hackathon with Qwen Cloud

A persistent memory layer where Qwen actively manages what to remember, forget, and recall.

## What it does
- Stores memories with Qwen-scored importance (0-1)
- Semantic search via pgvector embeddings
- Smart forgetting — Qwen decides what to delete, not just TTL
- Context window synthesis — Qwen summarizes relevant memories into prompt-ready context

## Stack
- FastAPI + asyncpg + Neon (PostgreSQL + pgvector)
- Redis (Upstash)
- Qwen Cloud API (qwen3.7-plus + text-embedding-v3)

## Run locally
1. Clone repo
2. Copy `.env.example` to `.env` and fill in keys
3. `pip install -r requirements.txt`
4. `python memory_api.py`

## API
- `POST /memory` — store a memory
- `POST /recall` — semantic search + context synthesis
- `DELETE /forget` — trigger smart forgetting
- `GET /memories/{user_id}` — list all memories