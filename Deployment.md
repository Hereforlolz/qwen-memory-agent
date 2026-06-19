# Alibaba Cloud Deployment

This file documents the live deployment of Qwen MemoryAgent on Alibaba Cloud infrastructure, as required for hackathon submission eligibility.

## Infrastructure

| Item | Value |
|---|---|
| Service | Alibaba Cloud Elastic Compute Service (ECS) |
| Instance Type | Economy Type e — 2 vCPU / 2 GiB |
| OS | Ubuntu 22.04 64-bit |
| Region | US (Silicon Valley) |
| Billing | Pay-as-you-go |
| Instance ID | `i-rj9fpf5zw40qz7c306t0` |

## Services Running

Both backend services run directly on the ECS instance via `nohup` + `uvicorn`:

```bash
nohup python3 memory_api.py > memory_api.log 2>&1 &
nohup python3 agent.py > agent.log 2>&1 &
```

- `memory_api.py` → port `8000` — core memory CRUD, Qwen scoring, recall, forget
- `agent.py` → port `8001` — chat layer, memory injection, frontend host

## Verified Health Checks

Run from outside the instance (proving public reachability over Alibaba Cloud's network):

```
GET http://<ECS-public-ip>:8000/health
→ {"status":"healthy","timestamp":"2026-06-19T01:44:57.301718"}

GET http://<ECS-public-ip>:8001/health
→ {"status":"ok","service":"MemoryAgent Chat"}
```

## End-to-End Functional Test

A full chat turn was executed against the live Alibaba Cloud deployment, confirming:
- Qwen Cloud API connectivity (chat completion + embeddings) from the ECS instance
- Neon Postgres (pgvector) connectivity for memory storage/recall
- Upstash Redis connectivity for caching
- Memory recall and context injection working end-to-end on the deployed instance

```
POST http://<ECS-public-ip>:8001/chat
Body: {"user_id":"nidhi","message":"hello, testing new api key"}

Response (truncated):
{
  "reply": "Hello again, Nidhi! It looks like the new API key for your Qwen MemoryAgent project is working perfectly...",
  "context_injected": true,
  "memories_used": [ ...5 memories recalled via pgvector similarity search... ]
}
```

## Security Group Configuration

Inbound rules opened on the ECS security group to allow external access:

| Port | Protocol | Source | Purpose |
|---|---|---|---|
| 22 | TCP | 0.0.0.0/0 | SSH management |
| 8000 | TCP | 0.0.0.0/0 | memory_api.py |
| 8001 | TCP | 0.0.0.0/0 | agent.py + frontend |

## Architecture

See [`architecture.md`](./architecture.md) for the full system diagram showing how Qwen Cloud, Alibaba Cloud ECS, Neon, and Upstash connect.

## Notes

- All credentials (Qwen API key, database URL, Redis URL) are stored in a `.env` file on the instance, excluded from version control via `.gitignore`.
- Credentials used during development/testing were rotated prior to final submission as a security precaution.
- Services are configured to auto-restart detached from the SSH session via `nohup`, ensuring uptime independent of any single terminal connection.