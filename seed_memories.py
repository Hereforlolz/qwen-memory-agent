import os
import requests

SEED_USERNAME = "nidhi"
SEED_PASSWORD = os.getenv("SEED_USER_PASSWORD")
if not SEED_PASSWORD:
    SEED_PASSWORD = "seed-demo-password-123"
    print(f"[WARN] SEED_USER_PASSWORD not set — using a hardcoded demo password for {SEED_USERNAME!r}.")

memories = [
    {"content": "I am a Technical Program Manager with IoT and AI experience", "session_id": "session_001"},
    {"content": "I dislike verbose responses and unnecessary corporate jargon", "session_id": "session_001"},
    {"content": "I am participating in the Qwen Cloud hackathon building a MemoryAgent", "session_id": "session_002"},
    {"content": "I prefer Python over JavaScript for backend work", "session_id": "session_002"},
]

# Register on first run; on subsequent runs the username is already taken (409),
# so fall back to logging in instead.
auth_body = {"username": SEED_USERNAME, "password": SEED_PASSWORD}
r = requests.post("http://localhost:8000/auth/register", json=auth_body)
if r.status_code == 409:
    r = requests.post("http://localhost:8000/auth/login", json=auth_body)
if r.status_code != 200:
    raise SystemExit(f"Could not authenticate as {SEED_USERNAME!r}: {r.status_code} {r.text[:200]}")
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

for m in memories:
    r = requests.post("http://localhost:8000/memory", json=m, headers=headers)
    if r.status_code == 200:
        data = r.json()
        print(f"stored: [{data['importance_score']}] {m['content'][:50]}")
    elif r.status_code == 409:
        print(f"skipped (already seeded): {m['content'][:50]}")
    else:
        print(f"failed ({r.status_code}): {m['content'][:50]}")
