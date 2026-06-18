import requests

memories = [
    {"content": "I am a Technical Program Manager with IoT and AI experience", "session_id": "session_001", "user_id": "nidhi"},
    {"content": "I dislike verbose responses and unnecessary corporate jargon", "session_id": "session_001", "user_id": "nidhi"},
    {"content": "I am participating in the Qwen Cloud hackathon building a MemoryAgent", "session_id": "session_002", "user_id": "nidhi"},
    {"content": "I prefer Python over JavaScript for backend work", "session_id": "session_002", "user_id": "nidhi"},
]

for m in memories:
    r = requests.post("http://localhost:8000/memory", json=m)
    data = r.json()
    print(f"stored: [{data['importance_score']}] {m['content'][:50]}")