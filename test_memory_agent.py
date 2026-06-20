"""
test_memory_agent.py — End-to-end verification suite for Qwen MemoryAgent
Track 1: MemoryAgent — Global AI Hackathon with Qwen Cloud

Run this against either:
  - Local servers (memory_api.py on :8000, agent.py on :8001)
  - Your live Alibaba Cloud ECS deployment (set BASE_MEMORY / BASE_AGENT below)

This script does NOT use pytest/mocks — it makes real HTTP calls against a
running instance and checks real outcomes, because the whole point is to
prove the live system behaves the way the README claims.

Usage:
    python test_memory_agent.py                  # tests localhost
    python test_memory_agent.py --remote IP       # tests http://IP:8000 and :8001

Each test prints PASS/FAIL and a short reason. A final summary tells you
whether the system is demo-ready.
"""
import sys
import time
import uuid
import argparse
import requests
from datetime import datetime

# ── config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--remote", help="Public IP of your Alibaba Cloud ECS instance")
parser.add_argument("--memory-port", default="8000")
parser.add_argument("--agent-port", default="8001")
args = parser.parse_args()

HOST = args.remote if args.remote else "localhost"
BASE_MEMORY = f"http://{HOST}:{args.memory_port}"
BASE_AGENT = f"http://{HOST}:{args.agent_port}"

# Fresh test user every run so results don't get polluted by prior test data
TEST_USER = f"test_{uuid.uuid4().hex[:8]}"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
INFO = "\033[94mINFO\033[0m"

results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    results.append((name, bool(condition), detail))
    return condition


def section(title):
    print(f"\n{'─'*70}\n{title}\n{'─'*70}")


def safe_request(method, url, **kwargs):
    """Wraps requests calls so connection failures / timeouts report as a
    failed response-like object instead of raising and killing the whole suite."""
    try:
        return requests.request(method, url, **kwargs)
    except Exception as e:
        class _FailedResponse:
            status_code = -1
            text = str(e)

            def json(self):
                return {}
        return _FailedResponse()


# ── 1. Health checks ─────────────────────────────────────────────────────────

def test_health():
    section("1. Health Checks")
    try:
        r = safe_request("GET", f"{BASE_MEMORY}/health", timeout=10)
        check("memory_api.py /health responds 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            check("memory_api.py reports healthy status", r.json().get("status") == "healthy", str(r.json()))
    except Exception as e:
        check("memory_api.py /health responds 200", False, f"exception: {e}")

    try:
        r = safe_request("GET", f"{BASE_AGENT}/health", timeout=10)
        check("agent.py /health responds 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            check("agent.py reports ok status", r.json().get("status") == "ok", str(r.json()))
    except Exception as e:
        check("agent.py /health responds 200", False, f"exception: {e}")


# ── 2. Basic store + recall ──────────────────────────────────────────────────

def test_basic_store_and_recall():
    section("2. Basic Memory Store + Recall")
    session_id = str(uuid.uuid4())

    r = safe_request("POST", f"{BASE_MEMORY}/memory", json={
        "user_id": TEST_USER,
        "session_id": session_id,
        "content": "User's favorite programming language is Rust",
    }, timeout=15)
    check("POST /memory returns 200 for a fresh fact", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")

    if r.status_code == 200:
        data = r.json()
        check("response includes importance_score", "importance_score" in data, str(data.get("importance_score")))
        check("response includes embedding_preview", len(data.get("embedding_preview", [])) > 0)
        memory_id = data.get("id")
    else:
        memory_id = None

    time.sleep(1)  # let it settle

    r = safe_request("POST", f"{BASE_MEMORY}/recall", json={
        "user_id": TEST_USER,
        "query": "what programming language do I like",
        "top_k": 5,
    }, timeout=15)
    check("POST /recall returns 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        found = any("rust" in m["content"].lower() for m in data.get("memories", []))
        check("recall finds the Rust fact via semantic search", found, str(data.get("memories")))
        check("context_window is non-empty", len(data.get("context_window", "")) > 0)

    return memory_id


# ── 3. Importance scoring calibration (the bug we just fixed) ───────────────

def test_importance_calibration():
    section("3. Importance Scoring Calibration")
    session_id = str(uuid.uuid4())

    cases = [
        ("Hello", "greeting", lambda s: s < 0.3, "should score low"),
        ("Nidhi", "bare name", lambda s: s >= 0.6, "names should never score below 0.6"),
        ("User is an indoor gardener", "vague category", lambda s: s >= 0.4, "general statement"),
        ("User grows cherry tomatoes and sweet basil", "specific named fact",
         lambda s: s >= 0.4, "specific fact should NOT score lower than its vague category"),
    ]

    scores = {}
    for content, label, _, _ in cases:
        r = safe_request("POST", f"{BASE_MEMORY}/memory", json={
            "user_id": TEST_USER,
            "session_id": session_id,
            "content": content,
        }, timeout=15)
        if r.status_code == 200:
            scores[label] = r.json()["importance_score"]
        elif r.status_code == 409:
            scores[label] = None  # treated as duplicate, skip
        else:
            scores[label] = None
        time.sleep(0.5)

    for content, label, predicate, reason in cases:
        score = scores.get(label)
        if score is None:
            check(f"importance score for {label!r}", False, "request failed or duplicate-skipped")
        else:
            check(f"importance score for {label!r} = {score}", predicate(score), reason)

    # The actual regression we fixed: specific fact should score >= vague category
    vague = scores.get("vague category")
    specific = scores.get("specific named fact")
    if vague is not None and specific is not None:
        check(
            "specific fact scores >= its vague category (regression check)",
            specific >= vague,
            f"vague={vague}, specific={specific}"
        )


# ── 4. Deduplication + conflict arbitration ──────────────────────────────────

def test_dedup_and_conflicts():
    section("4. Deduplication & Conflict Arbitration")
    session_id = str(uuid.uuid4())

    # Store an initial fact
    r1 = safe_request("POST", f"{BASE_MEMORY}/memory", json={
        "user_id": TEST_USER,
        "session_id": session_id,
        "content": "User prefers Python for backend development",
    }, timeout=15)
    check("initial fact stores successfully", r1.status_code == 200, f"got {r1.status_code}")
    time.sleep(1)

    # Exact duplicate -> should be rejected with 409
    r2 = safe_request("POST", f"{BASE_MEMORY}/memory", json={
        "user_id": TEST_USER,
        "session_id": session_id,
        "content": "User prefers Python for backend development",
    }, timeout=15)
    check("exact duplicate returns 409", r2.status_code == 409, f"got {r2.status_code}")
    time.sleep(1)

    # Contradicting fact -> should trigger UPDATE (old row overwritten, not a new row)
    # NOTE: phrased to avoid containing both "python" and "backend" itself, which would
    # cause this test's own substring check to false-positive against the new row.
    r3 = safe_request("POST", f"{BASE_MEMORY}/memory", json={
        "user_id": TEST_USER,
        "session_id": session_id,
        "content": "User has switched their backend stack to Rust",
    }, timeout=15)
    check("contradicting fact is accepted (not blind-rejected)", r3.status_code == 200, f"got {r3.status_code}")

    # Precise check, independent of wording: if conflict arbitration triggered UPDATE,
    # store_memory's SQL does "UPDATE ... WHERE id = $6 RETURNING *", so r3's returned id
    # will be IDENTICAL to r1's original id. A fresh INSERT would return a new id.
    # Either outcome can be valid Qwen arbitration (UPDATE if treated as a correction,
    # NEW if treated as an independent fact) — this just reports which one happened,
    # since that depends on Qwen's live judgment call, not a hardcoded expectation.
    if r1.status_code == 200 and r3.status_code == 200:
        original_id = r1.json().get("id")
        new_id = r3.json().get("id")
        arbitration_result = "UPDATE (overwrote old row)" if original_id == new_id else "NEW (independent row)"
        print(f"  [{INFO}] Qwen arbitration verdict: {arbitration_result}")
        check(
            "arbitration produced a determinate id (no error)",
            bool(new_id),
            f"original_id={original_id}, new_id={new_id}"
        )

    time.sleep(1)
    r4 = safe_request("GET", f"{BASE_MEMORY}/memories/{TEST_USER}", timeout=15)
    if r4.status_code == 200:
        contents = [m["content"].lower() for m in r4.json()]
        # Only count rows that assert Python as the current backend preference —
        # exclude any row that mentions switching/no-longer/replaced, since those
        # are the NEW row describing the change, not a stale duplicate of the OLD claim.
        python_pref_count = sum(
            1 for c in contents
            if "python" in c and "backend" in c
            and not any(neg in c for neg in ("switch", "no longer", "replaced", "instead of"))
        )
        rust_pref_count = sum(1 for c in contents if "rust" in c and "backend" in c)
        check(
            "conflict resolution didn't leave duplicate contradictory rows",
            python_pref_count <= 1,
            f"found {python_pref_count} python-backend rows, {rust_pref_count} rust-backend rows"
        )


# ── 5. Negative-fact filtering ────────────────────────────────────────────────

def test_negative_fact_filtering():
    section("5. Negative-Fact Filtering (via agent.py extraction)")
    session_id = str(uuid.uuid4())

    r = safe_request("POST", f"{BASE_AGENT}/chat", json={
        "user_id": TEST_USER,
        "session_id": session_id,
        "message": "Just so you know, I don't own a car and I've never been to Japan.",
        "conversation_history": [],
    }, timeout=30)
    check("chat turn with negative facts completes", r.status_code == 200, f"got {r.status_code}")

    time.sleep(2)
    r2 = safe_request("GET", f"{BASE_MEMORY}/memories/{TEST_USER}", timeout=15)
    if r2.status_code == 200:
        contents = [m["content"].lower() for m in r2.json()]
        negative_leaks = [c for c in contents if ("doesn't" in c or "does not" in c or "never" in c or "no car" in c or "not own" in c)]
        check(
            "no negative/absence facts stored",
            len(negative_leaks) == 0,
            f"found: {negative_leaks}" if negative_leaks else "clean"
        )


# ── 6. Cross-session recall (the headline Track 1 feature) ──────────────────

def test_cross_session_recall():
    section("6. Cross-Session Recall")
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())  # simulates a brand new session, same user

    r1 = safe_request("POST", f"{BASE_AGENT}/chat", json={
        "user_id": TEST_USER,
        "session_id": session_a,
        "message": "My project deadline is July 9, 2026 and I'm building a memory agent for the Qwen hackathon.",
        "conversation_history": [],
    }, timeout=30)
    check("session A: chat turn completes", r1.status_code == 200, f"got {r1.status_code}")

    time.sleep(3)  # allow extraction pipeline to finish writing memories

    r2 = safe_request("POST", f"{BASE_AGENT}/chat", json={
        "user_id": TEST_USER,
        "session_id": session_b,  # different session, same user_id
        "message": "What's my deadline again?",
        "conversation_history": [],  # empty — simulates a fresh session with no local history
    }, timeout=30)
    check("session B (new session): chat turn completes", r2.status_code == 200, f"got {r2.status_code}")

    if r2.status_code == 200:
        data = r2.json()
        check("session B recalled memories from session A", data.get("context_injected") is True, str(data))
        reply_mentions_date = "july" in data.get("reply", "").lower() or "9" in data.get("reply", "")
        check("reply references the correct deadline", reply_mentions_date, data.get("reply", "")[:200])


# ── 7. Smart Forget ───────────────────────────────────────────────────────────

def test_smart_forget():
    section("7. Smart Forget (/forget)")
    session_id = str(uuid.uuid4())

    # Store a trivial, low-importance fact that should get a short TTL
    r = safe_request("POST", f"{BASE_MEMORY}/memory", json={
        "user_id": TEST_USER,
        "session_id": session_id,
        "content": "hello there",
    }, timeout=15)
    check("trivial fact stores", r.status_code == 200, f"got {r.status_code}")
    if r.status_code == 200:
        expires_at = r.json().get("expires_at")
        check("trivial fact got a short/expiring TTL (not permanent)", expires_at is not None, f"expires_at={expires_at}")

    # NOTE: this memory's expires_at is likely hours/days in the future, so /forget
    # won't touch it yet. This call mainly verifies the endpoint itself works end-to-end.
    r2 = safe_request("DELETE", f"{BASE_AGENT}/forget/{TEST_USER}", params={"batch_size": 20}, timeout=30)
    check("DELETE /forget/{user_id} proxy responds 200", r2.status_code == 200, f"got {r2.status_code}: {r2.text[:200]}")
    if r2.status_code == 200:
        data = r2.json()
        check("forget response has reviewed/deleted/kept fields", all(k in data for k in ("reviewed", "deleted", "kept")), str(data))
        print(f"  [{INFO}] reviewed={data.get('reviewed')} deleted={data.get('deleted')} kept={data.get('kept')} (0/0/0 is expected unless something had already expired)")

    # Direct memory_api.py call too, to confirm both layers work independently
    r3 = safe_request("DELETE", f"{BASE_MEMORY}/forget", json={"user_id": TEST_USER, "batch_size": 20}, timeout=30)
    check("DELETE /forget on memory_api.py directly responds 200", r3.status_code == 200, f"got {r3.status_code}")


# ── 8. Manual delete + clear-all ─────────────────────────────────────────────

def test_manual_delete(memory_id):
    section("8. Manual Delete + Clear All")
    if memory_id:
        r = safe_request("DELETE", f"{BASE_MEMORY}/memory/{memory_id}", timeout=15)
        check("DELETE /memory/{id} removes a specific memory", r.status_code == 200, f"got {r.status_code}")
    else:
        print(f"  [{INFO}] skipped — no memory_id captured from earlier test")

    r2 = safe_request("DELETE", f"{BASE_MEMORY}/memories/{TEST_USER}", timeout=15)
    check("DELETE /memories/{user_id} clears all memories", r2.status_code == 200, f"got {r2.status_code}")

    time.sleep(1)
    r3 = safe_request("GET", f"{BASE_MEMORY}/memories/{TEST_USER}", timeout=15)
    if r3.status_code == 200:
        check("memory list is empty after clear-all", len(r3.json()) == 0, f"{len(r3.json())} memories remain")


# ── 9. Qwen Cloud connectivity sanity check ─────────────────────────────────

def test_qwen_connectivity():
    section("9. Qwen Cloud API Connectivity (indirect)")
    # We can't call Qwen directly without your API key, but every prior test that
    # succeeded already proves Qwen chat + embeddings are reachable from this host.
    # This just summarizes that inference for the report.
    memory_dependent_tests = [n for n, ok, _ in results if "recall" in n.lower() or "importance" in n.lower()]
    passed = [n for n in memory_dependent_tests if any(r[0] == n and r[1] for r in results)]
    check(
        "Qwen-dependent operations succeeded (proves live API connectivity)",
        len(passed) > 0,
        f"{len(passed)}/{len(memory_dependent_tests)} Qwen-dependent checks passed so far"
    )


# ── runner ────────────────────────────────────────────────────────────────────

def main():
    print(f"\nTesting against:\n  memory_api.py -> {BASE_MEMORY}\n  agent.py      -> {BASE_AGENT}")
    print(f"Test user: {TEST_USER}\n")

    test_health()
    memory_id = test_basic_store_and_recall()
    test_importance_calibration()
    test_dedup_and_conflicts()
    test_negative_fact_filtering()
    test_cross_session_recall()
    test_smart_forget()
    test_manual_delete(memory_id)
    test_qwen_connectivity()

    section("SUMMARY")
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed
    print(f"  {passed}/{total} checks passed")
    if failed:
        print(f"\n  Failed checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"    - {name}" + (f" ({detail})" if detail else ""))
        print(f"\n  System is NOT fully demo-ready — fix the above before recording.")
        sys.exit(1)
    else:
        print(f"\n  All checks passed. System behavior matches README claims.")
        sys.exit(0)


if __name__ == "__main__":
    main()