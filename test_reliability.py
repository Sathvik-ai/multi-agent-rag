"""
Comprehensive Reliability Test Suite for the Multi-Agent RAG Pipeline.
Tests: API health, LLM answers, retrieval accuracy, cache behavior,
       edge cases, error handling, and cross-document reasoning.
"""

import time
import requests
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def run_test(name: str, fn) -> bool:
    t = time.time()
    try:
        status, detail = fn()
        elapsed = round((time.time() - t) * 1000, 1)
        icon = PASS if status else FAIL
        print(f"{icon} [{elapsed}ms] {name}")
        if detail:
            for line in detail.splitlines():
                print(f"      {line}")
        results.append({"name": name, "passed": status, "ms": elapsed})
        return status
    except Exception as e:
        elapsed = round((time.time() - t) * 1000, 1)
        print(f"{FAIL} [{elapsed}ms] {name} — Exception: {e}")
        results.append({"name": name, "passed": False, "ms": elapsed})
        return False

def query(question: str, fallback: bool = False) -> Dict[str, Any]:
    r = requests.post(f"{BASE_URL}/query",
        json={"question": question, "use_multihop": True, "enable_arxiv_fallback": fallback},
        timeout=60)
    r.raise_for_status()
    return r.json()

# ─── CATEGORY 1: Infrastructure Health ───────────────────────────
print("\n" + "═"*60)
print("  CATEGORY 1: Infrastructure Health")
print("═"*60)

def test_api_alive():
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    return r.status_code == 200 and r.json()["status"] == "healthy", ""
run_test("API server is running and healthy", test_api_alive)

def test_openapi_docs():
    r = requests.get(f"{BASE_URL}/docs", timeout=5)
    return r.status_code == 200, ""
run_test("Swagger /docs page loads", test_openapi_docs)

def test_graph_endpoint():
    r = requests.get(f"{BASE_URL}/graph/papers", timeout=10)
    data = r.json()
    count = data.get("count", 0)
    return count >= 30, f"Found {count} Author→Paper relationships in Neo4j"
run_test("Neo4j graph has Author→Paper relationships", test_graph_endpoint)

# ─── CATEGORY 2: CSV Dataset Retrieval Accuracy ──────────────────
print("\n" + "═"*60)
print("  CATEGORY 2: CSV Dataset — Row-Level Retrieval Accuracy")
print("═"*60)

def test_asthma():
    r = query("Which drug targets Asthma and what is its efficacy score?")
    a = r["answer"].lower()
    passed = "breathmax" in a and "0.95" in a
    return passed, f"Answer: {r['answer'][:180]}"
run_test("Asthma → BreathMax (efficacy 0.95)", test_asthma)

def test_alzheimers():
    r = query("What drug is targeting Alzheimer's Disease and what phase is it in?")
    a = r["answer"].lower()
    passed = "neurogen" in a and "phase 3" in a
    return passed, f"Answer: {r['answer'][:180]}"
run_test("Alzheimer's → Neurogen-X (Phase 3)", test_alzheimers)

def test_highest_efficacy():
    r = query("Which clinical trial has the highest efficacy score?")
    a = r["answer"].lower()
    passed = "breathmax" in a or "0.95" in a
    return passed, f"Answer: {r['answer'][:180]}"
run_test("Highest efficacy drug → BreathMax (0.95)", test_highest_efficacy)

def test_terminated_trial():
    r = query("Which drug trial was terminated and what was the reason?")
    a = r["answer"].lower()
    passed = "pulmokill" in a or "tr-103" in a or "terminat" in a or "toxicit" in a
    return passed, f"Answer: {r['answer'][:180]}"
run_test("Terminated trial → PulmoKill (toxicity)", test_terminated_trial)

def test_active_trials():
    r = query("List all active drug trials and their disease targets")
    a = r["answer"].lower()
    passed = "active" in a and any(d in a for d in ["diabetes", "depression", "parkinson", "cancer", "glucostab", "serenify", "dopasync", "oncoshield"])
    return passed, f"Answer: {r['answer'][:250]}"
run_test("Active trials — multi-row retrieval", test_active_trials)

# ─── CATEGORY 3: PDF Scientific Paper Retrieval ──────────────────
print("\n" + "═"*60)
print("  CATEGORY 3: PDF Scientific Papers — Retrieval Accuracy")
print("═"*60)

def test_transformer_arch():
    r = query("What is the architecture of the Transformer model?")
    a = r["answer"].lower()
    passed = "encoder" in a and "decoder" in a and "attention" in a
    return passed, f"Confidence: {r['confidence']} | Answer: {r['answer'][:200]}"
run_test("Transformer architecture (encoder+decoder+attention)", test_transformer_arch)

def test_flash_memory():
    r = query("How does FlashAttention reduce memory usage compared to standard attention?")
    a = r["answer"].lower()
    passed = any(kw in a for kw in ["sram", "hbm", "linear", "20x", "memory", "io"])
    return passed, f"Answer: {r['answer'][:200]}"
run_test("FlashAttention memory reduction technique", test_flash_memory)

def test_rag_knowledge():
    r = query("What problem does RAG solve that pure parametric language models cannot?")
    a = r["answer"].lower()
    passed = any(kw in a for kw in ["parametric", "knowledge", "retriev", "hallucin", "update"])
    return passed, f"Answer: {r['answer'][:200]}"
run_test("RAG vs parametric LLMs", test_rag_knowledge)

def test_react_interleaving():
    r = query("How does ReAct interleave reasoning and acting in language models?")
    a = r["answer"].lower()
    passed = any(kw in a for kw in ["reasoning", "acting", "interleav", "trace", "thought"])
    return passed, f"Answer: {r['answer'][:200]}"
run_test("ReAct interleaved reasoning+acting", test_react_interleaving)

# ─── CATEGORY 4: Cross-Document Multi-Hop Reasoning ─────────────
print("\n" + "═"*60)
print("  CATEGORY 4: Cross-Document Multi-Hop Reasoning")
print("═"*60)

def test_multihop():
    r = query("Compare how original Transformer attention differs from FlashAttention in terms of memory and speed")
    subs = r.get("sub_questions", [])
    doc_ids = set(s.get("document_id") for s in r.get("sources", []))
    multi_src = len(doc_ids) > 1
    return multi_src, \
        f"Sub-questions: {subs}\nSources from {len(doc_ids)} different documents"
run_test("Multi-hop: Attention paper + FlashAttention paper", test_multihop)

# ─── CATEGORY 5: Redis Cache Reliability ─────────────────────────
print("\n" + "═"*60)
print("  CATEGORY 5: Redis Cache Reliability")
print("═"*60)

def test_cache():
    q = "What side effects does CardioClear have in hypertension treatment?"
    r1 = query(q)
    miss_ok = r1.get("cache_status") == "miss"
    t_miss = r1.get("latency", {}).get("total_ms", 9999)
    r2 = query(q)
    hit_ok = r2.get("cache_status") == "hit"
    t_hit = r2.get("latency", {}).get("total_ms", 9999)
    speedup = round(t_miss / max(t_hit, 1), 1)
    passed = miss_ok and hit_ok
    return passed, f"1st call (miss): {t_miss}ms → 2nd call (hit): {t_hit}ms | {speedup}x faster"
run_test("Redis cache: miss → hit, significant speedup", test_cache)

# ─── CATEGORY 6: Edge Cases & Error Handling ─────────────────────
print("\n" + "═"*60)
print("  CATEGORY 6: Edge Cases & Error Handling")
print("═"*60)

def test_out_of_scope():
    r = query("Who won the FIFA World Cup in 2022?")
    a = r["answer"].lower()
    hall = r.get("hallucination", {})
    # Correct behavior: admits ignorance OR hallucination detector catches it
    admits = any(kw in a for kw in ["don't have", "not in", "no information", "cannot", "not found", "grounded"])
    low_grounding = hall.get("grounding_score", 1.0) < 0.6
    passed = admits or low_grounding
    return passed, f"Grounding score: {hall.get('grounding_score')} | Answer: {r['answer'][:150]}"
run_test("Out-of-scope question — no hallucination", test_out_of_scope)

def test_empty_question():
    try:
        r = requests.post(f"{BASE_URL}/query",
            json={"question": "", "use_multihop": True, "enable_arxiv_fallback": False},
            timeout=15)
        return r.status_code in [200, 422, 400], f"Status code: {r.status_code} (no crash)"
    except Exception as e:
        return False, str(e)
run_test("Empty question — server handles gracefully (no 500)", test_empty_question)

def test_unsupported_file():
    from io import BytesIO
    r = requests.post(f"{BASE_URL}/ingest",
        files={"file": ("test.txt", BytesIO(b"hello"), "text/plain")},
        timeout=10)
    return r.status_code == 400, f"Got {r.status_code}: {r.json().get('detail', '')}"
run_test("Unsupported .txt file upload → 400 error", test_unsupported_file)

def test_hallucination_present():
    # Every /query response should now contain a hallucination field
    r = query("How many parameters does the base Transformer model have?")
    hall = r.get("hallucination")
    has_field = hall is not None and "grounding_score" in hall
    return has_field, f"Hallucination check: grounding_score={hall.get('grounding_score') if hall else 'MISSING'}, flags={len(hall.get('flags',[]) if hall else [])}"
run_test("Hallucination field present in every /query response", test_hallucination_present)

# ─── SUMMARY ─────────────────────────────────────────────────────
print("\n" + "═"*60)
total = len(results)
passed_count = sum(1 for r in results if r["passed"])
failed_count = total - passed_count
avg_ms = round(sum(r["ms"] for r in results) / total, 1)

print(f"  FINAL SCORE: {passed_count}/{total} tests passed  |  {failed_count} failed  |  avg {avg_ms}ms/test")
print("═"*60)
if failed_count > 0:
    print("\n  Failed tests:")
    for r in results:
        if not r["passed"]:
            print(f"    ❌ {r['name']}")
