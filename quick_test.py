import requests
import time
import json
import sys

BASE_URL = "http://localhost:8000"

def ask_question(question: str):
    print(f"\n🤔 Question: {question}")
    print("⏳ Waiting for response...")
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={
                "question": question,
                "use_multihop": True,
                "enable_arxiv_fallback": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        
        elapsed = round(time.time() - start_time, 2)
        
        print("\n✅ Response Received!")
        print(f"🎯 Cache status: {data.get('cache_status', 'unknown').upper()}")
        print(f"📈 Retrieval Confidence: {data.get('confidence', 0.0)}")
        
        # Grounding / Hallucination Detection (Level 3 audit)
        hall = data.get("hallucination", {})
        if hall:
            print(f"🛡️  Grounding Score: {hall.get('grounding_score', 'N/A')} (Audit: {'PASS' if hall.get('grounding_score', 0) >= 0.7 else 'WARNING'})")
            if hall.get("flags"):
                print(f"⚠️  Hallucination Flags: {', '.join(hall['flags'])}")
        
        # Latency breakdown
        latency = data.get("latency", {})
        print("\n⏱️  Detailed Latency Breakdown:")
        print(f"  • Total turnaround:  {elapsed} seconds")
        if latency:
            print(f"  • Query Decomposition: {latency.get('decomposition_ms', 0)} ms")
            print(f"  • Vector Retrieval:    {latency.get('retrieval_ms', 0)} ms")
            print(f"  • LLM Generation:      {latency.get('llm_ms', 0)} ms")
            print(f"  • Server Pipeline:     {latency.get('total_ms', 0)} ms")

        print("\n📝 Answer:")
        print("-" * 65)
        print(data.get("answer", "No answer provided."))
        print("-" * 65)
        
        if data.get("sub_questions"):
            print("\n📌 Decomposition Breakdown:")
            for i, sq in enumerate(data["sub_questions"], 1):
                print(f"  {i}. {sq}")
                
        # Source Citations and Evidence
        sources = data.get("sources", [])
        if sources:
            print("\n📚 Grounded Evidence Citations:")
            for i, src in enumerate(sources, 1):
                print(f"  [{i}] Source: '{src.get('title', 'Unknown Title')}' (Doc ID: {src.get('document_id', 'Unknown')})")
                print(f"      Similarity Score: {src.get('score', 0.0)}")
                print(f"      Evidence: \"{src.get('text', '')[:120].strip()}...\"")
                
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error connecting to API: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is the efficacy of the drug targeting Asthma?"
        
    ask_question(question)
