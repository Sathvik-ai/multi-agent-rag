import os
import json
import time
from google import genai
from typing import List, Dict, Any

from .retrieval import RetrievalAgent
from .decomposition import QueryDecompositionAgent
from .tools import ArxivTool

# Confidence threshold below which we trigger ArXiv fallback
CONFIDENCE_THRESHOLD = 0.25

class ReasoningAgent:
    """
    Multi-Hop Reasoning Agent with Query Decomposition and ArXiv Fallback.
    
    Unlike standard single-shot RAG, this agent:
    1. Decomposes complex questions into targeted sub-queries.
    2. Runs independent retrieval for each sub-query.
    3. Synthesizes all evidence into one grounded, cited answer.
    4. If retrieval confidence is too low, uses the ArXiv Tool to find and
       ingest new papers on-the-fly, then re-retrieves.
    """
    
    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            self.api_key_valid = False
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)
            self.api_key_valid = True
        
        self.model_name = model_name
        self.retriever = RetrievalAgent()
        self.decomposer = QueryDecompositionAgent()
        self.arxiv_tool = ArxivTool()

    def _multi_hop_retrieve(self, sub_questions: List[str]) -> List[Dict]:
        """
        Runs parallel retrieval for each sub-question.
        Deduplicates chunks by their document_id + chunk_index.
        Returns a combined, deduplicated list of evidence with sub-question labels.
        """
        seen_ids = set()
        all_evidence = []
        
        for sub_q in sub_questions:
            results = self.retriever.search(sub_q, limit=3)
            for r in results:
                # Create a unique key to avoid the same chunk appearing twice
                key = f"{r['document_id']}_{r['text'][:50]}"
                if key not in seen_ids:
                    seen_ids.add(key)
                    r['sub_question'] = sub_q  # Tag source with which sub-question found it
                    all_evidence.append(r)
                    
        return all_evidence

    def _format_evidence(self, evidence: List[Dict]) -> str:
        """Formats multi-hop evidence with sub-question traceability."""
        if not evidence:
            return "No relevant context found."
        
        formatted = ""
        for i, doc in enumerate(evidence, 1):
            formatted += f"\n--- Source [{i}] from '{doc['title']}' | Found via: \"{doc['sub_question']}\" ---\n"
            formatted += f"{doc['text']}\n"
        return formatted

    def _compute_confidence(self, evidence: List[Dict]) -> float:
        """
        Heuristic confidence score based on average retrieval similarity scores.
        """
        if not evidence:
            return 0.0
        scores = [doc.get('score', 0) for doc in evidence]
        return round(sum(scores) / len(scores), 4)

    def ask(self, question: str, ingest_pipeline=None) -> Dict[str, Any]:
        """
        Full multi-hop RAG pipeline:
        1. Decompose question into sub-queries
        2. Multi-hop retrieval
        3. ArXiv fallback if confidence is low (and pipeline is provided)
        4. Synthesize with Gemini
        5. Return structured response with latency breakdown
        """
        timings = {}
        
        # --- Step 1: Decompose ---
        t0 = time.time()
        sub_questions = self.decomposer.decompose(question)
        timings['decomposition_ms'] = round((time.time() - t0) * 1000, 2)
        print(f"Sub-questions: {sub_questions}")
        
        # --- Step 2: Multi-Hop Retrieval ---
        t0 = time.time()
        evidence = self._multi_hop_retrieve(sub_questions)
        timings['retrieval_ms'] = round((time.time() - t0) * 1000, 2)
        
        # --- Step 3: ArXiv Fallback (if enabled and confidence is low) ---
        confidence = self._compute_confidence(evidence)
        arxiv_used = False
        
        if confidence < CONFIDENCE_THRESHOLD and ingest_pipeline is not None:
            print(f"Low confidence ({confidence}). Triggering ArXiv fallback...")
            pdf_path = self.arxiv_tool.search_and_download(question)
            if pdf_path:
                try:
                    ingest_pipeline.process_file(pdf_path)
                    evidence = self._multi_hop_retrieve(sub_questions)
                    confidence = self._compute_confidence(evidence)
                    arxiv_used = True
                except Exception as e:
                    print(f"ArXiv fallback ingestion failed: {e}")
        
        # --- Step 4: Generate Answer ---
        t0 = time.time()
        context = self._format_evidence(evidence)
        
        prompt = f"""
        You are an expert scientific reasoning agent.
        Answer the user's question using ONLY the provided multi-source evidence below.
        
        Rules:
        1. If the answer is NOT in the evidence, say "I don't have enough grounded information."
        2. Cite sources using [Source N] format inline in your answer.
        3. Be precise, technical, and concise.
        
        Multi-Source Evidence:
        {context}
        
        User Question: {question}
        
        Answer:
        """
        
        if self.api_key_valid:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                answer_text = response.text
            except Exception as e:
                print(f"Gemini generation failed: {e}. Falling back to deterministic summary.")
                # Self-healing fallback: compile a clean summary of the evidence
                if evidence:
                    top_text = evidence[0]['text'][:300].strip().replace('\n', ' ')
                    answer_text = (
                        f"[Service Degradation Fallback: {str(e)}]\n\n"
                        f"Based on retrieved sources (e.g., '{evidence[0].get('title', 'Document')}'):\n"
                        f"\"{top_text}...\" [Source 1]"
                    )
                else:
                    answer_text = (
                        f"[Service Degradation Fallback: {str(e)}]\n\n"
                        "I don't have enough grounded information to answer this question."
                    )
        else:
            answer_text = (
                "[Mock Mode: No GEMINI_API_KEY provided]\n\n"
                f"Based on {len(evidence)} evidence chunks retrieved across {len(sub_questions)} sub-questions, "
                "I would synthesize a grounded answer here using Gemini."
            )
        timings['llm_ms'] = round((time.time() - t0) * 1000, 2)
        timings['total_ms'] = sum(timings.values())
        
        return {
            "question": question,
            "sub_questions": sub_questions,
            "answer": answer_text,
            "sources": evidence,
            "confidence": confidence,
            "arxiv_fallback_used": arxiv_used,
            "latency": timings
        }
