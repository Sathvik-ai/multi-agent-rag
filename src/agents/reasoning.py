import os
import google.generativeai as genai
from typing import List, Dict, Any
from .retrieval import RetrievalAgent

class ReasoningAgent:
    """
    Reasoning Agent that takes a question, retrieves context using the RetrievalAgent,
    and formulates a grounded, attributable answer using the Gemini API.
    Avoids LangChain's blackbox chains in favor of a clear, raw prompt workflow.
    """
    
    def __init__(self, model_name: str = 'gemini-1.5-flash'):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            print("WARNING: Valid GEMINI_API_KEY not found in .env. Using mock response mode.")
            self.api_key_valid = False
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.api_key_valid = True
            
        self.retriever = RetrievalAgent()
        
    def _format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Formats the retrieved chunks into a clean text block for the prompt."""
        if not retrieved_docs:
            return "No relevant context found."
            
        context_str = ""
        for i, doc in enumerate(retrieved_docs, 1):
            context_str += f"\n--- Source [{i}] ({doc['title']}) ---\n"
            context_str += f"{doc['text']}\n"
        return context_str

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Executes the RAG flow: Retrieve -> Prompt -> Generate
        """
        # 1. Retrieve
        docs = self.retriever.search(question, limit=5)
        context = self._format_context(docs)
        
        # 2. Build the System Prompt (Instructional guidance to avoid hallucination)
        prompt = f"""
        You are a highly intelligent scientific reasoning agent. 
        Your task is to answer the user's question based strictly on the provided context below.
        
        Rules:
        1. If the answer is not in the context, say "I don't have enough information to answer that." Do not hallucinate.
        2. Always cite your sources using the [Source Number] format.
        
        Context:
        {context}
        
        User Question:
        {question}
        
        Answer:
        """
        
        # 3. Generate Answer
        if self.api_key_valid:
            response = self.model.generate_content(prompt)
            answer_text = response.text
        else:
            answer_text = f"[Mock Mode: No API Key Provided]\n\nBased on Source [1], I can confirm that the text discusses topics related to your question. (Please provide a real GEMINI_API_KEY to see actual LLM output)."

        # 4. Return complete attributable response
        return {
            "question": question,
            "answer": answer_text,
            "sources": docs,
            "confidence": 0.95 if docs else 0.1 # Placeholder for Level 3 hallucination scoring
        }
