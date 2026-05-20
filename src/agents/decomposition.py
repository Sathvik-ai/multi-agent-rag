import os
import json
from openai import OpenAI
from typing import List

class QueryDecompositionAgent:
    """
    UNIQUE FEATURE (Level 3):
    Instead of a single-shot RAG query, this agent breaks a complex scientific
    question into N targeted sub-questions. Each sub-question is then passed
    individually to the retrieval agent, enabling multi-hop reasoning across
    multiple documents.
    
    This allows the system to answer questions that require evidence from
    multiple sources simultaneously (e.g., comparing techniques across papers).
    """
    
    def __init__(self, primary_model: str = 'gemini-2.5-flash', fallback_model: str = 'deepseek-ai/DeepSeek-V4-Flash:novita'):
        # API keys are read lazily at first call via _ensure_clients()
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.gemini_client = None
        self.deepseek_client = None
        self._clients_ready = False

    def _ensure_clients(self) -> bool:
        """Lazily builds the OpenAI clients at request time, not import time."""
        if self._clients_ready:
            return True
            
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        hf_token = os.getenv("HF_TOKEN")

        if gemini_api_key and gemini_api_key != "your_gemini_api_key_here":
            self.gemini_client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_api_key
            )
            
        if hf_token and hf_token.lower() != "your_hf_token_here":
            self.deepseek_client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token
            )

        if self.gemini_client or self.deepseek_client:
            self._clients_ready = True
            return True
            
        return False

    def decompose(self, question: str, max_subqueries: int = 3) -> List[str]:
        """
        Takes a complex question and returns a list of focused sub-questions
        that together answer the original question completely.
        """
        if not self._ensure_clients():
            # Mock decomposition for development without API keys
            print("[Mock] Query decomposition: returning original question as single sub-query.")
            return [question]
        
        prompt = f"""
        You are a scientific query decomposition expert.
        
        Break the following complex question into at most {max_subqueries} focused, independent sub-questions.
        Each sub-question should be targeted enough to retrieve a specific piece of evidence.
        
        Return ONLY a valid JSON array of strings. Example:
        ["Sub-question 1?", "Sub-question 2?", "Sub-question 3?"]
        
        Question: {question}
        """
        
        response_content = None
        
        # Primary: Gemini
        if self.gemini_client:
            try:
                response = self.gemini_client.chat.completions.create(
                    model=self.primary_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response_content = response.choices[0].message.content
            except Exception as e:
                print(f"Gemini Decomposition failed: {e}. Falling back to DeepSeek...")
                
        # Fallback: DeepSeek
        if not response_content and self.deepseek_client:
            try:
                response = self.deepseek_client.chat.completions.create(
                    model=self.fallback_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response_content = response.choices[0].message.content
            except Exception as e:
                print(f"DeepSeek Decomposition failed: {e}. Using original question.")
                
        if response_content:
            try:
                clean = response_content.replace('```json', '').replace('```', '').strip()
                sub_questions = json.loads(clean)
                # Ensure we always have a list of strings
                if isinstance(sub_questions, list):
                    return [str(q) for q in sub_questions[:max_subqueries]]
            except Exception as e:
                print(f"Failed to parse decomposition JSON: {e}")
        
        return [question]
