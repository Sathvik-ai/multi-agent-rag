import os
import json
from google import genai
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
    
    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            self.api_key_valid = False
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)
            self.api_key_valid = True
        self.model_name = model_name

    def decompose(self, question: str, max_subqueries: int = 3) -> List[str]:
        """
        Takes a complex question and returns a list of focused sub-questions
        that together answer the original question completely.
        """
        if not self.api_key_valid:
            # Mock decomposition for development without API key
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
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            clean = response.text.replace('```json', '').replace('```', '').strip()
            sub_questions = json.loads(clean)
            # Ensure we always have a list of strings
            if isinstance(sub_questions, list):
                return [str(q) for q in sub_questions[:max_subqueries]]
        except Exception as e:
            print(f"Decomposition failed: {e}. Using original question.")
        
        return [question]
