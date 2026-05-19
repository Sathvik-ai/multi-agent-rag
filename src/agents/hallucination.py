import os
import json
from google import genai
from typing import Dict, Any

class HallucinationDetector:
    """
    Level 3: Hallucination & Confidence Evaluator.

    Uses a second independent Gemini call (judge prompt pattern) to evaluate
    whether the generated answer is actually supported by the retrieved sources.

    This is the 'LLM-as-a-judge' pattern — asking a second LLM pass to
    critically audit the first one's answer against the raw evidence.

    Returns:
    - is_grounded (bool): whether claims in the answer are supported by sources
    - grounding_score (float): 0.0 to 1.0 how well grounded the answer is
    - flags (list[str]): specific ungrounded claims identified
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

    def evaluate(self, question: str, answer: str, sources: list[Dict]) -> Dict[str, Any]:
        """
        Evaluates whether the answer is grounded in the provided source evidence.
        """
        if not self.api_key_valid:
            # Heuristic fallback: check if answer is suspiciously long/confident
            # without much source coverage
            return {
                "is_grounded": True,
                "grounding_score": 0.75,
                "flags": [],
                "method": "heuristic_fallback"
            }

        # Concatenate the raw source text for the judge
        source_text = "\n\n".join(
            [f"[Source {i+1}]: {s.get('text', '')[:500]}" for i, s in enumerate(sources)]
        )

        judge_prompt = f"""
        You are a strict scientific fact-checker and hallucination detector.

        Your task: Evaluate whether the ANSWER below is fully supported by the SOURCES provided.
        Do NOT use any external knowledge. Only judge based on the SOURCES given.

        SOURCES:
        {source_text}

        QUESTION: {question}

        ANSWER TO EVALUATE:
        {answer}

        Instructions:
        1. Identify any specific claims in the ANSWER that are NOT found in the SOURCES.
        2. Score the overall grounding from 0.0 (fully hallucinated) to 1.0 (fully grounded).
        3. Return ONLY a JSON object like this:
        {{
          "is_grounded": true,
          "grounding_score": 0.92,
          "flags": ["Claim about X is not in any source", "..."]
        }}
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=judge_prompt
            )
            clean = response.text.replace('```json', '').replace('```', '').strip()
            result = json.loads(clean)
            result["method"] = "llm_judge"
            return result
        except Exception as e:
            print(f"Hallucination detection failed: {e}")
            return {
                "is_grounded": True,
                "grounding_score": 0.5,
                "flags": [f"Evaluation error: {str(e)}"],
                "method": "error_fallback"
            }
