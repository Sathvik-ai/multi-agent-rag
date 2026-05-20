import os
import json
from openai import OpenAI

class ExtractionAgent:
    """
    (Level 2) Extraction Agent: Reads raw text from a document's first page
    and uses the LLM to extract structured metadata (Title, Authors, Topics)
    for accurate Knowledge Graph population in Neo4j.
    """
    
    def __init__(self, model_name: str = 'deepseek-ai/DeepSeek-V4-Flash:novita'):
        api_key = os.getenv("HF_TOKEN")
        if not api_key or api_key == "your_hf_token_here":
            self.api_key_valid = False
            self.client = None
        else:
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=api_key
            )
            self.api_key_valid = True
        self.model_name = model_name

    def extract_metadata(self, first_page_text: str) -> dict:
        """
        Extracts Title and Authors from the raw text of a paper.
        """
        if not self.api_key_valid:
            # Fallback mock extraction based on known papers for demonstration
            text_lower = first_page_text.lower()
            if "attention is all you need" in text_lower:
                return {"title": "Attention Is All You Need", "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"]}
            elif "flashattention" in text_lower:
                return {"title": "FlashAttention: Fast and Memory-Efficient Exact Attention", "authors": ["Tri Dao", "Daniel Y. Fu", "Stefano Ermon", "Atri Rudra", "Christopher Re"]}
            elif "retrieval-augmented generation" in text_lower:
                return {"title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus", "Fabio Petroni", "Vladimir Karpukhin", "Naman Goyal", "Heinrich Kuttler", "Mike Lewis", "Wen-tau Yih", "Tim Rocktaschel", "Sebastian Riedel", "Douwe Kiela"]}
            elif "react : synergizing" in text_lower or "react: synergizing" in text_lower:
                return {"title": "ReAct: Synergizing Reasoning and Acting in Language Models", "authors": ["Shunyu Yao", "Jeffrey Zhao", "Dian Yu", "Nan Du", "Izhak Shafran", "Karthik Narasimhan", "Yuan Cao"]}
            
            return {"title": "Unknown Title", "authors": ["Unknown Author"]}

        prompt = f"""
        You are a metadata extraction assistant. Read the following text from the first page of a scientific paper.
        Extract the exact Title and the list of Authors.
        
        Return ONLY a valid JSON object in this exact format:
        {{"title": "The Paper Title", "authors": ["Author 1", "Author 2"]}}
        
        Text:
        {first_page_text[:2000]}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            # Clean up the response in case it contains markdown code blocks
            clean_json = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"Extraction Agent Error: {e}")
            return {"title": "Unknown Title", "authors": ["Unknown Author"]}
