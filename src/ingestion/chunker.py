import re
from typing import List

class TextChunker:
    """
    A raw, transparent text chunker. Avoids black-box wrappers.
    Splits text by paragraphs, then sentences if necessary, ensuring chunks
    stay under the max_tokens limit with a defined overlap to preserve context.
    
    (Assuming ~4 chars per token as a rough heuristic for MiniLM).
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size # Approx tokens
        self.chunk_overlap = chunk_overlap
        self.chars_per_token = 4 # Heuristic

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of optimal size.
        """
        max_chars = self.chunk_size * self.chars_per_token
        overlap_chars = self.chunk_overlap * self.chars_per_token

        # 1. Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
                
            # If a single paragraph is larger than max_chars, split by sentences
            if len(p) > max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', p)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chars:
                        current_chunk += " " + sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        # Keep overlap
                        overlap_text = current_chunk[-overlap_chars:] if current_chunk else ""
                        current_chunk = overlap_text + " " + sentence
            else:
                if len(current_chunk) + len(p) <= max_chars:
                    current_chunk += "\n\n" + p
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    overlap_text = current_chunk[-overlap_chars:] if current_chunk else ""
                    current_chunk = overlap_text + "\n\n" + p
                    
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        # Final cleanup pass to remove empty chunks
        return [c for c in chunks if c.strip()]
