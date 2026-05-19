import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

class DocumentParser:
    """
    Parses heterogeneous scientific data sources (PDFs and CSVs).
    Extracts raw text and basic metadata.
    """
    
    @staticmethod
    def parse_pdf(file_path: str | Path) -> Dict[str, Any]:
        """
        Parses a scientific research paper (PDF).
        """
        doc = fitz.open(file_path)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content.append(page.get_text("text"))
            
        full_text = "\n".join(text_content)
        metadata = {
            "title": doc.metadata.get("title", Path(file_path).name),
            "author": doc.metadata.get("author", "Unknown"),
            "pages": len(doc),
            "source_type": "pdf"
        }
        
        return {
            "text": full_text,
            "metadata": metadata
        }

    @staticmethod
    def parse_csv(file_path: str | Path, text_columns: List[str] = None) -> Dict[str, Any]:
        """
        Parses a structured scientific dataset (CSV).
        
        KEY FIX: Returns each row as a SEPARATE pre-chunked entry instead of
        joining all rows into one blob. This ensures each row gets its OWN
        embedding vector in Qdrant, so semantic search for 'Asthma' returns
        the BreathMax row specifically — not a chunk that averages across
        all disease rows.
        """
        df = pd.read_csv(file_path)
        
        # Use ALL columns (both text and numeric) for maximum context per row
        all_columns = df.columns.tolist()
            
        # Each row becomes its own self-contained text document
        row_documents = []
        for index, row in df.iterrows():
            row_parts = []
            for col in all_columns:
                if pd.notna(row[col]):
                    row_parts.append(f"{col}: {row[col]}")
            row_documents.append("\n".join(row_parts))
            
        metadata = {
            "title": Path(file_path).name,
            "rows": len(df),
            "columns": list(df.columns),
            "source_type": "csv"
        }
        
        # Return pre_chunked=True so the pipeline skips the text chunker
        # and uses these individual row strings directly as embeddings
        return {
            "text": row_documents[0] if row_documents else "",  # For Neo4j graph extraction
            "pre_chunked": row_documents,                        # All rows as individual chunks
            "metadata": metadata
        }


    @classmethod
    def parse_file(cls, file_path: str | Path) -> Dict[str, Any]:
        """
        Routes the file to the correct parser based on extension.
        """
        path = Path(file_path)
        if path.suffix.lower() == '.pdf':
            return cls.parse_pdf(path)
        elif path.suffix.lower() == '.csv':
            return cls.parse_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
