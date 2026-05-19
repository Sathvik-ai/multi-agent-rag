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
        Converts rows into text documents.
        """
        df = pd.read_csv(file_path)
        
        # If text columns aren't specified, use all columns of dtype object/string
        if not text_columns:
            text_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
            
        # Combine the selected text columns into a single string representation per row
        documents = []
        for index, row in df.iterrows():
            row_text = []
            for col in text_columns:
                if pd.notna(row[col]):
                    row_text.append(f"{col}: {row[col]}")
            documents.append("\n".join(row_text))
            
        full_text = "\n\n---\n\n".join(documents)
        
        metadata = {
            "title": Path(file_path).name,
            "rows": len(df),
            "columns": list(df.columns),
            "source_type": "csv"
        }
        
        return {
            "text": full_text,
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
