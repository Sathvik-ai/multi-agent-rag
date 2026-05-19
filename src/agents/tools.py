import os
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

class ArxivTool:
    """
    UNIQUE FEATURE: Live ArXiv Fallback Tool.
    
    If our local vector corpus doesn't have enough information to answer a question
    (low confidence scores from retrieval), this tool automatically:
    1. Searches the ArXiv API for relevant papers.
    2. Downloads the top result as a PDF.
    3. Returns the PDF path for immediate ingestion into the pipeline.
    
    This means our RAG system can HEAL ITS OWN KNOWLEDGE GAPS in real-time!
    """
    
    ARXIV_API = "http://export.arxiv.org/api/query"
    DOWNLOAD_DIR = Path("data/papers")
    
    def search(self, query: str, max_results: int = 1) -> list[dict]:
        """
        Searches ArXiv for papers matching the query.
        Returns list of dicts with title, summary, and pdf_url.
        """
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "max_results": max_results,
            "sortBy": "relevance"
        })
        
        url = f"{self.ARXIV_API}?{params}"
        
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = response.read().decode("utf-8")
                
            root = ET.fromstring(data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            
            results = []
            for entry in entries:
                arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                summary = entry.find('atom:summary', ns).text.strip()
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                
                results.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "summary": summary,
                    "pdf_url": pdf_url
                })
            
            return results
        
        except Exception as e:
            print(f"ArXiv search failed: {e}")
            return []
    
    def download_paper(self, pdf_url: str, arxiv_id: str) -> Path | None:
        """
        Downloads a PDF from ArXiv and saves it to our data/papers directory.
        Returns the path to the downloaded file.
        """
        safe_name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', arxiv_id)
        out_path = self.DOWNLOAD_DIR / f"{safe_name}.pdf"
        
        # Skip if already downloaded
        if out_path.exists():
            print(f"Paper already exists locally: {out_path}")
            return out_path
        
        try:
            print(f"Downloading paper from ArXiv: {pdf_url}")
            urllib.request.urlretrieve(pdf_url, out_path)
            return out_path
        except Exception as e:
            print(f"Download failed: {e}")
            return None
    
    def search_and_download(self, query: str) -> Path | None:
        """
        Combined search + download. Returns the local PDF path if successful.
        """
        results = self.search(query, max_results=1)
        if not results:
            return None
        
        top = results[0]
        print(f"Found on ArXiv: '{top['title']}'")
        return self.download_paper(top["pdf_url"], top["arxiv_id"])
