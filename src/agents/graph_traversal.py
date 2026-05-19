from src.database.connection import get_neo4j_driver
from typing import List, Dict, Any

class GraphTraversalAgent:
    """
    (Level 2) Graph Traversal Agent for finding relationships between entities.
    Instead of semantic similarity, this uses Neo4j to find structural connections
    like Authors -> Papers -> Topics.
    """
    
    def __init__(self):
        self.driver = get_neo4j_driver()
        
    def find_related_papers_by_author(self, author_name: str) -> List[Dict[str, Any]]:
        """
        Cypher query to traverse the knowledge graph:
        MATCH (a:Author {name: $name})-[:WROTE]->(p:Paper) RETURN p
        """
        query = """
        MATCH (a:Author {name: $author_name})-[:WROTE]->(p:Paper)
        RETURN p.title AS title, p.id AS document_id
        """
        
        results = []
        with self.driver.session() as session:
            try:
                records = session.run(query, author_name=author_name)
                for record in records:
                    results.append({
                        "title": record["title"],
                        "document_id": record["document_id"]
                    })
            except Exception as e:
                print(f"Graph Traversal Error: {e}")
                
        return results
