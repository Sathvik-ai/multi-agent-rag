from .connection import engine, Base
from .models import DocumentMetadata, DocumentChunk, QueryLog

def init_db():
    print("Creating database tables in PostgreSQL...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

if __name__ == "__main__":
    init_db()
