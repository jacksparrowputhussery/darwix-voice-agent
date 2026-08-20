import os
import sys
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from knowledge_base.ingestion.loaders import TextLoader, CSVLoader, PDFLoader, WebLoader
from knowledge_base.cleaning.cleaner import DocumentCleaner
from knowledge_base.chunking.chunker import SemanticChunker
from knowledge_base.retrieval.vector_store import VectorStore

def main():
    print("Starting Knowledge Base Ingestion...")
    data_dir = Path("./data/raw")
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        return

    all_docs = []
    
    # Loaders
    for file_path in data_dir.glob("*"):
        if file_path.suffix == ".txt":
            all_docs.extend(TextLoader(str(file_path)).load())
        elif file_path.suffix == ".csv":
            all_docs.extend(CSVLoader(str(file_path)).load())
        elif file_path.suffix == ".pdf":
            all_docs.extend(PDFLoader(str(file_path)).load())
            
    print(f"Loaded {len(all_docs)} raw documents.")
    
    # Clean
    cleaner = DocumentCleaner()
    cleaned_docs, report = cleaner.clean(all_docs)
    print("Cleaning Report:", report)
    
    # Chunk
    chunker = SemanticChunker(target_words=300)
    all_chunks = []
    for doc in cleaned_docs:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
        
    print(f"Created {len(all_chunks)} chunks.")
    
    # Index
    store = VectorStore()
    store.add_chunks(all_chunks)
    print("Ingestion complete. Embeddings stored in ChromaDB.")

if __name__ == "__main__":
    main()
