import os
import chromadb
import google.generativeai as genai
from typing import List, Dict, Any
from knowledge_base.schemas.document import Chunk, RetrievalResult
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "knowledge_base"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
            
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            # Fallback to dummy embeddings for local testing without an API key
            print("Using mock embeddings because GEMINI_API_KEY is not set.")
            return [[hash(t) % 1000 / 1000.0] * 768 for t in texts]
            
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=texts,
                task_type="retrieval_document"
            )
            if isinstance(result['embedding'][0], float):
                return [result['embedding']]
            return result['embedding']
        except Exception as e:
            print(f"Embedding API error: {e}. Falling back to mock embeddings.")
            return [[hash(t) % 1000 / 1000.0] * 768 for t in texts]

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
            
        texts = [chunk.content for chunk in chunks]
        embeddings = self._get_embeddings(texts)
        
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "record_id": chunk.record_id,
                "title": chunk.title,
                "section": chunk.section,
                "source": chunk.source,
                "version": chunk.version,
                "category": chunk.category,
                "pii": str(chunk.pii)
            }
            for chunk in chunks
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )

    def retrieve(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[RetrievalResult]:
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key or api_key == "your_gemini_api_key_here":
            query_embedding = [hash(query) % 1000 / 1000.0] * 768
        else:
            try:
                query_embedding = genai.embed_content(
                    model=self.embedding_model,
                    content=query,
                    task_type="retrieval_query"
                )['embedding']
            except Exception as e:
                print(f"Embedding API error: {e}. Falling back to mock embedding.")
                query_embedding = [hash(query) % 1000 / 1000.0] * 768
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        retrieval_results = []
        if not results['ids'] or not results['ids'][0]:
            return retrieval_results
            
        for i in range(len(results['ids'][0])):
            retrieval_results.append(RetrievalResult(
                record_id=results['metadatas'][0][i].get('record_id', ''),
                chunk_id=results['ids'][0][i],
                content=results['documents'][0][i],
                score=1.0 - results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0, # Chroma returns distance, lower is better
                source=results['metadatas'][0][i].get('source', ''),
                title=results['metadatas'][0][i].get('title', ''),
                version=results['metadatas'][0][i].get('version', ''),
                metadata=results['metadatas'][0][i]
            ))
            
        return retrieval_results
