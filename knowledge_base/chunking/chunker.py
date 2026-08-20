import uuid
from typing import List
from knowledge_base.schemas.document import CleanedDocument, Chunk

class SemanticChunker:
    def __init__(self, target_words: int = 400, overlap_words: int = 50):
        self.target_words = target_words
        self.overlap_words = overlap_words

    def chunk(self, doc: CleanedDocument) -> List[Chunk]:
        # Simple heuristic chunking by paragraphs, targeting a specific word count.
        paragraphs = doc.content.split("\n")
        chunks = []
        current_chunk_text = ""
        current_word_count = 0
        section = "General"
        
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            
            # Simple section detection: Short lines might be headings
            if len(p) < 60 and not p.endswith(".") and p.istitle():
                section = p
                
            p_words = len(p.split())
            
            if current_word_count + p_words > self.target_words and current_chunk_text:
                chunks.append(self._create_chunk(doc, current_chunk_text, section))
                # Keep overlap (simplified: keep the last paragraph if it fits in overlap limit)
                current_chunk_text = p + "\n"
                current_word_count = p_words
            else:
                current_chunk_text += p + "\n"
                current_word_count += p_words
                
        if current_chunk_text:
            chunks.append(self._create_chunk(doc, current_chunk_text, section))
            
        return chunks

    def _create_chunk(self, doc: CleanedDocument, text: str, section: str) -> Chunk:
        record_id = f"kb_{doc.document_id}"
        return Chunk(
            chunk_id=f"chk_{uuid.uuid4().hex[:8]}",
            document_id=doc.document_id,
            record_id=record_id,
            title=doc.title,
            section=section,
            content=text.strip(),
            source=doc.source_uri,
            version=doc.source_version,
            pii=doc.pii_detected,
            metadata=doc.metadata
        )
