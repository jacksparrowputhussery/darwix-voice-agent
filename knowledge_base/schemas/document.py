from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class Document(BaseModel):
    document_id: str
    source_type: str
    source_uri: str
    title: str
    raw_content: str
    extracted_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    source_version: str = "1.0"
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CleanedDocument(BaseModel):
    document_id: str
    source_type: str
    source_uri: str
    title: str
    content: str
    extracted_at: str
    source_version: str
    pii_detected: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    record_id: str
    title: str
    section: str = ""
    content: str
    source: str
    version: str
    category: str = "general"
    effective_date: Optional[str] = None
    pii: bool = False
    embedding_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RetrievalResult(BaseModel):
    record_id: str
    chunk_id: str
    content: str
    score: float
    source: str
    title: str
    version: str
    metadata: Dict[str, Any]
