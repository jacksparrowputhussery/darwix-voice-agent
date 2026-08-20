import re
import hashlib
from typing import List, Tuple
from knowledge_base.schemas.document import Document, CleanedDocument

class DocumentCleaner:
    def __init__(self):
        self.seen_hashes = set()
        
    def clean(self, docs: List[Document]) -> Tuple[List[CleanedDocument], dict]:
        cleaned_docs = []
        report = {
            "input_docs": len(docs),
            "output_docs": 0,
            "duplicates_removed": 0,
            "pii_detected_count": 0
        }
        
        for doc in docs:
            # 1. Basic cleaning: remove excessive whitespace and boilerplate
            content = self._remove_boilerplate(doc.raw_content)
            
            # 2. Exact Deduplication via hashing
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            if content_hash in self.seen_hashes:
                report["duplicates_removed"] += 1
                continue
            self.seen_hashes.add(content_hash)
            
            # 3. PII Redaction
            content, pii_detected = self._redact_pii(content)
            if pii_detected:
                report["pii_detected_count"] += 1
                
            cleaned_doc = CleanedDocument(
                document_id=doc.document_id,
                source_type=doc.source_type,
                source_uri=doc.source_uri,
                title=doc.title,
                content=content,
                extracted_at=doc.extracted_at,
                source_version=doc.source_version,
                pii_detected=pii_detected,
                metadata=doc.metadata
            )
            cleaned_docs.append(cleaned_doc)
            report["output_docs"] += 1
            
        return cleaned_docs, report

    def _remove_boilerplate(self, text: str) -> str:
        # Simple heuristic to remove typical nav text, headers, footers
        lines = text.split("\n")
        cleaned_lines = []
        boilerplate_keywords = ["cookie policy", "accept cookies", "all rights reserved", "navigation", "skip to content"]
        
        for line in lines:
            if not line.strip():
                continue
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in boilerplate_keywords):
                continue
            cleaned_lines.append(line.strip())
            
        return "\n".join(cleaned_lines)

    def _redact_pii(self, text: str) -> Tuple[str, bool]:
        pii_detected = False
        
        # Redact Emails
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        if re.search(email_pattern, text):
            pii_detected = True
            text = re.sub(email_pattern, '[EMAIL]', text)
            
        # Redact Phone Numbers (Basic pattern)
        phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
        if re.search(phone_pattern, text):
            pii_detected = True
            text = re.sub(phone_pattern, '[PHONE]', text)
            
        # Redact SSN/ID numbers (example)
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        if re.search(ssn_pattern, text):
            pii_detected = True
            text = re.sub(ssn_pattern, '[GOV_ID]', text)
            
        return text, pii_detected
