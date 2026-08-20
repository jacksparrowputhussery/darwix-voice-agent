import uuid
import csv
import json
import requests
from bs4 import BeautifulSoup
from typing import List
from pathlib import Path

# Note: PyPDF2 is installed via requirements.txt
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

from knowledge_base.schemas.document import Document

class BaseLoader:
    def load(self) -> List[Document]:
        raise NotImplementedError

class TextLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return [Document(
                document_id=f"doc_txt_{uuid.uuid4().hex[:8]}",
                source_type="text",
                source_uri=self.file_path,
                title=Path(self.file_path).name,
                raw_content=content
            )]
        except Exception as e:
            print(f"Error loading text file {self.file_path}: {e}")
            return []

class CSVLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        documents = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    content = "\n".join([f"{k}: {v}" for k, v in row.items() if v])
                    documents.append(Document(
                        document_id=f"doc_csv_{uuid.uuid4().hex[:8]}",
                        source_type="csv",
                        source_uri=f"{self.file_path}#row={i+1}",
                        title=f"{Path(self.file_path).name} Row {i+1}",
                        raw_content=content
                    ))
            return documents
        except Exception as e:
            print(f"Error loading CSV file {self.file_path}: {e}")
            return []

class PDFLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        if not PdfReader:
            print("PyPDF2 is not installed.")
            return []
        try:
            reader = PdfReader(self.file_path)
            content = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n"
            
            return [Document(
                document_id=f"doc_pdf_{uuid.uuid4().hex[:8]}",
                source_type="pdf",
                source_uri=self.file_path,
                title=Path(self.file_path).name,
                raw_content=content
            )]
        except Exception as e:
            print(f"Error loading PDF file {self.file_path}: {e}")
            return []

class WebLoader(BaseLoader):
    def __init__(self, url: str):
        self.url = url

    def load(self) -> List[Document]:
        try:
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract text while removing script/style tags
            for script in soup(["script", "style"]):
                script.extract()
                
            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = "\n".join(chunk for chunk in chunks if chunk)
            
            return [Document(
                document_id=f"doc_web_{uuid.uuid4().hex[:8]}",
                source_type="web",
                source_uri=self.url,
                title=soup.title.string if soup.title else self.url,
                raw_content=content
            )]
        except Exception as e:
            print(f"Error loading Web URL {self.url}: {e}")
            return []
