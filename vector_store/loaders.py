import logging
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Enhanced document loader supporting multiple formats"""
    
    def __init__(self):
        self.loaders = {
            ".pdf": self._load_pdf,
            ".txt": self._load_text,
            ".md": self._load_markdown,
            ".docx": self._load_docx
        }
    
    def load_documents(self, docs_folder: Path, supported_formats: List[str]) -> List[Document]:
        """Load all supported documents from folder"""
        documents = []
        file_count = 0
        
        logger.info(f"Loading documents from: {docs_folder}")
        
        for format_ext in supported_formats:
            pattern = f"*{format_ext}"
            files = list(docs_folder.glob(pattern))
            
            logger.info(f"Found {len(files)} {format_ext} files")
            
            for file_path in files:
                try:
                    docs = self._load_file(file_path)
                    documents.extend(docs)
                    file_count += 1
                    logger.debug(f"Loaded {len(docs)} documents from {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {file_count} files")
        return documents
    
    def _load_file(self, file_path: Path) -> List[Document]:
        """Load a single file based on its extension"""
        suffix = file_path.suffix.lower()
        
        if suffix in self.loaders:
            return self.loaders[suffix](file_path)
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            return []
    
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """Load PDF documents"""
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        
        for i, doc in enumerate(docs):
            doc.metadata.update({
                "source_file": file_path.name,
                "file_type": "pdf",
                "page_number": i + 1,
                "total_pages": len(docs)
            })
        
        return docs
    
    def _load_text(self, file_path: Path) -> List[Document]:
        """Load text documents"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        
        for doc in docs:
            doc.metadata.update({
                "source_file": file_path.name,
                "file_type": "text"
            })
        
        return docs
    
    def _load_markdown(self, file_path: Path) -> List[Document]:
        """Load markdown documents"""
        loader = UnstructuredMarkdownLoader(str(file_path))
        docs = loader.load()
        
        for doc in docs:
            doc.metadata.update({
                "source_file": file_path.name,
                "file_type": "markdown"
            })
        
        return docs
    
    def _load_docx(self, file_path: Path) -> List[Document]:
        """Load DOCX documents"""
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()
        
        for doc in docs:
            doc.metadata.update({
                "source_file": file_path.name,
                "file_type": "docx"
            })
        
        return docs