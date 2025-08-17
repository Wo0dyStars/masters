import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    text = " ".join(text.split())
    text = text.replace("\x00", "")
    
    return text.strip()

def filter_chunks(chunks: List[Document], min_length: int = 50, max_length: int = 4000) -> List[Document]:
    """Filter chunks by length and quality"""
    filtered = []
    
    for chunk in chunks:
        content = clean_text(chunk.page_content)
        
        if len(content) < min_length or len(content) > max_length:
            continue
        
        alpha_ratio = sum(c.isalpha() for c in content) / len(content) if content else 0
        if alpha_ratio < 0.5:
            continue
        
        chunk.page_content = content
        filtered.append(chunk)
    
    logger.info(f"Filtered {len(chunks)} chunks to {len(filtered)} high-quality chunks")
    return filtered

def remove_duplicate_chunks(chunks: List[Document]) -> List[Document]:
    """Remove duplicate chunks based on content similarity"""
    seen_content = set()
    unique_chunks = []
    
    for chunk in chunks:
        content_hash = hash(chunk.page_content[:100])
        
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_chunks.append(chunk)
    
    logger.info(f"Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
    return unique_chunks

def load_existing_store(store_path: Path, embeddings: Optional[OpenAIEmbeddings] = None) -> Optional[FAISS]:
    """Load an existing FAISS vector store"""
    if not store_path.exists():
        logger.warning(f"Vector store not found at: {store_path}")
        return None
    
    try:
        if embeddings is None:
            embeddings = OpenAIEmbeddings()
        
        vectorstore = FAISS.load_local(
            str(store_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        logger.info(f"Loaded existing vector store from: {store_path}")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None

def verify_store(store_path: Path, sample_query: str = "test query") -> Dict[str, Any]:
    """Verify vector store integrity and performance"""
    results = {
        "exists": False,
        "loadable": False,
        "searchable": False,
        "document_count": 0,
        "error": None
    }
    
    try:
        if not store_path.exists():
            results["error"] = "Vector store directory not found"
            return results
        
        results["exists"] = True
        
        vectorstore = load_existing_store(store_path)
        if vectorstore is None:
            results["error"] = "Failed to load vector store"
            return results
        
        results["loadable"] = True
        
        docs = vectorstore.similarity_search(sample_query, k=1)
        results["searchable"] = True
        results["document_count"] = vectorstore.index.ntotal
        
        logger.info(f"Vector store verification passed")
        logger.info(f"   - Documents: {results['document_count']}")
        logger.info(f"   - Sample search returned: {len(docs)} results")
        
    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Vector store verification failed: {e}")
    
    return results