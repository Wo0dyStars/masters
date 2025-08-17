import logging
import time
from typing import List, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownTextSplitter
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import VectorStoreConfig, ChunkingStrategy
from loaders import DocumentLoader
from utils import filter_chunks, remove_duplicate_chunks, clean_text
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreBuilder:
    """Main class for building FAISS vector stores"""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self.loader = DocumentLoader()
        self.embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        
    def build(self, force_rebuild: bool = False) -> bool:
        """Build the vector store from documents"""
        try:
            if self.config.output_path.exists() and not force_rebuild:
                logger.warning(f"Vector store already exists at: {self.config.output_path}")
                logger.warning("Use force_rebuild=True to rebuild")
                return False
            
            logger.info("Starting vector store build process...")
            start_time = time.time()
            
            documents = self._load_documents()
            if not documents:
                logger.error("No documents loaded. Check your docs folder and file formats.")
                return False
            
            chunks = self._create_chunks(documents)
            if not chunks:
                logger.error("No chunks created from documents.")
                return False
            
            chunks = self._process_chunks(chunks)
            if not chunks:
                logger.error("No valid chunks after filtering.")
                return False
            
            vectorstore = self._create_vectorstore(chunks)
            
            self._save_vectorstore(vectorstore)
            self._verify_build()
            
            build_time = time.time() - start_time
            logger.info(f"Vector store built successfully in {build_time:.2f} seconds")
            logger.info(f"   - Output: {self.config.output_path}")
            logger.info(f"   - Total chunks: {len(chunks)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Vector store build failed: {e}")
            return False
    
    def _load_documents(self) -> List[Document]:
        """Load documents from the docs folder"""
        logger.info(f"Loading documents from: {self.config.docs_folder}")
        
        if not self.config.docs_folder.exists():
            logger.error(f"Docs folder not found: {self.config.docs_folder}")
            return []
        
        documents = self.loader.load_documents(
            self.config.docs_folder,
            self.config.supported_formats
        )
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def _create_chunks(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        logger.info(f"Splitting documents into chunks...")
        
        splitter = self._get_text_splitter()
        chunks = splitter.split_documents(documents)
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _get_text_splitter(self):
        """Get text splitter based on configuration"""
        common_args = {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap
        }
        
        if self.config.chunking_strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(**common_args)
        elif self.config.chunking_strategy == ChunkingStrategy.CHARACTER:
            return CharacterTextSplitter(**common_args)
        elif self.config.chunking_strategy == ChunkingStrategy.MARKDOWN_AWARE:
            return MarkdownTextSplitter(**common_args)
        else:
            return RecursiveCharacterTextSplitter(**common_args)
    
    def _process_chunks(self, chunks: List[Document]) -> List[Document]:
        """Filter and clean chunks"""
        logger.info("Processing and filtering chunks...")
        
        chunks = filter_chunks(
            chunks, 
            self.config.min_chunk_length, 
            self.config.max_chunk_length
        )
        
        if self.config.remove_duplicates:
            chunks = remove_duplicate_chunks(chunks)
        
        for chunk in chunks:
            chunk.page_content = clean_text(chunk.page_content)
        
        logger.info(f"Processed {len(chunks)} final chunks")
        return chunks
    
    def _create_vectorstore(self, chunks: List[Document]) -> FAISS:
        """Create FAISS vector store from chunks"""
        logger.info("Creating embeddings and building vector store...")
        
        if len(chunks) > self.config.batch_size:
            logger.info(f"Processing {len(chunks)} chunks in batches of {self.config.batch_size}")
            
            first_batch = chunks[:self.config.batch_size]
            vectorstore = FAISS.from_documents(first_batch, self.embeddings)
            
            for i in range(self.config.batch_size, len(chunks), self.config.batch_size):
                batch = chunks[i:i + self.config.batch_size]
                batch_store = FAISS.from_documents(batch, self.embeddings)
                vectorstore.merge_from(batch_store)
                logger.info(f"Processed batch {i//self.config.batch_size + 1}")
        else:
            vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        logger.info(f"Vector store created with {vectorstore.index.ntotal} vectors")
        return vectorstore
    
    def _save_vectorstore(self, vectorstore: FAISS) -> None:
        """Save vector store to disk"""
        logger.info(f"Saving vector store to: {self.config.output_path}")
        
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        vectorstore.save_local(str(self.config.output_path))
        
        logger.info("Vector store saved successfully")
    
    def _verify_build(self) -> None:
        """Verify the built vector store"""
        logger.info("Verifying vector store...")
        
        from utils import verify_store
        results = verify_store(self.config.output_path)
        
        if results["searchable"]:
            logger.info("Vector store verification passed")
        else:
            logger.error(f"Vector store verification failed: {results.get('error', 'Unknown error')}")