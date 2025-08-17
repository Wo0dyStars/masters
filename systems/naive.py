import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field, model_validator
from fastapi import APIRouter, HTTPException
from langchain.schema import Document

from shared.vector_store import load_vector_store
from shared.prompts import DEFAULT_RAG_PROMPT
from shared.llm_factory import get_llm
from shared.context import truncate_context
from shared.costs import calculate_cost
from shared.save_to_file import save_to_file
from shared.config import ModelType

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class NaiveRAGConfig:
    retrieval_k: int = 5
    max_context_length: int = 4000
    timeout_seconds: int = 30
    vector_store_path: str = "vector_store/stores/md_medium_recursive"

DEFAULT_CONFIG = NaiveRAGConfig()

class SourceDocument(BaseModel):
    metadata: Dict[str, Any]
    content_preview: str
    relevance_score: float

class NaiveRAGQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    model: ModelType = Field(default=ModelType.GPT4)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=120, ge=50, le=4000)
    frequency_penalty: float = Field(default=0.8, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.6, ge=-2.0, le=2.0)
    config: Optional[NaiveRAGConfig] = None

    @model_validator(mode="after")
    def apply_default_config(self) -> "NaiveRAGQuery":
        if self.config is None:
            self.config = DEFAULT_CONFIG
        return self

class NaiveRAGResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    latency_seconds: float
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str
    timestamp: datetime
    parameters: Dict[str, Any]
    contexts: List[str]
    retrieval_k: int

class NaiveRAGProcessor:
    def __init__(self, vector_store_path: str = "vector_store/stores/md_medium_recursive") -> None:
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.embeddings = None
        self._load_vector_store()
        self.prompt_template = DEFAULT_RAG_PROMPT
    
    def _load_vector_store(self) -> None:
        try:
            self.vector_store, self.embeddings = load_vector_store(self.vector_store_path)
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            self.vector_store = None
            self.embeddings = None
    
    def retrieve_documents(self, question: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for the question"""
        if not self.vector_store:
            logger.warning("Vector store not initialized, returning empty list")
            return []
        
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            docs = retriever.get_relevant_documents(question)
            
            if not docs:
                logger.warning(f"No documents retrieved for question: {question[:50]}...")
            
            return docs
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def generate_answer(self, query: NaiveRAGQuery, contexts: List[str]) -> Tuple[str, Dict[str, Any], float]:
        """Generate answer using LLM"""
        try:
            llm = get_llm(
                model=query.model,
                temperature=query.temperature,
                top_p=query.top_p,
                max_tokens=query.max_tokens,
                frequency_penalty=query.frequency_penalty,
                presence_penalty=query.presence_penalty
            )
            
            context_text = "\n\n".join(contexts)
            full_prompt = self.prompt_template.format(
                context=context_text,
                question=query.question
            )
            
            start_time = time.time()
            response = llm.invoke(full_prompt)
            latency = time.time() - start_time
            
            token_usage = getattr(response, "response_metadata", {}).get("token_usage", {})
            
            return response.content, token_usage, latency
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}", {}, 0.0
    
    def process(self, query: NaiveRAGQuery) -> NaiveRAGResponse:
        """Main processing function"""
        try:
            docs = self.retrieve_documents(query.question, query.config.retrieval_k)
            
            if docs:
                contexts = [doc.page_content for doc in docs]
                contexts = truncate_context(contexts, query.config.max_context_length)
            else:
                contexts = ["No relevant documents found."]
            
            answer, token_usage, latency = self.generate_answer(query, contexts)
            
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
            
            cost = calculate_cost(query.model, prompt_tokens, completion_tokens)
            
            sources = []
            for doc in docs:
                sources.append(SourceDocument(
                    metadata=doc.metadata,
                    content_preview=doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
                    relevance_score=float(getattr(doc, 'score', 0.0) or 0.0)
                ))
            
            response = NaiveRAGResponse(
                answer=answer,
                sources=sources,
                latency_seconds=round(latency, 3),
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=round(cost, 6),
                model=str(query.model),
                timestamp=datetime.now(),
                parameters={
                    "temperature": query.temperature,
                    "top_p": query.top_p,
                    "max_tokens": query.max_tokens,
                    "frequency_penalty": query.frequency_penalty,
                    "presence_penalty": query.presence_penalty
                },
                contexts=contexts,
                retrieval_k=query.config.retrieval_k
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return NaiveRAGResponse(
                answer=f"Processing failed: {str(e)}",
                sources=[],
                latency_seconds=0.0,
                tokens_used=0,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                model=str(query.model),
                timestamp=datetime.now(),
                parameters={},
                contexts=[],
                retrieval_k=query.config.retrieval_k if query.config else 5
            )

router = APIRouter()
processor = None

def get_processor() -> NaiveRAGProcessor:
    """Get or create processor instance"""
    global processor
    if processor is None:
        processor = NaiveRAGProcessor()
    return processor

@router.post("/naive-rag", response_model=NaiveRAGResponse)
def run_naive_rag(query: NaiveRAGQuery) -> NaiveRAGResponse:
    """
    Execute naive RAG process
    
    This endpoint performs a straightforward RAG process:
    1. Retrieves relevant documents based on the question
    2. Generates an answer using the retrieved context
    3. Returns the answer with metadata and performance metrics
    """
    try:
        rag_processor = get_processor()
        result = rag_processor.process(query)
        
        try:
            save_to_file(result, "evaluation/runouts/naive", "naive_rag")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error in naive RAG: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")