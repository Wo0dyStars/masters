import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field, model_validator
from fastapi import APIRouter, HTTPException
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from shared.vector_store import load_vector_store
from shared.prompts import DEFAULT_RAG_PROMPT
from shared.llm_factory import get_llm
from shared.costs import calculate_cost
from shared.save_to_file import save_to_file
from shared.config import ModelType

logger = logging.getLogger(__name__)

class FusionMethod(str, Enum):
    RECIPROCAL_RANK = "reciprocal_rank"
    WEIGHTED_SUM = "weighted_sum"

@dataclass(frozen=True)
class AdvancedRAGConfig:
    retrieval_k: int = 7
    max_context_length: int = 4000
    timeout_seconds: int = 45
    vector_store_path: str = "vector_store/stores/md_medium_recursive"
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    similarity_weight: float = 0.7
    keyword_weight: float = 0.2
    content_quality_weight: float = 0.1
    min_similarity_threshold: float = 0.08
    max_parts_per_document: int = 3

DEFAULT_CONFIG = AdvancedRAGConfig()

class SourceDocument(BaseModel):
    metadata: Dict[str, Any]
    content_preview: str
    relevance_score: str
    similarity_score: Optional[float] = None

class AdvancedRAGQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    model: ModelType = Field(default=ModelType.GPT4)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=120, ge=50, le=4000)
    frequency_penalty: float = Field(default=0.8, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.6, ge=-2.0, le=2.0)
    use_reranking: bool = Field(default=True)
    use_query_expansion: bool = Field(default=True)
    fusion_method: FusionMethod = Field(default=FusionMethod.RECIPROCAL_RANK)
    context_window_size: int = Field(default=4000, ge=1000, le=8000)
    config: Optional[AdvancedRAGConfig] = None

    @model_validator(mode="after")
    def apply_default_config(self) -> "AdvancedRAGQuery":
        if self.config is None:
            self.config = DEFAULT_CONFIG
        return self

class AdvancedRAGResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    contexts: Optional[List[str]] = None
    latency_seconds: float
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    expanded_queries: Optional[List[str]] = None
    fusion_method: str
    context_length: int
    num_documents_retrieved: int
    model: str
    timestamp: datetime
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    parameters: Dict[str, Any]
    processing_steps: Dict[str, Any]

class QueryExpander:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.expansion_prompt = PromptTemplate.from_template(
            """Rephrase the question into 3 specific variants for better document retrieval.
Targets: (1) what it is, (2) how it works, (3) where or when it's used.
Return numbered questions only. No explanations, markdown, or formatting.

Original question: What does 'charge type' mean?
Alternative questions:
1. What are the available charge types and their definitions?
2. How does charge type determine billing calculations?
3. In which scenarios is each charge type applied?

Original question: {query}
Alternative questions:
1.
2.
3."""
        )

    def expand_query(self, query: str) -> List[str]:
        """Expands the given query into up to 3 specific alternatives using the LLM"""
        try:
            logger.debug(f"Expanding query: {query[:50]}...")
            response = self.llm.invoke(self.expansion_prompt.format(query=query))
            content = response.content.strip()

            lines = content.splitlines()
            alternatives = []

            for line in lines:
                line = line.strip()
                if line and any(line.startswith(prefix) for prefix in ("1.", "2.", "3.", "- ", "• ")):
                    cleaned = line.split(maxsplit=1)[-1].strip()
                elif len(line) > 15:
                    cleaned = line.strip()
                else:
                    continue

                if cleaned and len(cleaned) > 10:
                    alternatives.append(cleaned)

            expanded = [query] + alternatives[:3]
            logger.debug(f"Expanded to {len(expanded)} queries")
            return expanded

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]

class SemanticReranker:
    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        similarity_weight: float = 0.7,
        keyword_weight: float = 0.2,
        content_quality_weight: float = 0.1
    ):
        self.embeddings = embeddings
        self.similarity_weight = similarity_weight
        self.keyword_weight = keyword_weight
        self.content_quality_weight = content_quality_weight

    def rerank_documents(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Reranks retrieved documents based on combined scoring system"""
        if not documents:
            return []
        
        try:
            logger.debug(f"Reranking {len(documents)} documents")
            query_emb = np.array(self.embeddings.embed_query(query))
            query_words = set(query.lower().split())
            scored_docs = []
            
            for doc in documents:
                try:
                    doc_emb = np.array(self.embeddings.embed_query(doc.page_content))
                    similarity = np.dot(query_emb, doc_emb) / (
                        np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
                    )
                    doc_words = set(doc.page_content.lower().split())
                    keyword_overlap = len(query_words & doc_words) / len(query_words | doc_words)
                    content_quality = min(len(doc.page_content) / 500, 1.0)
                    
                    final_score = (
                        self.similarity_weight * similarity +
                        self.keyword_weight * keyword_overlap +
                        self.content_quality_weight * content_quality
                    )
                    scored_docs.append((doc, final_score, similarity))
                except Exception as scoring_error:
                    logger.warning(f"Failed to score document: {scoring_error}")
                    scored_docs.append((doc, 0.0, 0.0))
            
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, _, _ in scored_docs[:top_k]]
            logger.debug(f"Selected top {len(top_docs)} documents after reranking")
            return top_docs
            
        except Exception as e:
            logger.error(f"Document reranking failed: {e}")
            return documents[:top_k]

class AdvancedRAGProcessor:
    def __init__(self, vector_store_path: str = "vector_store/stores/md_medium_recursive"):
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.embeddings = None
        self.reranker = None
        self.query_expander = None
        self._initialize_system()
        self.prompt_template = DEFAULT_RAG_PROMPT

    def _initialize_system(self) -> None:
        """Initialize vector store and reranker"""
        try:
            self.vector_store, self.embeddings = load_vector_store(self.vector_store_path)
            self.reranker = SemanticReranker(self.embeddings)
            logger.info(f"Advanced RAG system initialized with vector store: {self.vector_store_path}")
        except Exception as e:
            logger.error(f"Failed to initialize advanced RAG system: {e}")
            self.vector_store = None
            self.embeddings = None

    def _retrieve_and_rank(self, query: str, top_k: int = 10) -> List[Any]:
        """Simple retrieval with reranking"""
        if not self.vector_store:
            return []
            
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": top_k * 2}
            )
            
            docs = retriever.get_relevant_documents(query)
            
            if self.reranker and docs:
                docs = self.reranker.rerank_documents(query, docs, top_k)
            
            logger.debug(f"Retrieved and ranked {len(docs)} documents")
            return docs
            
        except Exception as e:
            logger.error(f"Retrieval and ranking failed: {e}")
            return []

    def _multi_query_retrieval(self, original_query: str, docs_per_query: int = 4) -> List[Any]:
        """Retrieve documents using multiple query variations with deduplication"""
        all_docs = []
        queries_used = [original_query]
        
        try:
            expanded_queries = self.query_expander.expand_query(original_query)
            queries_used = expanded_queries
            logger.debug(f"Using {len(queries_used)} queries: {[q[:50] + '...' for q in queries_used]}")
        except Exception as e:
            logger.warning(f"Query expansion failed, using original query: {e}")
            queries_used = [original_query]
        
        for i, query in enumerate(queries_used):
            try:
                docs = self._retrieve_and_rank(query, top_k=docs_per_query)
                for doc in docs:
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata['source_query'] = f"Query {i+1}"
                    doc.metadata['query_text'] = query[:100]
                all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Retrieval failed for query '{query[:50]}': {e}")
                continue
        
        seen_hashes = set()
        unique_docs = []
        for doc in all_docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        
        logger.debug(f"Retrieved {len(all_docs)} total docs, {len(unique_docs)} unique after deduplication")
        return unique_docs[:10]

    def _create_context(self, documents: List[Any], max_length: int = 4000) -> str:
        """Simple context creation - just concatenate until limit"""
        context_parts = []
        current_length = 0
        
        for doc in documents:
            doc_content = doc.page_content
            if current_length + len(doc_content) <= max_length:
                context_parts.append(doc_content)
                current_length += len(doc_content)
            else:
                remaining_space = max_length - current_length
                if remaining_space > 200:
                    context_parts.append(doc_content[:remaining_space-3] + "...")
                break
        
        return "\n\n".join(context_parts)

    def process(self, query: AdvancedRAGQuery) -> AdvancedRAGResponse:
        """Process an advanced RAG query"""
        start_time = time.time()
        
        try:
            logger.info(f"Processing advanced RAG query: {query.question[:50]}...")
            
            llm = get_llm(
                model=query.model,
                temperature=query.temperature,
                top_p=query.top_p,
                max_tokens=query.max_tokens,
                frequency_penalty=query.frequency_penalty,
                presence_penalty=query.presence_penalty
            )

            if not self.query_expander:
                expansion_llm = get_llm(
                    model=query.model,
                    temperature=0.3,
                    max_tokens=200,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                self.query_expander = QueryExpander(expansion_llm)
            
            retrieved_docs = self._multi_query_retrieval(
                query.question,
                docs_per_query=4
            )
            
            if not retrieved_docs:
                retrieved_docs = []
                context = "No relevant documents found."
            else:
                context = self._create_context(retrieved_docs, query.context_window_size)
            
            full_prompt = self.prompt_template.format(
                context=context,
                question=query.question
            )
            
            response = llm.invoke(full_prompt)
            
            total_latency = time.time() - start_time
            token_usage = getattr(response, "response_metadata", {}).get("token_usage", {})
            
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
            
            cost = calculate_cost(query.model, prompt_tokens, completion_tokens)
            
            sources = []
            for i, doc in enumerate(retrieved_docs):
                source = SourceDocument(
                    metadata=doc.metadata,
                    content_preview=(
                        doc.page_content[:300] + "..." 
                        if len(doc.page_content) > 300 
                        else doc.page_content
                    ),
                    relevance_score=f"Rank {i+1}",
                    similarity_score=getattr(doc, 'similarity_score', None)
                )
                sources.append(source)
            
            response_obj = AdvancedRAGResponse(
                answer=response.content,
                sources=sources,
                latency_seconds=round(total_latency, 3),
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=round(cost, 6),
                expanded_queries=None,
                fusion_method="reranking_only",
                context_length=len(context),
                num_documents_retrieved=len(retrieved_docs),
                model=str(query.model),
                timestamp=datetime.now(),
                parameters={
                    "temperature": query.temperature,
                    "top_p": query.top_p,
                    "max_tokens": query.max_tokens,
                    "frequency_penalty": query.frequency_penalty,
                    "presence_penalty": query.presence_penalty,
                    "use_reranking": True,
                    "use_query_expansion": True,
                    "context_window_size": query.context_window_size
                },
                processing_steps={
                    "retrieval_and_ranking": {
                        "documents_retrieved": len(retrieved_docs),
                        "method": "vector_similarity_with_reranking"
                    }
                },
                temperature=query.temperature,
                top_p=query.top_p,
                max_tokens=query.max_tokens,
                frequency_penalty=query.frequency_penalty,
                presence_penalty=query.presence_penalty,
                contexts=[doc.page_content for doc in retrieved_docs]
            )
            
            logger.info(f"Advanced RAG completed in {total_latency:.3f}s")
            return response_obj
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return AdvancedRAGResponse(
                answer=f"Processing failed: {str(e)}",
                sources=[],
                latency_seconds=0.0,
                tokens_used=0,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
                expanded_queries=None,
                fusion_method="error",
                context_length=0,
                num_documents_retrieved=0,
                model=str(query.model),
                timestamp=datetime.now(),
                parameters={},
                processing_steps={},
                contexts=[]
            )

router = APIRouter()
processor = None

def get_processor() -> AdvancedRAGProcessor:
    """Get or create processor instance"""
    global processor
    if processor is None:
        processor = AdvancedRAGProcessor()
    return processor

@router.post("/advanced-rag", response_model=AdvancedRAGResponse)
def run_advanced_rag(query: AdvancedRAGQuery) -> AdvancedRAGResponse:
    """
    Execute advanced RAG process
    
    This endpoint performs a sophisticated RAG process that includes:
    1. Query expansion for better retrieval coverage
    2. Hybrid search combining dense and sparse retrieval
    3. Result fusion using multiple ranking strategies
    4. Semantic reranking of retrieved documents
    5. Context optimization for improved answer generation
    6. Comprehensive performance tracking and analysis
    """
    try:
        rag_processor = get_processor()
        result = rag_processor.process(query)
        
        try:
            save_to_file(result, "evaluation/runouts/advanced", "advanced_rag")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error in advanced RAG: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")