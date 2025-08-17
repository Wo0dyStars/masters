import asyncio
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field, model_validator
from fastapi import APIRouter, HTTPException
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from ragas import evaluate
from ragas.metrics import SemanticSimilarity
from datasets import Dataset
import pandas as pd

from shared.vector_store import load_vector_store
from shared.prompts import DEFAULT_RAG_PROMPT
from shared.context import truncate_context
from shared.llm_factory import get_llm
from shared.costs import calculate_cost
from shared.save_to_file import save_to_file
from shared.config import ModelType

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class AgenticConfig:
    max_iterations: int = 3
    similarity_threshold: float = 0.95
    retrieval_k: int = 5
    max_context_length: int = 4000
    timeout_seconds: int = 30

DEFAULT_CONFIG = AgenticConfig()

class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        pass

class IterationResult(BaseModel):
    iteration: int
    query: str
    answer: str
    similarity_score: float
    latency_seconds: float
    contexts: List[str]
    sources: List[Dict[str, Any]]
    tokens_used: int
    reasoning: Optional[str] = None

class AgenticRAGQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    ground_truth: str = Field(..., min_length=1)
    model: ModelType = Field(default=ModelType.GPT4)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=120, ge=50, le=4000)
    frequency_penalty: float = Field(default=0.8, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.6, ge=-2.0, le=2.0)
    config: Optional[AgenticConfig] = None

    @model_validator(mode="after")
    def apply_default_config(self) -> "AgenticRAGQuery":
        if self.config is None:
            self.config = DEFAULT_CONFIG
        return self

class AgenticRAGResponse(BaseModel):
    answer: str
    final_query_used: str
    similarity_score: float
    contexts: List[str]
    sources: List[Dict[str, Any]]
    total_latency_seconds: float
    total_tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model: str
    success: bool
    best_iteration: int
    total_iterations: int
    threshold: float
    timestamp: datetime
    attempts: List[IterationResult]

class EvaluatorAgent(BaseAgent):
    def __init__(self):
        super().__init__("EvaluatorAgent")
        self.metric = SemanticSimilarity()
    
    async def score(self, answer: str, ground_truth: str, question: str, contexts: List[str]) -> float:
        """Calculate semantic similarity score between answer and ground truth"""
        if not answer or not ground_truth:
            return 0.0
            
        try:
            dataset = Dataset.from_list([{
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts": contexts or ["No context provided"]
            }])
            
            result = await asyncio.to_thread(evaluate, dataset, metrics=[self.metric])
            df = result.to_pandas()
            
            if not df.empty and pd.notna(df.iloc[0]['semantic_similarity']):
                return float(df.iloc[0]['semantic_similarity'])
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 0.0
    
    async def run(self, answer: str, ground_truth: str, question: str, contexts: List[str]) -> float:
        return await self.score(answer, ground_truth, question, contexts)

class ReasonerAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI):
        super().__init__("ReasonerAgent")
        self.llm = llm
        self.reasoning_prompt = PromptTemplate.from_template(
"""You are a ReAct agent for marina management documentation queries. Use Thought-Action-Observation reasoning.

CURRENT SITUATION:
Query: {query}
Previous Answer: {answer_preview}
Similarity Score: {similarity:.3f} (Target: {threshold:.3f})
Iteration: {iteration}

THOUGHT: Analyze what went wrong and what action to take.

ACTION: Choose ONE action:
1. REFINE_QUERY - Make the query more specific for marina documentation
2. ADJUST_FOCUS - Change what aspect of the topic to emphasize  
3. ADD_CONTEXT - Add marina/yacht haven context to the query
4. STOP - Current answer is good enough

OBSERVATION: Based on the low similarity score, I need to improve retrieval.

Respond in this EXACT JSON format:
{{
  "thought": "Brief analysis of the issue (max 150 chars)",
  "action": "REFINE_QUERY|ADJUST_FOCUS|ADD_CONTEXT|STOP",
  "new_query": "improved query or original if STOP",
  "reasoning": "why this action will help (max 100 chars)"
}}

Focus on marina/yacht haven operations, billing, contracts, boats, berths, jobs, and customers."""
        )
    
    async def reason(self, query: str, answer: str, similarity: float, threshold: float, iteration: int) -> Dict[str, Any]:
        """ReAct-style reasoning for query improvement"""
        try:
            answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
            
            prompt = self.reasoning_prompt.format(
                query=query,
                answer_preview=answer_preview,
                similarity=similarity,
                threshold=threshold,
                iteration=iteration
            )
            
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            content = response.content.strip()
            
            json_str = self._extract_json_from_response(content)
            if not json_str:
                return self._fallback_response(query)
            
            parsed = json.loads(json_str)
            
            valid_actions = ["REFINE_QUERY", "ADJUST_FOCUS", "ADD_CONTEXT", "STOP"]
            if parsed.get("action") not in valid_actions:
                parsed["action"] = "STOP"
            
            if parsed.get("action") == "STOP" or similarity > 0.7:
                parsed["new_query"] = query
            
            return parsed
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"ReAct reasoning failed: {e}")
            return self._fallback_response(query)
        except Exception as e:
            logger.error(f"ReAct reasoning error: {e}")
            return self._fallback_response(query)
    
    def _fallback_response(self, query: str) -> Dict[str, Any]:
        return {
            "thought": "Reasoning failed, using original query",
            "action": "STOP", 
            "new_query": query,
            "reasoning": "Fallback due to parsing error"
        }
    
    def _extract_json_from_response(self, content: str) -> Optional[str]:
        if "```json" in content:
            try:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    return content[start:end].strip()
            except:
                pass
        
        try:
            start = content.find("{")
            if start != -1:
                brace_count = 0
                for i, char in enumerate(content[start:], start):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            return content[start:i+1]
        except:
            pass
    
    async def run(self, query: str, answer: str, similarity: float, threshold: float, iteration: int = 1) -> Dict[str, Any]:
        return await self.reason(query, answer, similarity, threshold, iteration)

class RAGAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI, retriever, config: AgenticConfig):
        super().__init__("RAGAgent")
        self.llm = llm
        self.retriever = retriever
        self.config = config
        self.default_prompt = DEFAULT_RAG_PROMPT
    
    async def run(self, question: str, custom_prompt: Optional[str] = None) -> Tuple[str, Dict, List[str], List[Dict]]:
        """Handles document retrieval and answer generation"""
        try:
            docs = await self._retrieve_documents(question)
            if not docs:
                docs = []
                contexts = ["No relevant documents found."]
            else:
                contexts = [doc.page_content for doc in docs]
                contexts = truncate_context(contexts, self.config.max_context_length)
            
            context_text = "\n\n".join(contexts)
            
            prompt_template = (
                PromptTemplate.from_template(custom_prompt) 
                if custom_prompt else self.default_prompt
            )
            prompt = prompt_template.format(context=context_text, question=question)
            
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            usage = getattr(response, "response_metadata", {}).get("token_usage", {})
            
            sources = [
                {
                    "metadata": doc.metadata,
                    "content_preview": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
                    "relevance_score": float(getattr(doc, 'score', 0.0) or 0.0)
                }
                for doc in docs
            ]
            
            return response.content, usage, contexts, sources
            
        except Exception as e:
            logger.error(f"RAG execution failed: {e}")
            return f"RAG execution failed: {str(e)}", {}, [], []
    
    async def _retrieve_documents(self, question: str) -> List[Document]:
        try:
            return await asyncio.to_thread(self.retriever.invoke, question)
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []

class AgenticRAGOrchestrator:
    def __init__(self, query: AgenticRAGQuery):
        self.query = query
        self.config = query.config
        self.llm = get_llm(
            model=query.model,
            temperature=query.temperature,
            top_p=query.top_p,
            max_tokens=query.max_tokens,
            frequency_penalty=query.frequency_penalty,
            presence_penalty=query.presence_penalty
        )
        
        try:
            vector_store, _ = load_vector_store()
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.config.retrieval_k}
            )
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            retriever = None
        
        self.rag_agent = RAGAgent(self.llm, retriever, self.config)
        self.evaluator = EvaluatorAgent()
        self.reasoner = ReasonerAgent(self.llm)
        
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.history: List[IterationResult] = []
    
    async def run(self) -> AgenticRAGResponse:
        """Execute agentic RAG with iterative improvement"""
        query = self.query.question
        best_result = None
        best_score = -1.0
        total_latency = 0.0
        
        logger.info(f"Starting agentic RAG for query: {query[:100]}...")
        
        for iteration in range(self.config.max_iterations):
            start_time = time.time()
            
            try:
                answer, usage, contexts, sources = await self.rag_agent.run(query)
                iteration_latency = time.time() - start_time
                total_latency += iteration_latency
                
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                
                score = await self.evaluator.score(
                    answer, self.query.ground_truth, query, contexts
                )
                
                iteration_result = IterationResult(
                    iteration=iteration + 1,
                    query=query,
                    answer=answer,
                    similarity_score=score,
                    latency_seconds=round(iteration_latency, 3),
                    contexts=contexts,
                    sources=sources,
                    tokens_used=prompt_tokens + completion_tokens
                )
                
                self.history.append(iteration_result)
                
                if score > best_score:
                    best_result = iteration_result
                    best_score = score
                
                logger.info(f"Iteration {iteration + 1}: Score {score:.3f} (Best: {best_score:.3f})")
                
                if score >= self.config.similarity_threshold:
                    logger.info(f"Threshold reached at iteration {iteration + 1}")
                    break
                
                if iteration < self.config.max_iterations - 1:
                    reasoning_result = await self.reasoner.run(
                        query, answer, score, self.config.similarity_threshold, iteration + 1
                    )
                    
                    if reasoning_result.get("action") != "STOP":
                        query = reasoning_result.get("new_query", query)
                    
                    self.history[-1].reasoning = reasoning_result.get("reasoning", "No reasoning")
                
            except Exception as e:
                logger.error(f"Iteration {iteration + 1} failed: {e}")
                if iteration == 0:
                    best_result = IterationResult(
                        iteration=1,
                        query=query,
                        answer=f"Processing failed: {str(e)}",
                        similarity_score=0.0,
                        latency_seconds=0.0,
                        contexts=[],
                        sources=[],
                        tokens_used=0
                    )
                    best_score = 0.0
                break
        
        if not best_result:
            best_result = IterationResult(
                iteration=1,
                query=query,
                answer="No successful iterations completed",
                similarity_score=0.0,
                latency_seconds=0.0,
                contexts=[],
                sources=[],
                tokens_used=0
            )
            best_score = 0.0
        
        cost = calculate_cost(
            model=self.query.model,
            prompt_tokens=self.total_prompt_tokens,
            completion_tokens=self.total_completion_tokens
        )
        success = best_score >= self.config.similarity_threshold
        
        return AgenticRAGResponse(
            answer=best_result.answer,
            final_query_used=best_result.query,
            similarity_score=round(best_score, 4),
            contexts=best_result.contexts,
            sources=best_result.sources,
            total_latency_seconds=round(total_latency, 3),
            total_tokens_used=self.total_prompt_tokens + self.total_completion_tokens,
            prompt_tokens=self.total_prompt_tokens,
            completion_tokens=self.total_completion_tokens,
            model=str(self.query.model),
            cost_usd=round(cost, 6),
            success=success,
            best_iteration=best_result.iteration,
            total_iterations=len(self.history),
            threshold=self.config.similarity_threshold,
            attempts=self.history,
            timestamp=datetime.now()
        )

router = APIRouter()

@router.post("/agentic-rag", response_model=AgenticRAGResponse)
async def run_agentic_rag(query: AgenticRAGQuery) -> AgenticRAGResponse:
    """
    Execute agentic RAG process with iterative improvement
    
    This endpoint runs a multi-iteration RAG process that:
    1. Retrieves relevant documents
    2. Generates an answer
    3. Evaluates answer quality
    4. Refines the query if needed
    5. Repeats until quality threshold is met or max iterations reached
    """
    try:
        logger.info(f"Received agentic RAG request for model: {query.model}")
        
        orchestrator = AgenticRAGOrchestrator(query)
        
        result = await asyncio.wait_for(
            orchestrator.run(),
            timeout=query.config.timeout_seconds * query.config.max_iterations
        )
        
        try:
            save_to_file(result, "evaluation/runouts/agentic", "agentic_rag")
        except Exception as e:
            logger.warning(f"Failed to save results: {e}")
        
        logger.info(f"Agentic RAG completed successfully. Score: {result.similarity_score}")
        return result
        
    except asyncio.TimeoutError:
        logger.error("Agentic RAG process timed out")
        raise HTTPException(
            status_code=504, 
            detail="Request timed out. Try reducing max_iterations or increasing timeout."
        )
    except Exception as e:
        logger.error(f"Unexpected error in agentic RAG: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")