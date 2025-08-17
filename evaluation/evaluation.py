import json
import httpx
import asyncio
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_similarity,
    faithfulness,
    answer_correctness,
    SemanticSimilarity,
    AnswerAccuracy,
    FactualCorrectness,
)
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import argparse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TOKEN_LIMITS = {
    "Simple Factual": 300,
    "Complex Reasoning": 600,
    "Multi-document": 800,
    "Specific Details": 400,
    "default": 500 
}

class RAGParadigm(str, Enum):
    NAIVE = "naive"
    ADVANCED = "advanced"
    AGENTIC = "agentic"

@dataclass
class EvaluationConfig:
    paradigm: RAGParadigm
    input_file: Path
    output_file: Path
    endpoint: str
    models: List[str]
    request_timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 2.0
    temperature: float = 1.0
    top_p: float = 0.95
    base_max_tokens: int = 500
    frequency_penalty: float = 0.8
    presence_penalty: float = 0.6

def get_max_tokens_for_complexity(complexity: str) -> int:
    """Get appropriate token limit based on question complexity."""
    return TOKEN_LIMITS.get(complexity, TOKEN_LIMITS["default"])

PARADIGM_CONFIGS = {
    RAGParadigm.NAIVE: EvaluationConfig(
        paradigm=RAGParadigm.NAIVE,
        input_file=Path("evaluation/data.json"),
        output_file=Path("results/iterations/fifth iteration/results/naive.jsonl"),
        endpoint="http://localhost:8000/api/v1/naive-rag",
        models=["gpt-4", "deepseek-chat"],
    ),
    RAGParadigm.ADVANCED: EvaluationConfig(
        paradigm=RAGParadigm.ADVANCED,
        input_file=Path("evaluation/data.json"),
        output_file=Path("results/iterations/fifth iteration/results/advanced.jsonl"),
        endpoint="http://localhost:8000/api/v1/advanced-rag",
        models=["gpt-4", "deepseek-chat"],
    ),
    RAGParadigm.AGENTIC: EvaluationConfig(
        paradigm=RAGParadigm.AGENTIC,
        input_file=Path("evaluation/data.json"),
        output_file=Path("results/iterations/fifth iteration/results/agentic.jsonl"),
        endpoint="http://localhost:8000/api/v1/agentic-rag",
        models=["gpt-4", "deepseek-chat"],
    )
}

def load_data(file_path: Path) -> List[Dict[str, Any]]:
    data = []
    
    try:
        with file_path.open("r", encoding="utf-8") as infile:
            if file_path.suffix == ".json":
                data = json.load(infile)
            else:
                for i, line in enumerate(infile, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON on line {i}: {e}")
                        continue
    
    except FileNotFoundError:
        logger.error(f"Evaluation data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading evaluation data: {e}")
        raise
    
    required_fields = ["question", "ground_truth"]
    valid_data = []
    
    for i, item in enumerate(data):
        if all(field in item for field in required_fields):
            valid_data.append(item)
        else:
            logger.warning(f"Skipping item {i}: missing required fields {required_fields}")
    
    return valid_data

def extract_contexts(result: Dict[str, Any], paradigm: RAGParadigm) -> List[str]:
    contexts = []
    
    if "contexts" in result and result["contexts"]:
        return result["contexts"]
    
    if "sources" in result:
        sources = result["sources"]
        if sources and isinstance(sources[0], dict):
            for doc in sources:
                if "content" in doc:
                    contexts.append(doc["content"])
                elif "content_preview" in doc:
                    contexts.append(doc["content_preview"])
                elif isinstance(doc, str):
                    contexts.append(doc)
        elif isinstance(sources, list) and sources:
            contexts = [str(source) for source in sources]
    
    return contexts

def create_payload(question: str, ground_truth: str, model: str, config: EvaluationConfig, complexity: str = "default") -> Dict[str, Any]:
    max_tokens = get_max_tokens_for_complexity(complexity)
    
    base_payload = {
        "question": question,
        "ground_truth": ground_truth,
        "model": model,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": max_tokens,
        "frequency_penalty": config.frequency_penalty,
        "presence_penalty": config.presence_penalty
    }
    
    if config.paradigm == RAGParadigm.ADVANCED:
        base_payload.update({
            "use_reranking": True,
            "use_query_expansion": True,
            "fusion_method": "weighted_sum",
            "context_window_size": 4000
        })
    
    return base_payload

class Evaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    async def call_api(self, question: str, ground_truth: str, model: str, complexity: str = "default") -> Optional[Dict[str, Any]]:
        payload = create_payload(question, ground_truth, model, self.config, complexity)
        
        logger.debug(f"Using {payload['max_tokens']} tokens for complexity: {complexity}")
        
        for attempt in range(self.config.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                    response = await client.post(self.config.endpoint, json=payload)
                    response.raise_for_status()
                    return response.json()
                    
            except httpx.TimeoutException:
                logger.warning(f"Timeout on attempt {attempt + 1} for {model}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    logger.warning(f"Server error {e.response.status_code} on attempt {attempt + 1}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay)
                        continue
                
            except Exception as e:
                logger.error(f"Unexpected error calling API: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
        
        return None
    
    async def run(self) -> List[Dict[str, Any]]:
        data = load_data(self.config.input_file)
        results = []
        total_iterations = len(data) * len(self.config.models)
        failed_requests = 0
        
        complexity_counts = {}
        for item in data:
            complexity = item.get("complexity", "default")
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        logger.info("Token limits by complexity:")
        for complexity, count in complexity_counts.items():
            tokens = get_max_tokens_for_complexity(complexity)
            logger.info(f"  {complexity}: {tokens} tokens ({count} questions)")
        
        with tqdm(total=total_iterations, desc=f"Evaluating {self.config.paradigm.value}") as pbar:
            for item in data:
                question = item["question"]
                ground_truth = item["ground_truth"]
                complexity = item.get("complexity", "default")
                
                for model in self.config.models:
                    pbar.set_description(f"Evaluating {model} - {complexity}")
                    
                    result = await self.call_api(question, ground_truth, model, complexity)
                    
                    if result:
                        contexts = extract_contexts(result, self.config.paradigm)
                        
                        output = {
                            "question": question,
                            "ground_truth": ground_truth,
                            "complexity": complexity,
                            "model": model,
                            "paradigm": self.config.paradigm.value,
                            "answer": result.get("answer", ""),
                            "contexts": contexts,
                            "max_tokens_used": get_max_tokens_for_complexity(complexity),
                            "latency": result.get("total_latency_seconds", result.get("latency_seconds", result.get("latency", 0))),
                            "tokens_used": result.get("total_tokens_used", result.get("tokens_used", 0)),
                            "prompt_tokens": result.get("prompt_tokens", 0),
                            "completion_tokens": result.get("completion_tokens", 0),
                            "cost_usd": result.get("cost_usd", 0),
                            "temperature": self.config.temperature,
                            "top_p": self.config.top_p,
                            "frequency_penalty": self.config.frequency_penalty,
                            "presence_penalty": self.config.presence_penalty
                        }
                        
                        if self.config.paradigm == RAGParadigm.AGENTIC:
                            output.update({
                                "similarity_score": result.get("similarity_score", 0),
                                "success": result.get("success", False),
                                "total_iterations": result.get("total_iterations", 1),
                                "best_iteration": result.get("best_iteration", 1)
                            })
                        elif self.config.paradigm == RAGParadigm.ADVANCED:
                            output.update({
                                "expanded_queries": result.get("expanded_queries"),
                                "fusion_method": result.get("fusion_method"),
                                "context_length": result.get("context_length"),
                                "num_documents_retrieved": result.get("num_documents_retrieved")
                            })
                        
                        results.append(output)
                    else:
                        failed_requests += 1
                        logger.error(f"Failed to get result for {model} on question: {question[:50]}...")
                    
                    pbar.update(1)
        
        if failed_requests > 0:
            logger.warning(f"Total failed requests: {failed_requests}")
        
        return results
    
    def add_scores(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not data:
            logger.warning("No data to evaluate")
            return data
        
        try:
            logger.info("🔍 Running evaluation...")
            eval_data = []
            for item in data:
                eval_item = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "ground_truth": item["ground_truth"],
                    "contexts": item.get("contexts", [])
                }
                eval_data.append(eval_item)
            
            dataset = Dataset.from_list(eval_data)
            
            scores = evaluate(
                dataset,
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_similarity,
                    answer_correctness,
                    SemanticSimilarity(),
                    AnswerAccuracy(),
                    FactualCorrectness(),
                ]
            )
            
            score_df = scores.to_pandas()
            for i, row in score_df.iterrows():
                if i < len(data): 
                    for metric, value in row.items():
                        if isinstance(value, (float, int)) and metric not in ["question", "answer", "ground_truth", "contexts"]:
                            data[i][metric] = round(value, 4) if isinstance(value, float) else value
                        elif metric not in ("user_input", "response", "reference", "retrieved_contexts"):
                            if not isinstance(value, (str, list, dict)) or metric in ["question", "answer", "ground_truth", "contexts"]:
                                continue
                            data[i][metric] = value
            
            for item in data:
                for key in ["user_input", "response", "reference", "retrieved_contexts"]:
                    item.pop(key, None)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
        
        return data
    
    def save_results(self, data: List[Dict[str, Any]]) -> None:
        if not data:
            logger.warning("No data to save")
            return
        
        try:
            self.config.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with self.config.output_file.open("w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            logger.info(f"Output saved to {self.config.output_file}")
            
            complexity_stats = {}
            for item in data:
                complexity = item.get("complexity", "default")
                if complexity not in complexity_stats:
                    complexity_stats[complexity] = {
                        "count": 0,
                        "avg_completion_tokens": 0,
                        "max_tokens_available": get_max_tokens_for_complexity(complexity)
                    }
                complexity_stats[complexity]["count"] += 1
                complexity_stats[complexity]["avg_completion_tokens"] += item.get("completion_tokens", 0)
            
            logger.info("Token usage summary by complexity:")
            for complexity, stats in complexity_stats.items():
                if stats["count"] > 0:
                    avg_tokens = stats["avg_completion_tokens"] / stats["count"]
                    utilization = (avg_tokens / stats["max_tokens_available"]) * 100
                    logger.info(f"  {complexity}: {avg_tokens:.1f} avg tokens used / {stats['max_tokens_available']} available ({utilization:.1f}% utilization)")
            
        except Exception as e:
            logger.error(f"Error saving output: {e}")

async def main():
    parser = argparse.ArgumentParser(description="Unified RAG Evaluation System")
    parser.add_argument(
        "paradigm", 
        choices=["naive", "advanced", "agentic", "all"],
        help="RAG paradigm to evaluate"
    )
    
    args = parser.parse_args()
    
    paradigms_to_evaluate = (
        list(RAGParadigm) if args.paradigm == "all" else [RAGParadigm(args.paradigm)]
    )
    
    for paradigm in paradigms_to_evaluate:
        try:
            config = PARADIGM_CONFIGS[paradigm]
            evaluator = Evaluator(config)
            results = await evaluator.run()
            
            if not results:
                logger.error(f"No results for {paradigm.value} RAG")
                continue
            
            results = evaluator.add_scores(results)
            evaluator.save_results(results)
            
        except Exception as e:
            logger.error(f"{paradigm.value.title()} RAG evaluation failed: {e}")
            continue

if __name__ == "__main__":
    asyncio.run(main())