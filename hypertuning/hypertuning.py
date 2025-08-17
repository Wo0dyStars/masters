import json
import httpx
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
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

OUTPUT_FILE = Path("results/hypertuning.jsonl")
EVAL_ENDPOINT = "http://localhost:8000/api/v1/naive-rag"

questions = [
    {
        "question": "What does the contract type tell you?",
        "ground_truth": "It defines the purpose and rules of the contract.",
        "complexity": "Simple Factual"
    },
    {
        "question": "What operational risks arise from unreturned or untracked access keys?",
        "ground_truth": "Unreturned keys can result in unauthorized access, breach of security, incorrect contract closure, and missing charges for replacement, all of which undermine auditability and customer accountability.",
        "complexity": "Complex Reasoning"
    },
    {
        "question": "What sequence of modules is involved in processing a customer's lift-in booking?",
        "ground_truth": "A lift-in booking links a boat and customer, may initiate a job, generate charges upon completion, and create transactions that reflect in the customer's financial records, all while blocking operational resources for the scheduled date.",
        "complexity": "Multi-document"
    },
    {
        "question": "What are the valid reading methods for electricity meters?",
        "ground_truth": "Manual, Automatic, Estimated, Customer-provided",
        "complexity": "Specific Details"
    }
]

configurations = [
    {"id": "A", "temperature": 0.0, "top_p": 1.0, "max_tokens": 100, "frequency_penalty": 0.0, "presence_penalty": 0.0},
    {"id": "B", "temperature": 0.7, "top_p": 0.9, "max_tokens": 100, "frequency_penalty": 0.0, "presence_penalty": 0.0},
    {"id": "C", "temperature": 0.9, "top_p": 0.85, "max_tokens": 100, "frequency_penalty": 0.5, "presence_penalty": 0.3},
    {"id": "D", "temperature": 0.5, "top_p": 0.8, "max_tokens": 100, "frequency_penalty": 0.0, "presence_penalty": 0.0},
    {"id": "E", "temperature": 0.3, "top_p": 1.0, "max_tokens": 80, "frequency_penalty": 0.0, "presence_penalty": 0.0},
    {"id": "F", "temperature": 1.0, "top_p": 0.95, "max_tokens": 120, "frequency_penalty": 0.8, "presence_penalty": 0.6},
]

def run():
    results = []
    
    for q in tqdm(questions, desc="Evaluating questions"):
        for config in configurations:
            payload = {
                "question": q["question"],
                "model": "gpt-4",
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "max_tokens": config["max_tokens"],
                "frequency_penalty": config["frequency_penalty"],
                "presence_penalty": config["presence_penalty"]
            }
            
            try:
                response = httpx.post(EVAL_ENDPOINT, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                
                output = {
                    "question": q["question"],
                    "ground_truth": q["ground_truth"],
                    "complexity": q["complexity"],
                    "answer": result["answer"],
                    "contexts": [doc["content_preview"] for doc in result.get("sources", [])],
                    "latency": result.get("latency"),
                    "tokens_used": result.get("tokens_used"),
                    "prompt_tokens": result.get("prompt_tokens"),
                    "completion_tokens": result.get("completion_tokens"),
                    "model": result.get("model"),
                    "cost_usd": result.get("cost_usd"),
                    "config_id": config["id"],
                    "temperature": config["temperature"],
                    "top_p": config["top_p"],
                    "max_tokens": config["max_tokens"],
                    "frequency_penalty": config["frequency_penalty"],
                    "presence_penalty": config["presence_penalty"]
                }
                
                results.append(output)
                
            except Exception as e:
                print(f"Error for: {q['question']} (config {config['id']})\n{e}\n")
                
    return results

def append_scores(data: list[dict]):
    """Add RAGAS evaluation scores with proper ground_truth mapping."""
    if not data:
        return data
    
    ragas_data = []
    for item in data:
        ragas_item = {
            "question": item["question"],
            "answer": item["answer"],
            "contexts": item["contexts"],
            "ground_truth": item["ground_truth"]
        }
        ragas_data.append(ragas_item)
    
    dataset = Dataset.from_list(ragas_data)
    
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
    
    for i, score_row in score_df.iterrows():
        for metric, value in score_row.items():
            if isinstance(value, (float, int)):
                data[i][metric] = round(value, 4)
            elif metric not in ("question", "answer", "contexts", "ground_truth"):
                data[i][metric] = value
    
    for item in data:
        for key in ["user_input", "response", "reference", "retrieved_contexts"]:
            item.pop(key, None)
    
    return data

def save_output(data: list[dict]):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with OUTPUT_FILE.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    raw_outputs = run()
    enriched_outputs = append_scores(raw_outputs)
    save_output(enriched_outputs)