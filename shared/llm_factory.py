import os
from langchain_openai import ChatOpenAI
from .config import ModelType

def get_llm(
    model: ModelType,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_tokens: int = 120,
    frequency_penalty: float = 0.8,
    presence_penalty: float = 0.6
) -> ChatOpenAI:
    """Factory function to create LLM instances"""
    
    base_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty
    }
    
    if model == ModelType.DEEPSEEK:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")
        
        return ChatOpenAI(
            model="deepseek-chat",
            openai_api_base="https://api.deepseek.com/v1",
            openai_api_key=api_key,
            **base_params
        )
    
    return ChatOpenAI(model=model.value, **base_params)