from dataclasses import dataclass
from enum import Enum
from typing import Dict

class ModelType(str, Enum):
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT35_TURBO = "gpt-3.5-turbo"
    DEEPSEEK = "deepseek-chat"

@dataclass(frozen=True)
class ModelPricing:
    prompt: float
    completion: float

PRICING: Dict[ModelType, ModelPricing] = {
    ModelType.GPT4: ModelPricing(prompt=0.03, completion=0.06),
    ModelType.GPT4_TURBO: ModelPricing(prompt=0.01, completion=0.03),
    ModelType.GPT35_TURBO: ModelPricing(prompt=0.002, completion=0.002),
    ModelType.DEEPSEEK: ModelPricing(prompt=0.00027, completion=0.00110),
}