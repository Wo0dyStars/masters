from shared.config import PRICING, ModelType

def calculate_cost(model: ModelType, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the total cost in USD for a model inference based on token usage.

    Args:
        model (ModelType): The LLM model used.
        prompt_tokens (int): Number of tokens in the prompt.
        completion_tokens (int): Number of tokens in the completion.

    Returns:
        float: Total cost in USD.
    """
    pricing = PRICING.get(model)
    if not pricing:
        return 0.0

    prompt_cost = (prompt_tokens / 1000) * pricing.prompt
    completion_cost = (completion_tokens / 1000) * pricing.completion
    return prompt_cost + completion_cost