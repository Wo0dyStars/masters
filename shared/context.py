from typing import List

def truncate_context(contexts: List[str], max_length: int) -> List[str]:
    """
    Truncate multiple context strings to ensure their combined length stays within a maximum limit.

    If the total length exceeds the limit, each context is proportionally reduced
    with a buffer to prevent overflow.

    Args:
        contexts (List[str]): List of context strings.
        max_length (int): Maximum total length allowed.

    Returns:
        List[str]: Truncated context strings.
    """
    total_length = sum(len(ctx) for ctx in contexts)
    if total_length <= max_length:
        return contexts

    ratio = max_length / total_length
    truncated = [
        ctx[: int(len(ctx) * ratio * 0.9)] + "..." if len(ctx) > int(len(ctx) * ratio * 0.9) else ctx
        for ctx in contexts
    ]
    
    return truncated
