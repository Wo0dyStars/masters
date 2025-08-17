from langchain.prompts import PromptTemplate

DEFAULT_RAG_PROMPT = PromptTemplate.from_template(
    """Using the provided context, give a complete but concise answer that fully addresses the question.

Context:
{context}

Question: {question}

Instructions:
- Answer directly and comprehensively
- Use only information from the provided context
- If the context doesn't contain enough information, state this clearly
- Be precise and avoid speculation

Answer:"""
)