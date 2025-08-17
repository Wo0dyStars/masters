from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class VectorStoreException(Exception):
    pass

def load_vector_store(path: str = "vector_store/stores/md_medium_recursive"):
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store, embeddings
    except Exception as e:
        raise VectorStoreException(f"Failed to load vector store: {str(e)}")