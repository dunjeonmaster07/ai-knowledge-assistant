"""
Retriever factory.

Local:           loads persisted ChromaDB from disk
Streamlit Cloud: receives in-memory vector store directly
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, EMBEDDING_MODEL, RETRIEVER_K, IS_STREAMLIT_CLOUD


def get_retriever(vector_store: Chroma | None = None, k: int = RETRIEVER_K):
    """
    Return a LangChain retriever.

    If vector_store is provided (Streamlit Cloud), use it directly.
    Otherwise, load from persisted directory (local).
    """
    if vector_store is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )

    return vector_store.as_retriever(search_kwargs={"k": k})
