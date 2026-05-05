"""
Retriever factory — loads the persisted ChromaDB and returns a retriever.

The embedding model MUST match what was used during ingestion.
Both files import from config.py to ensure consistency.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, EMBEDDING_MODEL, RETRIEVER_K, SCORE_THRESHOLD


def get_retriever(
    k: int = RETRIEVER_K,
    score_threshold: float = SCORE_THRESHOLD,
):
    """Load ChromaDB from disk and return a LangChain retriever."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": k},
    )
    return retriever
