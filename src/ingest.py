"""
Document ingestion pipeline: PDF → chunks → embeddings → ChromaDB.

Local:           persists to disk (chroma_db/)
Streamlit Cloud: in-memory only (no SQLite, no disk writes)

Usage:
    python -m src.ingest
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, IS_STREAMLIT_CLOUD


def load_pdfs(data_dir=None) -> list:
    """Load every PDF in data_dir and return a flat list of LangChain Documents."""
    data_dir = data_dir or DATA_DIR
    documents = []
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {data_dir}. "
            f"Add your PDF files to the data/docs/ folder and re-run."
        )

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        documents.extend(docs)
        print(f"  Loaded {pdf_path.name}: {len(docs)} pages")

    print(f"Total: {len(documents)} pages from {len(pdf_files)} PDFs\n")
    return documents


def chunk_documents(documents: list) -> list:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")
    return chunks


def create_vector_store(chunks: list, persist: bool = True) -> Chroma:
    """Embed chunks into ChromaDB. Persists to disk locally, in-memory on cloud."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    ids = [f"{doc.metadata['source']}_{i}" for i, doc in enumerate(chunks)]

    kwargs = {
        "documents": chunks,
        "embedding": embeddings,
        "ids": ids,
        "collection_metadata": {"hnsw:space": "cosine"},
    }

    if persist and not IS_STREAMLIT_CLOUD:
        kwargs["persist_directory"] = str(CHROMA_DIR)

    vector_store = Chroma.from_documents(**kwargs)
    location = "in-memory" if IS_STREAMLIT_CLOUD else str(CHROMA_DIR)
    print(f"Stored {len(chunks)} chunks in ChromaDB ({location})\n")
    return vector_store


def run_ingestion(data_dir=None) -> Chroma:
    """Full pipeline: load → chunk → embed → store."""
    print("=" * 60)
    print("INGESTION PIPELINE")
    print("=" * 60 + "\n")

    documents = load_pdfs(data_dir)
    chunks = chunk_documents(documents)
    vector_store = create_vector_store(chunks)

    print("Ingestion complete.")
    return vector_store


if __name__ == "__main__":
    run_ingestion()
