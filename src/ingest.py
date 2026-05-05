"""
Document ingestion pipeline: PDF → chunks → embeddings → ChromaDB.

Usage:
    python -m src.ingest
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdfs() -> list:
    """Load every PDF in DATA_DIR and return a flat list of LangChain Documents."""
    documents = []
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {DATA_DIR}. "
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


def create_vector_store(chunks: list) -> Chroma:
    """Embed chunks and persist to ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    ids = [f"{doc.metadata['source']}_{i}" for i, doc in enumerate(chunks)]

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        ids=ids,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {CHROMA_DIR}\n")
    return vector_store


def run_ingestion() -> Chroma:
    """Full pipeline: load → chunk → embed → store."""
    print("=" * 60)
    print("INGESTION PIPELINE")
    print("=" * 60 + "\n")

    documents = load_pdfs()
    chunks = chunk_documents(documents)
    vector_store = create_vector_store(chunks)

    print("Ingestion complete.")
    return vector_store


if __name__ == "__main__":
    run_ingestion()
