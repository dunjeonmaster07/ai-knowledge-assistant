"""
Centralized configuration — every tunable value lives here.

Change settings via .env file or environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_SHARING_MODE") or os.path.exists("/mount/src")

REPO_DOCS_DIR = PROJECT_ROOT / "data" / "docs"

if IS_STREAMLIT_CLOUD:
    _WRITABLE_ROOT = Path("/tmp/rag_data")
    DATA_DIR = _WRITABLE_ROOT / "docs"
    CHROMA_DIR = _WRITABLE_ROOT / "chroma_db"
else:
    DATA_DIR = PROJECT_ROOT / "data" / "docs"
    CHROMA_DIR = PROJECT_ROOT / "chroma_db"

EVAL_DIR = PROJECT_ROOT / "data" / "eval"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "750"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
