"""
Centralized configuration — every tunable value lives here.

Change settings via .env file or environment variables.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "docs"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "750"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
