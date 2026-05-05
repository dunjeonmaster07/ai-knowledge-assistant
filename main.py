"""CLI interface — ask questions from the terminal."""

from src.chain import ask_question
from src.config import CHROMA_DIR
from src.ingest import run_ingestion

import os

if not CHROMA_DIR.exists():
    print("No vector store found. Running ingestion first...\n")
    run_ingestion()

print("=" * 60)
print("  AI Knowledge Assistant — Ask questions from your documents")
print("  Type 'quit' to exit")
print("=" * 60)

while True:
    question = input("\nYou: ").strip()
    if question.lower() in ["quit", "exit", "q"]:
        break
    if not question:
        continue

    answer, docs = ask_question(question)

    if answer:
        print(f"\nAssistant: {answer}")
        print(f"\nSources ({len(docs)}):")
        for doc in docs:
            source = os.path.basename(doc.metadata["source"])
            page = doc.metadata.get("page", "?")
            print(f"  - {source}, page {page}")
    else:
        print("\nAssistant: No relevant documents found. Try rephrasing your question.")

print("\nGoodbye!")
