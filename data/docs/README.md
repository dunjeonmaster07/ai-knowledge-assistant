# Sample Documents

Place your PDF files in this folder. The system will automatically ingest them on first run.

## Getting started

1. Add one or more `.pdf` files to this folder
2. Run `python -m src.ingest` to build the vector store
3. Run `streamlit run app.py` to start asking questions

## Notes

- The system currently supports **PDF files only**
- Larger documents take longer to ingest (embedding step)
- Re-run ingestion after adding new documents (delete `chroma_db/` first to rebuild)
- A sample PDF is included so you can test immediately
