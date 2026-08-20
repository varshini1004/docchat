# DocChat — Day 1: Ingestion Pipeline

RAG-based document Q&A chatbot. This is the first slice: turning PDFs into a
searchable vector index.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

1. Put 1–3 PDF files into the `data/` folder (textbooks, notes, papers — pick
   a theme you can talk about confidently in an interview).
2. Build the index:
   ```bash
   python ingest.py
   ```
3. Test retrieval (no LLM yet, just confirms search works):
   ```bash
   python test_retrieval.py "your question about the document"
   ```

## What's happening under the hood

- **PyPDFLoader** reads each PDF page as a separate document, tagged with its
  source filename and page number (this is what powers citations later).
- **RecursiveCharacterTextSplitter** breaks pages into ~800-character chunks
  with 150-character overlap, so context isn't lost at chunk boundaries.
- **HuggingFaceEmbeddings** (`all-MiniLM-L6-v2`) turns each chunk into a
  384-dimension vector — runs locally, free, no API key.
- **FAISS** stores those vectors and lets you do fast similarity search.

## Next steps (Day 2+)

- `query.py`: retrieve top-k chunks for a question, pass them to an LLM
  (Groq/OpenAI) to generate a grounded answer.
- Add source citations to the LLM's answer.
- Wrap in FastAPI, then build the frontend.

## Notes for your README/resume later

Once you experiment with `CHUNK_SIZE`/`CHUNK_OVERLAP` in `ingest.py`, note
what values worked best and why — that's a concrete detail worth mentioning
in interviews ("I tuned chunk size from 500 to 800 chars because...").
