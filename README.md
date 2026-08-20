# DocChat — RAG Document Q&A

A retrieval-augmented generation (RAG) chatbot that lets you ask questions
about your own PDF documents. Currently implements the ingestion and
retrieval pipeline: turning PDFs into a searchable vector index.

## Setup

```
git clone https://github.com/varshini1004/docchat.git
cd docchat
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Put 1–3 PDF files into the `data/` folder (textbooks, notes, papers, etc.).
2. Build the index:

```
python ingest.py
```

3. Test retrieval (no LLM yet, just confirms search works):

```
python test_retrieval.py "your question about the document"
```

4. Start the server:

```
uvicorn app:app --reload
```

## How it works

- **PyPDFLoader** reads each PDF page as a separate document, tagged with its
  source filename and page number (this powers citations later).
- **RecursiveCharacterTextSplitter** breaks pages into ~800-character chunks
  with 150-character overlap, so context isn't lost at chunk boundaries.
- **HuggingFaceEmbeddings** (`all-MiniLM-L6-v2`) turns each chunk into a
  384-dimension vector — runs locally, free, no API key required.
- **FAISS** stores those vectors and enables fast similarity search.

## Roadmap

- `query.py`: retrieve top-k chunks for a question and pass them to an LLM
  (Groq/OpenAI) to generate a grounded answer.
- Add source citations to the LLM's answer.
- Wrap in FastAPI, then build out the frontend.

## Notes

Chunk size and overlap (`CHUNK_SIZE` / `CHUNK_OVERLAP` in `ingest.py`) are
tunable — worth experimenting with and documenting what works best for your
data.
