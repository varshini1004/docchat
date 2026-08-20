
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
INDEX_DIR = "faiss_index"

# Chunk size/overlap are tunable — this is exactly the kind of thing to
# experiment with on Day 3 and write about in your README.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free


def load_documents(data_dir: str):
    """Load every PDF in data_dir and tag each page with its source filename."""
    documents = []
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in '{data_dir}/'. Add at least one PDF and re-run."
        )

    print(f"Found {len(pdf_files)} PDF(s): {pdf_files}")

    for filename in pdf_files:
        path = os.path.join(data_dir, filename)
        loader = PyPDFLoader(path)
        pages = loader.load()  # one Document per page, with metadata['page']
        for page in pages:
            page.metadata["source"] = filename  # used later for citations
        documents.extend(pages)
        print(f"  Loaded {filename}: {len(pages)} pages")

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_and_save_index(chunks, index_dir: str):
    print(f"Loading embedding model: {EMBEDDING_MODEL} (first run downloads it)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding chunks and building FAISS index (this may take a minute)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(index_dir)
    print(f"Saved FAISS index to '{index_dir}/'")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)
    build_and_save_index(chunks, INDEX_DIR)
    print("\nDay 1 complete. Your documents are now searchable.")
    print("Next (Day 2): write query.py to retrieve chunks + call an LLM.")


if __name__ == "__main__":
    main()
