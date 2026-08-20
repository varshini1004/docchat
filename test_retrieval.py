
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if len(sys.argv) < 2:
        query = "What is this document about?"
        print(f"No query given, using default: '{query}'")
    else:
        query = " ".join(sys.argv[1:])

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )

    results = vectorstore.similarity_search_with_score(query, k=3)

    print(f"\nQuery: {query}\n")
    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"--- Result {i} (score={score:.4f}, source={source}, page={page}) ---")
        print(doc.page_content[:300].strip(), "...\n")


if __name__ == "__main__":
    main()
