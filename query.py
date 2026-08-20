"""
DocChat - Day 2: Retrieval + LLM Answer Generation
----------------------------------------------------
Retrieves relevant chunks from the FAISS index (built in Day 1) and passes
them to a free LLM (via Groq) to generate a real, grounded answer -- with
source citations.

Setup:
    1. Get a free API key from https://console.groq.com
    2. Create a `.env` file in this folder with:
       GROQ_API_KEY=your_key_here
    3. Run: pip install langchain-groq
    4. Run: python query.py "your question here"
"""

import os
import sys
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv()  # reads GROQ_API_KEY from .env

INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"  # free on Groq, strong quality (replaces deprecated llama-3.3-70b)
TOP_K = 3  # how many chunks to retrieve per question

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY \
the context below. If the answer isn't in the context, say you don't know -- \
do not make things up.

Context:
{context}

Question: {question}

Answer clearly and concisely. After your answer, do not repeat the context."""


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )


def build_context(retrieved_docs):
    """Turn retrieved chunks into a labeled context block for the prompt,
    and a separate list of sources to show under the answer."""
    context_parts = []
    sources = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[{i}] {doc.page_content}")
        sources.append(f"  [{i}] {source}, page {page}")
    return "\n\n".join(context_parts), "\n".join(sources)


def answer_question(question: str):
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY not found. Make sure you created a .env file "
            "with GROQ_API_KEY=your_key_here"
        )

    vectorstore = load_vectorstore()
    retrieved_docs = vectorstore.similarity_search(question, k=TOP_K)

    if not retrieved_docs:
        print("No relevant content found in your documents.")
        return

    context, sources = build_context(retrieved_docs)

    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = llm.invoke(prompt)

    print(f"\nQuestion: {question}\n")
    print(f"Answer: {response.content}\n")
    print("Sources:")
    print(sources)


def main():
    if len(sys.argv) < 2:
        question = "What is this document about?"
        print(f"No question given, using default: '{question}'")
    else:
        question = " ".join(sys.argv[1:])

    answer_question(question)


if __name__ == "__main__":
    main()
