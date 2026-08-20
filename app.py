"""
DocChat - Day 4: FastAPI Backend
-----------------------------------
Wraps the ingestion + retrieval + LLM pipeline in a web API so a frontend
(Day 5) can talk to it over HTTP.

Endpoints:
    GET  /health         - simple check that the server is alive
    POST /chat            - ask a question, get an answer + sources
    POST /upload           - upload a new PDF and rebuild the index

Run:
    pip install fastapi uvicorn python-multipart
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for an interactive test UI (FastAPI
gives you this automatically -- no frontend needed to try it out).
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from query import load_vectorstore, build_context, PROMPT_TEMPLATE, LLM_MODEL, TOP_K
from ingest import load_documents, split_documents, build_and_save_index, DATA_DIR, INDEX_DIR
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI(title="DocChat API")

# Allows a frontend running on a different port (e.g. React on :3000) to
# call this API during local development. Tighten this before deploying
# publicly (Day 7).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not os.path.exists(INDEX_DIR):
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a PDF via /upload first.",
        )

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured on server.")

    vectorstore = load_vectorstore()
    retrieved_docs = vectorstore.similarity_search(request.question, k=TOP_K)

    if not retrieved_docs:
        return ChatResponse(answer="No relevant content found in your documents.", sources=[])

    context, sources_str = build_context(retrieved_docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=request.question)

    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    response = llm.invoke(prompt)

    sources_list = [line.strip() for line in sources_str.split("\n") if line.strip()]

    return ChatResponse(answer=response.content, sources=sources_list)


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Rebuild the index to include the new file. For a bigger project you'd
    # do this incrementally instead of from scratch -- worth mentioning as
    # a "future improvement" in your README.
    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)
    build_and_save_index(chunks, INDEX_DIR)

    return {"message": f"Uploaded and indexed '{file.filename}' successfully."}
