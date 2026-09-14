"""
Free local FastAPI REST API for Local Agentic RAG.
Run:  uvicorn src.api.server:app --host 127.0.0.1 --port 8000
No cloud keys required — talks to your local/ngrok Ollama endpoint only.
"""

import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

app = FastAPI(
    title="Local Agentic RAG API",
    description="Free, local REST API for chat, RAG search, and document ingestion.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User question or instruction")
    session_id: Optional[str] = Field(None, description="Optional chat session id to persist")


class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


class IngestRequest(BaseModel):
    text: Optional[str] = None
    source_name: str = "api_upload.txt"


@app.get("/health")
def health() -> Dict[str, Any]:
    from src.agent.graph import LLM_BASE_URL, LLM_MODEL

    return {
        "status": "ok",
        "model": LLM_MODEL,
        "base_url": LLM_BASE_URL,
        "cost": 0,
        "requires_credit_card": False,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Run one full agent turn and return the final answer plus RAG citations."""
    from langchain_core.messages import HumanMessage
    from src.agent.graph import app as agent_app
    from src.rag.citations import clear_citations, get_citations
    from src.memory import chat_sessions

    session_id = req.session_id
    if session_id:
        chat_sessions.save_message(session_id, "user", req.message)

    clear_citations()
    inputs = {"messages": [HumanMessage(content=req.message)]}
    final = ""
    try:
        for event in agent_app.stream(inputs, stream_mode="values"):
            last = event["messages"][-1]
            content = getattr(last, "content", "")
            if content:
                final = content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}") from e

    citations = get_citations()
    if session_id and final:
        chat_sessions.save_message(session_id, "assistant", final)

    return ChatResponse(answer=final or "", citations=citations, session_id=session_id)


@app.post("/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    """2-stage hybrid RAG search with structured citations (no LLM)."""
    from src.rag.hybrid_search import hybrid_search
    from src.rag.reranker import rerank_documents

    candidates = hybrid_search(req.query, top_k=12)
    if not candidates:
        return {"query": req.query, "results": []}

    reranked = rerank_documents(req.query, candidates, top_k=req.top_k)
    results = []
    for doc, score in reranked:
        results.append(
            {
                "source": doc.metadata.get("source", "Unknown"),
                "score": float(score),
                "content": doc.page_content[:1500],
            }
        )
    return {"query": req.query, "results": results}


@app.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    """Ingest raw text into the local Chroma knowledge base."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from src.rag.vectorstore import get_vectorstore

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    docs = [Document(page_content=req.text, metadata={"source": req.source_name})]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return {"success": True, "source": req.source_name, "chunks": len(chunks)}


@app.get("/sessions")
def sessions() -> Dict[str, Any]:
    """List persisted chat sessions."""
    from src.memory import chat_sessions

    return {"sessions": chat_sessions.list_sessions()}


@app.get("/sessions/{session_id}/export")
def export_session(session_id: str) -> Dict[str, Any]:
    """Export a session as Markdown."""
    from src.memory import chat_sessions

    return {"session_id": session_id, "markdown": chat_sessions.export_session_markdown(session_id)}
