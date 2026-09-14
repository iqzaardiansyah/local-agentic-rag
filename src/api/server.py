"""
Free local FastAPI REST API for Local Agentic RAG.

Run:
  uvicorn src.api.server:app --host 127.0.0.1 --port 8000

No cloud keys required — talks to your local/ngrok Ollama endpoint only.

Endpoints:
  GET  /health
  POST /chat           — full turn (session-aware multi-turn) + citations
  POST /chat/stream    — Server-Sent Events stream (tokens, tools, citations)
  POST /search         — hybrid RAG only
  POST /ingest         — push text into Chroma
  GET  /sessions
  GET  /sessions/{id}/export
  GET  /sources        — resolve + preview a citation source file
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))

app = FastAPI(
    title="Local Agentic RAG API",
    description="Free, local REST API for chat, RAG search, and document ingestion.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User question or instruction")
    session_id: Optional[str] = Field(
        None,
        description="Optional chat session id. When set, prior turns in that session are used as multi-turn context and the exchange is persisted.",
    )
    include_cross_session: bool = Field(
        True,
        description="Inject a brief of other sessions + episodic memories into context.",
    )
    auto_memory: bool = Field(
        True,
        description="Heuristically capture preference phrases from this message into episodic memory.",
    )


class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: Optional[str] = None
    message_count: int = 0
    auto_memory: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


class IngestRequest(BaseModel):
    text: Optional[str] = None
    source_name: str = "api_upload.txt"


class IngestUrlRequest(BaseModel):
    url: str = Field(..., min_length=8, description="http(s) URL to scrape and index")


@app.get("/health")
def health() -> Dict[str, Any]:
    from src.agent.graph import LLM_BASE_URL, LLM_MODEL
    from src.agent.health import check_llm_endpoint
    from src.rag.hybrid_search import get_bm25_status

    llm = check_llm_endpoint(LLM_BASE_URL)
    bm25 = get_bm25_status()
    return {
        "status": "ok" if llm["ok"] else "degraded",
        "model": LLM_MODEL,
        "base_url": LLM_BASE_URL,
        "llm": llm,
        "bm25": bm25,
        "cost": 0,
        "requires_credit_card": False,
        "streaming": "/chat/stream",
    }


@app.post("/reindex/bm25")
def reindex_bm25() -> Dict[str, Any]:
    """Force a BM25 rebuild from the current Chroma collection (free, local)."""
    from src.rag.hybrid_search import rebuild_bm25_index, get_bm25_status

    result = rebuild_bm25_index()
    return {"result": result, "status": get_bm25_status()}


@app.get("/cache/websearch")
def websearch_cache_stats() -> Dict[str, Any]:
    """Local web_search cache stats."""
    from src.tools.web_cache import cache_stats

    return cache_stats()


@app.delete("/cache/websearch")
def websearch_cache_clear() -> Dict[str, Any]:
    """Clear the local web_search cache."""
    from src.tools.web_cache import clear_cache

    cleared = clear_cache()
    return {"cleared": cleared}


@app.post("/kb/export")
def export_kb(include_bodies: bool = True) -> Dict[str, Any]:
    """
    Export the knowledge base (data/ + GraphRAG + index stats) to Markdown.
    Default output: workspace/kb_export.md
    """
    from src.rag.kb_export import export_knowledge_base_markdown

    try:
        return export_knowledge_base_markdown(include_file_bodies=include_bodies)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}") from e


@app.get("/kb/export/content")
def export_kb_content(include_bodies: bool = True) -> Dict[str, Any]:
    """Return the KB export Markdown as JSON without writing a file (rebuilds in memory)."""
    from src.rag import kb_export
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w+", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = tmp.name
    try:
        info = kb_export.export_knowledge_base_markdown(
            output_path=tmp_path, include_file_bodies=include_bodies
        )
        with open(tmp_path, "r", encoding="utf-8") as f:
            markdown = f.read()
        return {"info": info, "markdown": markdown}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _sse(data: Dict[str, Any], event: Optional[str] = None) -> str:
    payload = json.dumps(data, default=str)
    if event:
        return f"event: {event}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


def _stream_chat_events(req: ChatRequest) -> Iterator[str]:
    """Yield SSE frames for one agent turn."""
    from src.memory import chat_sessions
    from src.memory.session_context import build_agent_messages
    from src.memory.auto_memory import maybe_autocapture_preferences
    from src.agent.runner import run_agent_stream

    session_id = req.session_id
    if session_id:
        chat_sessions.ensure_session(session_id, title="API Chat")
        last = chat_sessions.load_messages(session_id, limit=2)
        last_user = next((m["content"] for m in reversed(last) if m["role"] == "user"), None)
        if last_user != req.message.strip():
            chat_sessions.save_message(session_id, "user", req.message)

    auto_saved = maybe_autocapture_preferences(
        req.message, session_id=session_id, enabled=req.auto_memory
    )
    if auto_saved:
        yield _sse({"type": "memory_capture", "facts": auto_saved}, event="memory_capture")

    try:
        messages = build_agent_messages(
            req.message,
            session_id=session_id,
            include_cross_session=req.include_cross_session,
        )
    except Exception as e:
        yield _sse({"type": "error", "error": f"Failed to build context: {e}"})
        return

    final_answer = ""
    steps: List[Dict[str, Any]] = []
    citations: List[Dict[str, Any]] = []
    message_count = len(messages)

    for event in run_agent_stream(
        req.message,
        session_id=session_id,
        history=messages,
    ):
        etype = event.get("type")
        if etype == "meta":
            message_count = event.get("message_count", message_count)
            yield _sse(
                {
                    "type": "meta",
                    "session_id": session_id,
                    "model": event.get("model"),
                    "message_count": message_count,
                },
                event="meta",
            )
        elif etype == "token":
            yield _sse({"type": "token", "token": event.get("token", "")}, event="token")
        elif etype == "tool_call":
            yield _sse(
                {
                    "type": "tool_call",
                    "tool": event.get("tool"),
                    "args": event.get("args"),
                },
                event="tool_call",
            )
        elif etype == "tool_result":
            yield _sse(
                {
                    "type": "tool_result",
                    "tool": event.get("tool"),
                    "output": event.get("output"),
                },
                event="tool_result",
            )
        elif etype == "grade":
            yield _sse(
                {
                    "type": "grade",
                    "label": event.get("label"),
                    "detail": event.get("detail"),
                },
                event="grade",
            )
        elif etype in ("subagent_start", "subagent_done", "subagents_all_done"):
            yield _sse(event, event=etype)
        elif etype == "done":
            final_answer = event.get("answer") or ""
            steps = event.get("steps") or []
            citations = event.get("citations") or []
            if session_id and final_answer:
                chat_sessions.save_message(
                    session_id,
                    "assistant",
                    final_answer,
                    steps=steps,
                    citations=citations,
                )
            yield _sse(
                {
                    "type": "done",
                    "answer": final_answer,
                    "citations": citations,
                    "steps": steps,
                    "session_id": session_id,
                    "message_count": message_count,
                    "auto_memory": auto_saved,
                    "metrics": event.get("metrics") or {},
                },
                event="done",
            )
        elif etype == "error":
            yield _sse({"type": "error", "error": event.get("error")}, event="error")
            return


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Run one full agent turn and return the final answer plus RAG citations.

    When `session_id` is set:
      - prior messages in that session are sent as multi-turn context
      - optional cross-session brief + episodic recall are injected
      - the user/assistant exchange is persisted
    """
    from src.memory import chat_sessions
    from src.memory.session_context import build_agent_messages
    from src.memory.auto_memory import maybe_autocapture_preferences
    from src.agent.runner import run_agent_stream

    session_id = req.session_id
    if session_id:
        chat_sessions.ensure_session(session_id, title="API Chat")
        last = chat_sessions.load_messages(session_id, limit=2)
        last_user = next((m["content"] for m in reversed(last) if m["role"] == "user"), None)
        if last_user != req.message.strip():
            chat_sessions.save_message(session_id, "user", req.message)

    auto_saved = maybe_autocapture_preferences(
        req.message, session_id=session_id, enabled=req.auto_memory
    )

    try:
        messages = build_agent_messages(
            req.message,
            session_id=session_id,
            include_cross_session=req.include_cross_session,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    result = run_agent_stream(
        req.message,
        session_id=session_id,
        history=messages,
    )
    # Materialize the generator
    final = {"answer": "", "steps": [], "citations": []}
    error = None
    message_count = len(messages)
    for event in result:
        if event["type"] == "meta":
            message_count = event.get("message_count", message_count)
        elif event["type"] == "done":
            final = event
        elif event["type"] == "error":
            error = event.get("error")
    if error:
        raise HTTPException(status_code=500, detail=f"Agent error: {error}")

    if session_id and final.get("answer"):
        chat_sessions.save_message(
            session_id,
            "assistant",
            final["answer"],
            steps=final.get("steps") or [],
            citations=final.get("citations") or [],
        )

    return ChatResponse(
        answer=final.get("answer") or "",
        citations=final.get("citations") or [],
        steps=final.get("steps") or [],
        session_id=session_id,
        message_count=message_count,
        auto_memory=auto_saved,
        metrics=final.get("metrics") or {},
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """
    Server-Sent Events stream of one agent turn.

    Events (also set as SSE `event:` names):
      meta, token, tool_call, tool_result, grade,
      subagent_start, subagent_done, subagents_all_done,
      memory_capture, done, error

    Client example:
      const es = new EventSourcePolyfill('/chat/stream', { method: 'POST', ... })
      // or use fetch + ReadableStream and parse `data:` frames
    """
    return StreamingResponse(
        _stream_chat_events(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/search")
def search(req: SearchRequest) -> Dict[str, Any]:
    """2-stage hybrid RAG search with structured, path-resolved citations (no LLM)."""
    from src.rag.hybrid_search import hybrid_search
    from src.rag.reranker import rerank_documents
    from src.rag.citations import enrich_citation

    candidates = hybrid_search(req.query, top_k=12)
    if not candidates:
        return {"query": req.query, "results": []}

    reranked = rerank_documents(req.query, candidates, top_k=req.top_k)
    results = []
    for doc, score in reranked:
        meta = doc.metadata or {}
        item = enrich_citation(
            {
                "source": meta.get("source") or meta.get("path") or "Unknown",
                "score": float(score),
                "content": doc.page_content[:1500],
                "preview": (doc.page_content or "")[:160].replace("\n", " "),
            }
        )
        results.append(item)
    return {"query": req.query, "results": results}


@app.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    """Ingest raw text into the local Chroma knowledge base."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from src.rag.vectorstore import get_vectorstore, DATA_DIR

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    # Persist a copy under data/ so citations can open the source file.
    os.makedirs(DATA_DIR, exist_ok=True)
    safe_name = os.path.basename(req.source_name) or "api_upload.txt"
    dest = os.path.join(DATA_DIR, safe_name)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(req.text)

    docs = [
        Document(
            page_content=req.text,
            metadata={
                "source": safe_name,
                "path": f"data/{safe_name}",
                "abs_path": dest,
            },
        )
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    try:
        from src.rag.hybrid_search import invalidate_bm25_index

        invalidate_bm25_index()
    except Exception:
        pass
    return {"success": True, "source": safe_name, "path": dest, "chunks": len(chunks)}


@app.post("/ingest/url")
def ingest_url(req: IngestUrlRequest) -> Dict[str, Any]:
    """Scrape a webpage and index it into Chroma + GraphRAG (free, local)."""
    from src.rag.ingest_url import ingest_url_to_kb

    try:
        result = ingest_url_to_kb(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingest failed: {e}") from e
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message", "Ingest failed"))
    return result


@app.get("/sessions")
def sessions() -> Dict[str, Any]:
    """List persisted chat sessions."""
    from src.memory import chat_sessions

    return {"sessions": chat_sessions.list_sessions()}


@app.get("/sessions/search")
def search_sessions(
    q: str = Query(..., min_length=1, description="Substring to find in message content"),
    limit: int = Query(20, ge=1, le=100),
    session_id: Optional[str] = Query(None, description="Optional restrict to one session"),
) -> Dict[str, Any]:
    """Search message text across all chat sessions."""
    from src.memory import chat_sessions

    hits = chat_sessions.search_messages(q, limit=limit, session_id=session_id)
    return {"query": q, "count": len(hits), "hits": hits}


@app.get("/sessions/{session_id}/export")
def export_session(session_id: str) -> Dict[str, Any]:
    """Export a session as Markdown (includes citations)."""
    from src.memory import chat_sessions

    return {
        "session_id": session_id,
        "markdown": chat_sessions.export_session_markdown(session_id),
    }


@app.get("/sources")
def get_source(
    name: str = Query(..., description="Source filename or relative path from a citation"),
) -> Dict[str, Any]:
    """Resolve a citation source to disk and return path + optional text preview."""
    from src.rag.citations import enrich_citation, read_source_content

    item = enrich_citation({"source": name})
    if not item.get("exists"):
        raise HTTPException(status_code=404, detail=f"Source not found: {name}")
    body = ""
    if item.get("readable") and item.get("path"):
        body = read_source_content(item["path"], max_chars=50000)
    return {"citation": item, "content": body}
