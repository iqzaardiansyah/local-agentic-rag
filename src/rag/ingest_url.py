import os
import re
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from src.rag.vectorstore import (
    DATA_DIR,
    get_vectorstore,
    load_file_content,
)
from src.rag.hybrid_search import invalidate_bm25_index

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
)


def fetch_page_text(url: str, timeout: float = 15.0, max_chars: int = 50000) -> Tuple[str, str]:
    """
    Fetch a URL and return (clean_text, title).
    Raises on network/HTTP failure.
    """
    if not url or not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    headers = {"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml,*/*"}
    response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.extract()

    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[Content truncated]"
    return text, title


def _safe_filename(url: str, title: str = "") -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "page").replace(":", "_")
    path = (parsed.path or "").strip("/").replace("/", "_")
    base = f"{host}_{path}" if path else host
    base = re.sub(r"[^\w.\-]+", "_", base)[:80].strip("._") or "page"
    # Prefer title when short and useful
    if title:
        t = re.sub(r"[^\w\s\-]", "", title)[:40].strip().replace(" ", "_")
        if t:
            base = f"{t}__{base}"[:90]
    if not base.lower().endswith((".md", ".txt")):
        base += ".md"
    return base


def ingest_url_to_kb(url: str, max_chars: int = 50000) -> dict:
    """
    Scrape a webpage, save it under data/, chunk+embed into Chroma,
    extract GraphRAG triples, and invalidate the BM25 cache.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from src.rag.graph_rag import get_kg_engine

    text, title = fetch_page_text(url, max_chars=max_chars)
    if not text.strip():
        return {"success": False, "message": "Page had no extractable text.", "url": url}

    filename = _safe_filename(url, title)
    os.makedirs(DATA_DIR, exist_ok=True)
    dest = os.path.join(DATA_DIR, filename)
    header = f"<!-- source_url: {url} -->\n# {title or filename}\n\n"
    with open(dest, "w", encoding="utf-8") as f:
        f.write(header + text)

    docs = [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "path": f"data/{filename}",
                "abs_path": dest,
                "url": url,
                "title": title,
            },
        )
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vs = get_vectorstore()
    vs.add_documents(chunks)
    invalidate_bm25_index()

    triples = 0
    try:
        triples = get_kg_engine().extract_and_ingest_text(text, source_doc=filename)
    except Exception:
        triples = 0

    return {
        "success": True,
        "url": url,
        "title": title,
        "file": filename,
        "path": dest,
        "chunks": len(chunks),
        "triples": triples,
        "chars": len(text),
    }


@tool
def ingest_webpage(url: str) -> str:
    """
    Fetch a webpage, save it into the local knowledge base (data/ + Chroma + GraphRAG),
    and make it searchable via hybrid RAG. Use after web_search when a URL is highly relevant.
    Args:
        url: Full http(s) URL to ingest.
    """
    try:
        res = ingest_url_to_kb(url)
    except Exception as e:
        return f"Error ingesting URL {url}: {e}"

    if not res.get("success"):
        return f"Failed to ingest {url}: {res.get('message', 'unknown error')}"

    return (
        f"✅ Ingested `{url}`\n"
        f"- Title: {res.get('title') or '(none)'}\n"
        f"- Saved as: data/{res['file']}\n"
        f"- Chunks indexed: {res['chunks']}\n"
        f"- GraphRAG triples extracted: {res['triples']}\n"
        f"BM25 cache invalidated. The page is now searchable with `search_local_documents`."
    )
