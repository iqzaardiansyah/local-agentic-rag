import re
import threading
from typing import List, Dict, Optional, Tuple

from langchain_core.documents import Document
from src.rag.vectorstore import get_vectorstore
from rank_bm25 import BM25Okapi

_bm25_instance: Optional[BM25Okapi] = None
_indexed_documents: List[Document] = []
# Generation counter / chroma size snapshot used to detect staleness.
_bm25_generation: int = 0
_bm25_built_from_count: int = 0
_lock = threading.RLock()


def tokenize(text: str) -> List[str]:
    """Tokenize string into lowercase alphanumeric words."""
    return re.findall(r'\w+', text.lower())


def build_bm25_index(documents: List[Document]) -> Optional[BM25Okapi]:
    """Builds a BM25 index from a list of LangChain Document objects."""
    global _bm25_instance, _indexed_documents, _bm25_generation, _bm25_built_from_count
    with _lock:
        _indexed_documents = list(documents)
        _bm25_generation += 1
        if not documents:
            _bm25_instance = None
            _bm25_built_from_count = 0
            return None
        tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
        _bm25_instance = BM25Okapi(tokenized_corpus)
        _bm25_built_from_count = len(documents)
        return _bm25_instance


def invalidate_bm25_index() -> None:
    """Force the next search to rebuild BM25 from Chroma (call after ingest/clear)."""
    global _bm25_instance, _indexed_documents, _bm25_built_from_count
    with _lock:
        _bm25_instance = None
        _indexed_documents = []
        _bm25_built_from_count = 0


def _chroma_document_count() -> int:
    try:
        vs = get_vectorstore()
        return int(vs._collection.count())
    except Exception:
        return -1


def _load_documents_from_chroma() -> List[Document]:
    from src.rag.vectorstore import is_indexable_file

    vs = get_vectorstore()
    col_data = vs._collection.get()
    documents: List[Document] = []
    if col_data and col_data.get("documents"):
        metas = col_data.get("metadatas") or [{}] * len(col_data["documents"])
        for text, meta in zip(col_data["documents"], metas):
            meta = meta or {}
            src = meta.get("source") or meta.get("path") or ""
            if src and not is_indexable_file(src):
                continue
            documents.append(Document(page_content=text, metadata=meta))
    return documents


def rebuild_bm25_index() -> Dict[str, object]:
    """
    Rebuild the in-process BM25 index from the current Chroma collection.
    Returns stats: {"documents": int, "generation": int, "ok": bool}.
    """
    try:
        documents = _load_documents_from_chroma()
        build_bm25_index(documents)
        return {
            "ok": True,
            "documents": len(documents),
            "generation": _bm25_generation,
        }
    except Exception as e:
        invalidate_bm25_index()
        return {"ok": False, "documents": 0, "generation": _bm25_generation, "error": str(e)}


def get_bm25_status() -> Dict[str, object]:
    """Report whether the BM25 cache looks fresh relative to Chroma size."""
    chroma_count = _chroma_document_count()
    stale = False
    reason = ""
    if _bm25_instance is None:
        stale = True
        reason = "not built"
    elif chroma_count >= 0 and chroma_count != _bm25_built_from_count:
        stale = True
        reason = f"chroma has {chroma_count} docs, BM25 built from {_bm25_built_from_count}"
    return {
        "built": _bm25_instance is not None,
        "documents": len(_indexed_documents),
        "generation": _bm25_generation,
        "chroma_count": chroma_count,
        "stale": stale,
        "reason": reason,
    }


def get_bm25_index() -> Tuple[Optional[BM25Okapi], List[Document]]:
    """
    Retrieves or builds the BM25 index from all documents currently in ChromaDB.
    Automatically rebuilds when the cache is empty or Chroma size no longer matches.
    """
    global _bm25_instance, _indexed_documents
    with _lock:
        if _bm25_instance is not None and _indexed_documents:
            chroma_count = _chroma_document_count()
            # Rebuild if Chroma size drifted (new ingest / clear) since last build.
            if chroma_count < 0 or chroma_count == _bm25_built_from_count:
                return _bm25_instance, _indexed_documents

        try:
            documents = _load_documents_from_chroma()
            if documents:
                build_bm25_index(documents)
            return _bm25_instance, _indexed_documents
        except Exception as e:
            print(f"Note on BM25 index fetch: {e}")
            return None, []


def bm25_search(query: str, top_k: int = 10) -> List[Document]:
    """Searches documents using BM25 sparse keyword matching."""
    bm25, documents = get_bm25_index()
    if bm25 is None or not documents:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return documents[:top_k]

    scores = bm25.get_scores(query_tokens)
    scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    # Filter only docs with positive BM25 relevance
    results = [doc for doc, score in scored_docs[:top_k] if score > 0]
    return results


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
    top_n: int = 10
) -> List[Document]:
    """
    Combines multiple ranked document lists using Reciprocal Rank Fusion (RRF).
    RRF Score = Sum(1 / (k + rank)) across all retrievers.
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            # Unique signature for each chunk
            doc_id = f"{doc.metadata.get('source', '')}::{doc.page_content.strip()}"
            doc_map[doc_id] = doc

            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort by descending RRF score
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    return [doc_map[doc_id] for doc_id in sorted_doc_ids[:top_n]]


def hybrid_search(
    query: str,
    top_k: int = 10,
    include_graph: bool = True,
) -> List[Document]:
    """
    Performs Hybrid Search combining:
      1. Chroma Dense Vectors
      2. BM25 Sparse Search
      3. Optional GraphRAG Documents (metadata.kind='graphrag')
    fused via Reciprocal Rank Fusion.

    Graph docs are real Documents so they flow through the same reranker and
    citation pipeline as file chunks — not a side channel.
    """
    # 1. Dense retrieval (ChromaDB)
    vs = get_vectorstore()
    try:
        dense_docs = vs.similarity_search(query, k=top_k)
    except Exception:
        dense_docs = []

    # 2. Sparse retrieval (BM25)
    sparse_docs = bm25_search(query, top_k=top_k)

    # 3. GraphRAG retrieval (structured triples → Documents)
    graph_docs: List[Document] = []
    if include_graph:
        try:
            from src.rag.graph_rag import graph_documents_for_query

            graph_docs = graph_documents_for_query(query, max_docs=3)
        except Exception:
            graph_docs = []

    try:
        from src.rag.vectorstore import is_indexable_file

        def _ok(doc: Document) -> bool:
            src = (doc.metadata or {}).get("source") or (doc.metadata or {}).get("path") or ""
            if (doc.metadata or {}).get("kind") == "graphrag":
                return True
            return (not src) or is_indexable_file(src)

        dense_docs = [d for d in dense_docs if _ok(d)]
        sparse_docs = [d for d in sparse_docs if _ok(d)]
    except Exception:
        pass

    ranked_lists = [docs for docs in (dense_docs, sparse_docs, graph_docs) if docs]
    if not ranked_lists:
        return []
    if len(ranked_lists) == 1:
        return ranked_lists[0][:top_k]

    # 4. Fuse using Reciprocal Rank Fusion
    fused_docs = reciprocal_rank_fusion(ranked_lists, k=60, top_n=top_k)
    return fused_docs
