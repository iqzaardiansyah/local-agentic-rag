import re
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from src.rag.vectorstore import get_vectorstore
from rank_bm25 import BM25Okapi

_bm25_instance = None
_indexed_documents: List[Document] = []

def tokenize(text: str) -> List[str]:
    """Tokenize string into lowercase alphanumeric words."""
    return re.findall(r'\w+', text.lower())

def build_bm25_index(documents: List[Document]) -> BM25Okapi:
    """Builds a BM25 index from a list of LangChain Document objects."""
    global _bm25_instance, _indexed_documents
    _indexed_documents = documents
    if not documents:
        _bm25_instance = None
        return None
    tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
    _bm25_instance = BM25Okapi(tokenized_corpus)
    return _bm25_instance

def get_bm25_index() -> Tuple[BM25Okapi, List[Document]]:
    """Retrieves or builds the BM25 index from all documents currently in ChromaDB."""
    global _bm25_instance, _indexed_documents
    if _bm25_instance is not None and _indexed_documents:
        return _bm25_instance, _indexed_documents
        
    # Fetch all stored documents from ChromaDB collection
    try:
        vs = get_vectorstore()
        col_data = vs._collection.get()
        documents = []
        if col_data and col_data.get("documents"):
            for text, meta in zip(col_data["documents"], col_data["metadatas"] or [{}] * len(col_data["documents"])):
                documents.append(Document(page_content=text, metadata=meta or {}))
                
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

def hybrid_search(query: str, top_k: int = 10) -> List[Document]:
    """
    Performs Hybrid Search combining Chroma Dense Vectors and BM25 Sparse Search via RRF.
    """
    # 1. Dense retrieval (ChromaDB)
    vs = get_vectorstore()
    try:
        dense_docs = vs.similarity_search(query, k=top_k)
    except Exception:
        dense_docs = []
        
    # 2. Sparse retrieval (BM25)
    sparse_docs = bm25_search(query, top_k=top_k)
    
    # 3. If only one retriever has results, return it
    if not sparse_docs and dense_docs:
        return dense_docs
    if not dense_docs and sparse_docs:
        return sparse_docs
    if not dense_docs and not sparse_docs:
        return []
        
    # 4. Fuse using Reciprocal Rank Fusion
    fused_docs = reciprocal_rank_fusion([dense_docs, sparse_docs], k=60, top_n=top_k)
    return fused_docs
