import os
from typing import List, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

_reranker_instance = None
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def get_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoder:
    """Singleton loader for local Cross-Encoder model."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoder(model_name)
    return _reranker_instance

def rerank_documents(query: str, documents: List[Document], top_k: int = 3) -> List[Tuple[Document, float]]:
    """
    Reranks candidate documents against a query using a local Cross-Encoder.
    Returns a list of (Document, score) tuples sorted by highest relevance.
    """
    if not documents:
        return []
        
    if len(documents) <= 1:
        return [(documents[0], 1.0)]
        
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    
    # Predict cross-encoder scores
    scores = reranker.predict(pairs)
    
    # Pair documents with their cross-encoder score
    doc_score_pairs = []
    for doc, score in zip(documents, scores):
        doc.metadata["rerank_score"] = float(score)
        doc_score_pairs.append((doc, float(score)))
        
    # Sort in descending order of score
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return doc_score_pairs[:top_k]
