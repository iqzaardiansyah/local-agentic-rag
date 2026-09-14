from langchain_core.tools import tool
from src.rag.hybrid_search import hybrid_search
from src.rag.reranker import rerank_documents
from src.rag.citations import set_citations


@tool
def search_local_documents(query: str) -> str:
    """
    Search the local knowledge base (RAG) for information.
    Uses Hybrid Search (BM25 sparse keyword matching + Chroma dense vector search with Reciprocal Rank Fusion)
    followed by local Cross-Encoder reranking for maximum retrieval accuracy.
    Use this tool when you need to answer questions about the user's specific documents.
    """
    try:
        # Step 1: Hybrid Retrieval (BM25 + Dense Chroma fused via Reciprocal Rank Fusion)
        candidates = hybrid_search(query, top_k=12)
        if not candidates:
            set_citations([])
            return "No relevant information found in the documents."

        # Step 2: High-precision Cross-Encoder reranking (ms-marco-MiniLM)
        reranked_pairs = rerank_documents(query, candidates, top_k=3)

        results = []
        citations = []
        for doc, score in reranked_pairs:
            src = doc.metadata.get('source', 'Unknown')
            preview = (doc.page_content or "")[:160].replace("\n", " ")
            results.append(f"Source: {src} (Relevance Score: {score:.3f})\nContent: {doc.page_content}")
            citations.append({
                "source": src,
                "score": float(score),
                "preview": preview,
            })
        set_citations(citations)

        return "\n\n---\n\n".join(results)
    except Exception as e:
        set_citations([])
        return f"Error accessing vector database: {str(e)}"
