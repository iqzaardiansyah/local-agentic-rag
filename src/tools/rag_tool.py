from langchain_core.tools import tool
from src.rag.hybrid_search import hybrid_search
from src.rag.reranker import rerank_documents
from src.rag.citations import set_citations, enrich_citation


@tool
def search_local_documents(query: str) -> str:
    """
    Search the local knowledge base (RAG) for information.
    Uses Hybrid Search (BM25 sparse keyword matching + Chroma dense vector search + GraphRAG
    triples, fused with Reciprocal Rank Fusion) followed by local Cross-Encoder reranking.
    Use this tool when you need to answer questions about the user's specific documents
    or multi-hop entity relationships in the knowledge graph.
    """
    try:
        # Step 1: Hybrid Retrieval (BM25 + Dense Chroma + GraphRAG via RRF)
        candidates = hybrid_search(query, top_k=12, include_graph=True)
        if not candidates:
            set_citations([])
            return "No relevant information found in the documents."

        # Step 2: High-precision Cross-Encoder reranking (ms-marco-MiniLM)
        # Keep extra slots so graph docs can survive rerank even if scores differ.
        reranked_pairs = rerank_documents(query, candidates, top_k=5)

        results = []
        citations = []
        graph_sections = []
        doc_sections = []

        for doc, score in reranked_pairs:
            meta = doc.metadata or {}
            kind = meta.get("kind", "document")
            src = (
                meta.get("source")
                or meta.get("path")
                or meta.get("filename")
                or "Unknown"
            )
            preview = (doc.page_content or "")[:160].replace("\n", " ")

            citation = enrich_citation({
                "source": src,
                "score": float(score),
                "preview": preview,
                "chunk_preview": preview,
                "kind": kind,
                "path": meta.get("path"),
            })
            # Prefer path from metadata when enrich couldn't resolve (e.g. graph)
            if not citation.get("path") and meta.get("path"):
                citation["path"] = meta["path"]
                citation["rel_path"] = meta["path"]
                citation["exists"] = True
            if kind == "graphrag":
                citation["entities"] = meta.get("entities", "")
                graph_sections.append(
                    f"[GraphRAG] (Score: {score:.3f})\n{doc.page_content}"
                )
            else:
                doc_sections.append(
                    f"Source: {src} (Relevance Score: {score:.3f})\nContent: {doc.page_content}"
                )
            citations.append(citation)

        set_citations(citations)

        parts = []
        if doc_sections:
            parts.append("\n\n---\n\n".join(doc_sections))
        if graph_sections:
            header = "### Knowledge Graph Relations (GraphRAG)"
            parts.append(header + "\n\n" + "\n\n".join(graph_sections))
        if not parts:
            return "No relevant information found in the documents."
        return "\n\n".join(parts)
    except Exception as e:
        set_citations([])
        return f"Error accessing vector database: {str(e)}"
