from langchain_core.tools import tool
from src.rag.vectorstore import get_vectorstore
from src.rag.reranker import rerank_documents

@tool
def search_local_documents(query: str) -> str:
    """
    Search the local knowledge base (RAG) for information.
    Uses initial vector similarity search followed by local Cross-Encoder reranking for precision.
    Use this tool when you need to answer questions about the user's specific documents.
    """
    try:
        vectorstore = get_vectorstore()
        # Step 1: Broad candidate retrieval from ChromaDB
        candidates = vectorstore.similarity_search(query, k=10)
        if not candidates:
            return "No relevant information found in the documents."
        
        # Step 2: High-precision Cross-Encoder reranking (ms-marco-MiniLM)
        reranked_pairs = rerank_documents(query, candidates, top_k=3)
        
        results = []
        for doc, score in reranked_pairs:
            src = doc.metadata.get('source', 'Unknown')
            results.append(f"Source: {src} (Relevance Score: {score:.3f})\nContent: {doc.page_content}")
            
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Error accessing vector database: {str(e)}"

