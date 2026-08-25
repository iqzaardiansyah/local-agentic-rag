from langchain_core.tools import tool
from src.rag.vectorstore import get_vectorstore

@tool
def search_local_documents(query: str) -> str:
    """
    Search the local knowledge base (RAG) for information.
    Use this tool when you need to answer questions about the user's specific documents.
    """
    try:
        vectorstore = get_vectorstore()
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No relevant information found in the documents."
        
        results = [f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs]
        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"Error accessing vector database: {str(e)}"
