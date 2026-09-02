import os
import shutil
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Path to the data directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
CHROMA_DB_DIR = os.path.join(ROOT_DIR, "chroma_db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

_embeddings = None

def get_embeddings():
    """Returns singleton local HuggingFace embeddings to keep it free and fast."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def get_vectorstore():
    """Returns the Chroma vectorstore instance."""
    embeddings = get_embeddings()
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

def load_file_content(file_path: str) -> List[Document]:
    """Extract Document chunks from various file types (.pdf, .txt, .md, .csv, .json)."""
    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)
    
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif ext == ".csv":
            loader = CSVLoader(file_path)
            return loader.load()
        elif ext in [".txt", ".md", ".json", ".py", ".js", ".html"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": filename})]
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return [Document(page_content=text, metadata={"source": filename})]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def save_and_ingest_uploaded_files(uploaded_files) -> Dict[str, Any]:
    """Save uploaded Streamlit files to data/ and index them into ChromaDB."""
    os.makedirs(DATA_DIR, exist_ok=True)
    saved_files = []
    all_documents = []
    
    for uf in uploaded_files:
        save_path = os.path.join(DATA_DIR, uf.name)
        with open(save_path, "wb") as f:
            f.write(uf.getbuffer())
        saved_files.append(uf.name)
        
        docs = load_file_content(save_path)
        all_documents.extend(docs)
        
    if not all_documents:
        return {"success": False, "message": "No document content extracted.", "chunks": 0}
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)
    
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    
    return {
        "success": True,
        "saved_files": saved_files,
        "chunks": len(chunks),
        "total_documents": len(all_documents)
    }

def get_knowledge_base_stats() -> Dict[str, Any]:
    """Return live metrics and file listing for the knowledge base."""
    files = []
    if os.path.exists(DATA_DIR):
        files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
        
    total_chunks = 0
    try:
        if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
            vectorstore = get_vectorstore()
            total_chunks = vectorstore._collection.count()
    except Exception:
        total_chunks = 0
        
    return {
        "files": files,
        "total_files": len(files),
        "total_chunks": total_chunks
    }

def clear_vectorstore():
    """Wipe ChromaDB and reset collection cleanly via Chroma API and filesystem."""
    try:
        vs = get_vectorstore()
        all_ids = vs._collection.get().get('ids', [])
        if all_ids:
            vs._collection.delete(ids=all_ids)
    except Exception as e:
        print(f"Note on Chroma collection reset: {e}")
        
    if os.path.exists(CHROMA_DB_DIR):
        try:
            shutil.rmtree(CHROMA_DB_DIR, ignore_errors=True)
            os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        except Exception:
            pass


def reindex_all_data() -> int:
    """Re-reads everything in data/ and builds a fresh ChromaDB index."""
    clear_vectorstore()
    if not os.path.exists(DATA_DIR):
        return 0
    all_docs = []
    for f in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, f)
        if os.path.isfile(fpath):
            all_docs.extend(load_file_content(fpath))
            
    if not all_docs:
        return 0
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return len(chunks)

def ingest_documents():
    """CLI helper to load, chunk, and embed documents from data directory."""
    print(f"Loading documents from {DATA_DIR}...")
    chunk_count = reindex_all_data()
    print(f"Ingestion complete! Total chunks in ChromaDB: {chunk_count}")

if __name__ == "__main__":
    ingest_documents()

