import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Path to the data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")

def get_embeddings():
    """Returns local HuggingFace embeddings to keep it free."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    """Returns the Chroma vectorstore instance."""
    embeddings = get_embeddings()
    # Create directory if it doesn't exist
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    return Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)

def ingest_documents():
    """Loads, chunks, and embeds documents from the data directory."""
    print(f"Loading documents from {DATA_DIR}...")
    
    # Load TXT files
    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    txt_docs = txt_loader.load()
    
    # Load PDF files
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    try:
        pdf_docs = pdf_loader.load()
    except Exception as e:
        print(f"No PDFs found or error loading PDFs: {e}")
        pdf_docs = []
        
    documents = txt_docs + pdf_docs
    
    if not documents:
        print("No documents found to ingest. Add some .txt or .pdf files to the 'data' directory.")
        return
        
    print(f"Loaded {len(documents)} documents. Chunking...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks. Embedding and saving to ChromaDB...")
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_documents()
