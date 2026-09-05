import os
import json

import uuid
import datetime
from typing import List, Dict, Any, Optional
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


# Memory directory
MEMORY_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma_memory")
os.makedirs(MEMORY_DB_DIR, exist_ok=True)

_memory_embeddings = None
_memory_vectorstore = None

def get_memory_vectorstore():
    """Singleton loader for Episodic Memory ChromaDB vector store."""
    global _memory_embeddings, _memory_vectorstore
    if _memory_vectorstore is None:
        if _memory_embeddings is None:
            _memory_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _memory_vectorstore = Chroma(
            persist_directory=MEMORY_DB_DIR,
            embedding_function=_memory_embeddings,
            collection_name="episodic_chat_memory"
        )
    return _memory_vectorstore

def save_memory(fact: str, category: str = "general", metadata: Optional[Dict[str, Any]] = None) -> str:
    """Save an episodic memory fact into the vector database."""
    vs = get_memory_vectorstore()
    mem_id = str(uuid.uuid4())
    ts = datetime.datetime.now().isoformat()
    meta = {
        "id": mem_id,
        "category": category,
        "timestamp": ts,
        **(metadata or {})
    }
    vs.add_texts(texts=[fact], metadatas=[meta], ids=[mem_id])
    return mem_id

def recall_memories(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve top relevant memories from the episodic vector store."""
    vs = get_memory_vectorstore()
    try:
        results = vs.similarity_search_with_score(query, k=top_k)
        memories = []
        for doc, score in results:
            memories.append({
                "fact": doc.page_content,
                "category": doc.metadata.get("category", "general"),
                "timestamp": doc.metadata.get("timestamp", ""),
                "score": float(score)
            })
        return memories
    except Exception:
        return []

def list_all_memories() -> List[Dict[str, Any]]:
    """Retrieve all stored episodic memories."""
    vs = get_memory_vectorstore()
    try:
        data = vs._collection.get()
        memories = []
        if data and "documents" in data and data["documents"]:
            for i, text in enumerate(data["documents"]):
                meta = data["metadatas"][i] if i < len(data["metadatas"]) else {}
                mem_id = data["ids"][i] if i < len(data["ids"]) else str(i)
                memories.append({
                    "id": mem_id,
                    "fact": text,
                    "category": meta.get("category", "general"),
                    "timestamp": meta.get("timestamp", "")
                })
        return memories
    except Exception:
        return []

def clear_all_memories():
    """Wipe all episodic memories."""
    vs = get_memory_vectorstore()
    try:
        data = vs._collection.get()
        if data and data.get("ids"):
            vs._collection.delete(ids=data["ids"])
    except Exception:
        pass

# --- Agent Tools for Episodic Memory ---

@tool
def recall_past_memory(query: str) -> str:
    """
    Search long-term episodic memory for past user preferences, project conventions,
    architectural decisions, or previous discussion topics across all sessions.
    Args:
        query: What topic, preference, or fact to recall (e.g. 'coding style preferences', 'favorite libraries').
    """
    mems = recall_memories(query, top_k=4)
    if not mems:
        return "No relevant past memories found in long-term memory."
    formatted = ["🧠 **Recalled Long-Term Memories:**"]
    for m in mems:
        formatted.append(f"- `[{m['category'].upper()}]` {m['fact']} *(Saved: {m['timestamp'][:10]})*")
    return "\n".join(formatted)

@tool
def store_episodic_memory(fact: str, category: str = "general") -> str:
    """
    Save an important fact, user preference, project convention, or decision into long-term episodic memory.
    Args:
        fact: The specific insight or user preference to remember forever (e.g. 'User prefers concise responses with type annotations').
        category: Category of memory (e.g. 'preference', 'project_rule', 'architecture', 'user_info').
    """
    mem_id = save_memory(fact, category)
    return f"✅ Stored in long-term memory under [{category.upper()}]: \"{fact}\" (ID: {mem_id[:8]})"

# --- Context Window Compaction & Summarization Helper ---

def compact_messages_window(messages: List[BaseMessage], max_recent: int = 8) -> List[BaseMessage]:
    """
    Applies sliding-window compaction with background running summary.
    Preserves system message, ensures at least one HumanMessage (user query) is preserved,
    and maintains the most recent `max_recent` messages.
    """
    if not messages:
        return [HumanMessage(content="Hello")]

    system_msg = messages[0] if isinstance(messages[0], SystemMessage) else None
    working_msgs = messages[1:] if system_msg else messages
    
    if len(working_msgs) <= max_recent:
        return messages
        
    # Split into old messages (to summarize) and recent messages (to keep verbatim)
    old_msgs = working_msgs[:-max_recent]
    recent_msgs = working_msgs[-max_recent:]
    
    # Generate compact bullet points summary of older messages
    summary_lines = []
    last_old_human_msg = None
    for m in old_msgs:
        if isinstance(m, HumanMessage) and str(m.content).strip():
            last_old_human_msg = m
            role = "User"
        elif isinstance(m, AIMessage):
            role = "Assistant"
        else:
            role = "Tool"
        text = str(m.content)[:100].replace("\n", " ")
        if text.strip():
            summary_lines.append(f"- {role}: {text}...")
            
    summary_content = (
        "📌 [Previous Conversation Context Summary]:\n" +
        "\n".join(summary_lines[-6:]) + "\n" +
        "*(Earlier detailed messages compressed to preserve token budget)*"
    )
    
    summary_msg = SystemMessage(content=summary_content)
    
    # Check if recent_msgs already contains a valid user query (HumanMessage)
    has_recent_user_query = any(isinstance(m, HumanMessage) and bool(str(m.content).strip()) for m in recent_msgs)
    
    result = []
    if system_msg:
        result.append(system_msg)
    result.append(summary_msg)
    
    # If recent messages don't have a user message (e.g. only tool execution outputs),
    # carry forward the user query so Ollama/Qwen chat templates find the required user query.
    if not has_recent_user_query:
        if last_old_human_msg:
            result.append(last_old_human_msg)
        else:
            result.append(HumanMessage(content="Please continue with the user objective."))
            
    result.extend(recent_msgs)
    return result
