import streamlit as st
import os
import sys

# Ensure the root directory is in the path to import src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agent.graph import app as agent_app, LLM_MODEL, LLM_BASE_URL
from src.tools.coding_tools import list_workspace_files, clean_workspace, WORKSPACE_DIR
from src.rag.vectorstore import (
    save_and_ingest_uploaded_files,
    get_knowledge_base_stats,
    clear_vectorstore,
    reindex_all_data
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

st.set_page_config(page_title="Local Agentic RAG", page_icon="🤖", layout="wide")

# --- Sidebar: Workspace & Artifacts Controls ---
with st.sidebar:
    st.header("⚙️ Agent & Knowledge Base")
    
    st.markdown(f"**🤖 Model:** `{LLM_MODEL}`")
    st.markdown(f"**🔗 Endpoint:** `{LLM_BASE_URL}`")
    st.divider()
    
    # 1. Document Upload & Knowledge Base Management
    st.subheader("📚 Knowledge Base Manager")
    kb_stats = get_knowledge_base_stats()
    col_k1, col_k2 = st.columns(2)
    col_k1.metric("Indexed Docs", kb_stats["total_files"])
    col_k2.metric("Chroma Chunks", kb_stats["total_chunks"])
    
    with st.expander("📤 Upload & Ingest Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "csv", "json", "py"],
            accept_multiple_files=True,
            key="kb_uploader"
        )
        if uploaded_files:
            if st.button("📥 Index Uploaded Files", use_container_width=True, type="primary"):
                with st.spinner("Embedding and adding to ChromaDB..."):
                    res = save_and_ingest_uploaded_files(uploaded_files)
                    if res.get("success"):
                        st.success(f"Indexed {res['total_documents']} document(s) into {res['chunks']} chunks!")
                        st.rerun()
                    else:
                        st.error(res.get("message", "Ingestion failed."))

    if kb_stats["files"]:
        with st.expander("📄 View Knowledge Base Files", expanded=False):
            for f in kb_stats["files"]:
                st.caption(f"• `{f}`")

    col_reindex, col_clear_kb = st.columns(2)
    if col_reindex.button("🔄 Re-Index", use_container_width=True):
        with st.spinner("Rebuilding ChromaDB..."):
            cnt = reindex_all_data()
            st.success(f"Re-indexed {cnt} chunks.")
            st.rerun()

    if col_clear_kb.button("🗑️ Wipe DB", use_container_width=True):
        clear_vectorstore()
        st.warning("Knowledge Base wiped.")
        st.rerun()

    st.divider()

    # 2. Workspace Artifacts
    st.subheader("📁 Workspace Artifacts")
    workspace_files = list_workspace_files()
    if workspace_files:
        st.info(f"**{len(workspace_files)} file(s)** generated in `./workspace`")
        with st.expander("View Workspace Files", expanded=False):
            for wf in workspace_files:
                st.code(wf, language="text")
    else:
        st.caption("No artifacts currently in `./workspace`.")
        
    if st.button("🧹 Clean Workspace Files", use_container_width=True, type="secondary"):
        clean_workspace()
        st.success("Workspace cleaned! All generated artifacts wiped.")
        st.rerun()

    st.divider()
    
    # 3. Chat Controls
    st.subheader("💬 Chat Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.success("Chat history cleared.")
        st.rerun()

# --- Main App Header ---
st.title("🤖 Local Agentic RAG Portfolio Project")
st.markdown("""
This is a fully local, free-to-run AI Agent with **Live Reasoning & Tool Observability**:
- **LangGraph StateGraph**: Stateful loop with real-time thought inspection
- **Live Tool Observability**: See queries, tool calls, and outputs stream as they happen (`st.status`)
- **Sandboxed Workspace**: All code and generated artifacts are safely isolated in `./workspace`
- **RAG & MCP Integration**: Local ChromaDB + HuggingFace Embeddings + Mock Employee Database
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages with step history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # If this message has previous intermediate steps, render them in an expander
        if "steps" in message and message["steps"]:
            with st.expander("🔍 View Agent Thought Trace & Tool History", expanded=False):
                for step in message["steps"]:
                    if step["type"] == "call":
                        st.markdown(f"🧠 **Decided to call:** `{step['tool']}`")
                        st.json(step.get("args", {}))
                    elif step["type"] == "result":
                        st.markdown(f"🛠️ **Tool Result:** `{step['tool']}`")
                        st.code(step.get("output", "")[:1000])
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question, request data analysis, web research, or code execution..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response with Live Observability & Token Streaming
    with st.chat_message("assistant"):
        final_response = ""
        tool_steps = []
        registered_calls = set()
        
        status_box = st.status("🤖 Agent is analyzing & executing...", expanded=True)
        response_placeholder = st.empty()
        status_closed = False
        
        try:
            # Format messages for LangGraph
            formatted_messages = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    formatted_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_messages.append(AIMessage(content=msg["content"]))
            
            inputs = {"messages": formatted_messages}
            
            # Stream token and tool events from LangGraph
            for chunk, meta in agent_app.stream(inputs, stream_mode="messages"):
                node_name = meta.get("langgraph_node", "")
                
                if node_name == "agent":
                    # 1. Handle tool invocations
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            call_id = tc.get("id") or str(tc)
                            if call_id not in registered_calls:
                                registered_calls.add(call_id)
                                tool_name = tc.get("name", "tool")
                                tool_args = tc.get("args", {})
                                with status_box:
                                    st.markdown(f"🧠 **Model decided to use:** `{tool_name}`")
                                    with st.expander(f"📥 Input to `{tool_name}`", expanded=False):
                                        st.json(tool_args)
                                tool_steps.append({
                                    "type": "call",
                                    "tool": tool_name,
                                    "args": tool_args
                                })
                    # 2. Handle token-by-token streaming
                    elif getattr(chunk, "content", None):
                        if not status_closed and len(tool_steps) > 0:
                            status_box.update(label="✅ Tools executed. Streaming final answer...", state="complete", expanded=False)
                            status_closed = True
                        token = chunk.content
                        final_response += token
                        response_placeholder.markdown(final_response + "▌")
                        
                elif node_name == "action":
                    if isinstance(chunk, ToolMessage):
                        tool_name = getattr(chunk, "name", "tool")
                        tool_content = getattr(chunk, "content", "")
                        with status_box:
                            st.markdown(f"🛠️ **Tool Executed:** `{tool_name}`")
                            with st.expander(f"📤 Output from `{tool_name}`", expanded=False):
                                st.code(tool_content if len(tool_content) <= 1500 else tool_content[:1500] + "\n...[truncated for display]")
                        tool_steps.append({
                            "type": "result",
                            "tool": tool_name,
                            "output": tool_content[:1000]
                        })
                        
                elif node_name == "grade_retrieval":
                    if isinstance(chunk, ToolMessage):
                        content = getattr(chunk, "content", "")
                        with status_box:
                            if "High Confidence Match" in content:
                                st.markdown("🎯 **CRAG Grader:** `Verified local documents as relevant & grounded.`")
                            elif "Low Local Document Relevance" in content:
                                st.markdown("⚠️ **CRAG Grader:** `Low relevance score. Guiding agent to avoid hallucination.`")

            if not status_closed:
                status_box.update(label="✅ Agent finished reasoning and executing", state="complete", expanded=False)
                
            if final_response:
                response_placeholder.markdown(final_response)
            else:
                st.warning("Agent completed execution without generating a textual response.")
                
            # Add assistant response with recorded steps to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "steps": tool_steps
            })
            
        except Exception as e:
            status_box.update(label="❌ Error occurred during execution", state="error", expanded=True)
            st.error(f"Error during agent execution: {str(e)}")
            st.markdown("**Tip:** Ensure the ngrok URL in `.env` is reachable and Ollama is active.")
