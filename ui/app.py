import streamlit as st
import os
import sys
import difflib
import subprocess
import time
from pathlib import Path

# Ensure the root directory is in the path to import src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agent.graph import app as agent_app, LLM_MODEL, LLM_BASE_URL
from src.tools.coding_tools import (
    list_workspace_files,
    clean_workspace,
    WORKSPACE_DIR,
    read_local_file,
    write_local_file,
    execute_terminal_command
)
from src.rag.vectorstore import (
    save_and_ingest_uploaded_files,
    get_knowledge_base_stats,
    clear_vectorstore,
    reindex_all_data
)
from src.memory.episodic_memory import (
    list_all_memories,
    clear_all_memories,
    save_memory
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

    # 2. Episodic Long-Term Memory Manager
    st.subheader("🧠 Long-Term Memory")
    memories = list_all_memories()
    st.caption(f"**{len(memories)} fact(s)** stored in vector memory.")
    
    if memories:
        with st.expander("🔍 View Stored Memories", expanded=False):
            for m in memories:
                st.markdown(f"- `[{m['category'].upper()}]` {m['fact']}")
                
    with st.expander("➕ Store New Memory Fact", expanded=False):
        new_fact = st.text_input("Memory Fact:", key="new_mem_fact")
        new_cat = st.selectbox("Category:", ["preference", "project_rule", "architecture", "general"], key="new_mem_cat")
        if st.button("💾 Save to Memory", use_container_width=True):
            if new_fact.strip():
                save_memory(new_fact.strip(), new_cat)
                st.success("Memory saved!")
                st.rerun()

    if memories and st.button("🗑️ Clear Episodic Memory", use_container_width=True, type="secondary"):
        clear_all_memories()
        st.warning("All long-term memories cleared.")
        st.rerun()

    st.divider()

    # 3. Workspace Artifacts Quick Stats
    st.subheader("📁 Sandbox Status")
    workspace_files = list_workspace_files()
    if workspace_files:
        st.info(f"**{len(workspace_files)} file(s)** active in `./workspace`")
    else:
        st.caption("No artifacts currently in `./workspace`.")
        
    if st.button("🧹 Clean Workspace Files", use_container_width=True, type="secondary"):
        clean_workspace()
        st.success("Workspace cleaned! All generated artifacts wiped.")
        st.rerun()

    st.divider()
    
    # 4. Chat Controls
    st.subheader("💬 Chat Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.success("Chat history cleared.")
        st.rerun()


# --- Main App Header ---
st.title("🤖 Local Agentic RAG Portfolio Project")

tab_chat, tab_workbench = st.tabs([
    "💬 Agent Chat & Live Observability",
    "🛠️ Interactive Artifact & Code Diff Workbench"
])

# =========================================================================
# TAB 1: AGENT CHAT & LIVE OBSERVABILITY
# =========================================================================
with tab_chat:
    st.markdown("""
    Fully local, free-to-run AI Agent with **Live Reasoning & Observability**:
    - **LangGraph StateGraph**: Stateful loop with real-time thought inspection
    - **Live Tool Observability**: See queries, tool calls, and outputs stream as they happen (`st.status`)
    - **Self-RAG / CRAG & Reflexion Auto-Debugger**: Continuous document relevance and code self-healing
    - **Parallel Subagents**: Multi-slot parallel subagent dispatch on Ollama GPU
    """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages with step history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
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
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            final_response = ""
            tool_steps = []
            registered_calls = set()
            
            status_box = st.status("🤖 Agent is analyzing & executing...", expanded=True)
            response_placeholder = st.empty()
            status_closed = False
            
            try:
                formatted_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        formatted_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        formatted_messages.append(AIMessage(content=msg["content"]))
                
                inputs = {"messages": formatted_messages}
                
                for chunk, meta in agent_app.stream(inputs, stream_mode="messages"):
                    node_name = meta.get("langgraph_node", "")
                    
                    if node_name == "agent":
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
                                elif "Reflexion Auto-Debugger: Execution Failure Detected" in content:
                                    st.markdown("🔧 **Reflexion Auto-Debugger:** `Execution error caught! Guiding agent into auto-fix loop...`")
                                elif "Reflexion Auto-Debugger: Execution verified" in content:
                                    st.markdown("🎯 **Reflexion Auto-Debugger:** `Code execution verified with zero errors.`")

                if not status_closed:
                    status_box.update(label="✅ Agent finished reasoning and executing", state="complete", expanded=False)
                    
                if final_response:
                    response_placeholder.markdown(final_response)
                else:
                    st.warning("Agent completed execution without generating a textual response.")
                    
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "steps": tool_steps
                })
                
            except Exception as e:
                status_box.update(label="❌ Error occurred during execution", state="error", expanded=True)
                st.error(f"Error during agent execution: {str(e)}")
                st.markdown("**Tip:** Ensure the ngrok URL in `.env` is reachable and Ollama is active.")

# =========================================================================
# TAB 2: INTERACTIVE ARTIFACT & CODE DIFF WORKBENCH
# =========================================================================
with tab_workbench:
    st.markdown("### 🛠️ Interactive Sandbox Workbench & Code Diff Explorer")
    st.caption("Inspect, compare, edit, download, and execute files generated in the isolated `./workspace` sandbox.")
    
    current_files = list_workspace_files()
    
    if not current_files:
        st.info("ℹ️ No files currently in `./workspace`. Ask the agent to write a script or create a new file below!")
        
        with st.expander("➕ Create New File in Sandbox", expanded=True):
            new_filename = st.text_input("Filename (e.g. `main.py`, `analysis.sql`, `config.json`)", key="new_empty_file")
            new_code = st.text_area("File Content", height=200, key="new_empty_code")
            if st.button("💾 Create File", type="primary"):
                if new_filename.strip():
                    write_local_file.invoke({"file_path": new_filename.strip(), "content": new_code})
                    st.success(f"Created `{new_filename.strip()}` in sandbox!")
                    st.rerun()
                else:
                    st.warning("Please provide a filename.")
    else:
        # File selector and metrics
        col_select, col_stats = st.columns([3, 1])
        with col_select:
            selected_file = st.selectbox("📂 Select Sandbox File:", current_files)
        
        file_path = os.path.join(WORKSPACE_DIR, selected_file)
        file_size_kb = round(os.path.getsize(file_path) / 1024, 2)
        ext = os.path.splitext(selected_file)[1].lstrip(".").lower() or "text"
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            file_content = f.read()
        lines_count = len(file_content.splitlines())
        
        with col_stats:
            st.metric("File Size", f"{file_size_kb} KB", f"{lines_count} lines")
            
        subtab_viewer, subtab_diff, subtab_runner, subtab_editor = st.tabs([
            "📄 Code & Document Viewer",
            "🔍 Visual Code Diff Explorer",
            "⚡ Sandbox Terminal Runner",
            "✏️ In-Browser Editor"
        ])
        
        # 1. Code Viewer Sub-Tab
        with subtab_viewer:
            st.markdown(f"**Viewing:** `{selected_file}`")
            lang_map = {"py": "python", "js": "javascript", "json": "json", "sql": "sql", "md": "markdown", "csv": "csv", "sh": "bash"}
            st.code(file_content, language=lang_map.get(ext, "text"))
            
            col_d1, col_d2 = st.columns([1, 4])
            with col_d1:
                st.download_button(
                    label="📥 Download File",
                    data=file_content,
                    file_name=selected_file,
                    mime="text/plain",
                    use_container_width=True
                )
                
        # 2. Visual Diff Explorer Sub-Tab
        with subtab_diff:
            st.markdown("#### 🔍 Side-by-Side / Unified Git-Style Diff")
            st.caption("Compare the current sandbox file against another file or baseline version.")
            
            compare_col1, compare_col2 = st.columns(2)
            with compare_col1:
                st.markdown(f"**Baseline File A:** `{selected_file}`")
            with compare_col2:
                other_files = [f for f in current_files if f != selected_file]
                if other_files:
                    compare_target = st.selectbox("Compare against File B:", other_files)
                    target_path = os.path.join(WORKSPACE_DIR, compare_target)
                    with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                        compare_content = f.read()
                else:
                    compare_target = "Custom Baseline"
                    compare_content = st.text_area("Paste Baseline Code to Compare:", height=150)
                    
            if compare_content is not None:
                a_lines = file_content.splitlines(keepends=True)
                b_lines = compare_content.splitlines(keepends=True)
                
                diff = list(difflib.unified_diff(
                    b_lines,
                    a_lines,
                    fromfile=f"Baseline ({compare_target})",
                    tofile=f"Current ({selected_file})"
                ))
                
                if not diff:
                    st.success("✅ Files are identical. No differences found.")
                else:
                    diff_text = "".join(diff)
                    st.markdown("**Diff Output:**")
                    st.code(diff_text, language="diff")
                    
        # 3. Sandbox Terminal Runner Sub-Tab
        with subtab_runner:
            st.markdown("#### ⚡ Execute Sandbox Script Live")
            st.caption(f"Run `{selected_file}` securely inside `./workspace`.")
            
            custom_cmd = st.text_input(
                "Terminal Command:",
                value=f"python {selected_file}" if ext == "py" else f"cat {selected_file}",
                key="runner_custom_cmd"
            )
            
            if st.button("🚀 Run Command in Sandbox", type="primary"):
                with st.spinner("Executing in sandbox..."):
                    start_t = time.time()
                    res = execute_terminal_command.invoke({"command": custom_cmd})
                    dur = round((time.time() - start_t) * 1000, 1)
                    
                    st.markdown(f"⏱️ **Execution Time:** `{dur} ms`")
                    st.code(res, language="text")
                    
        # 4. In-Browser Editor Sub-Tab
        with subtab_editor:
            st.markdown(f"#### ✏️ Edit `{selected_file}`")
            edited_content = st.text_area("Code Editor", value=file_content, height=350, key="editor_textarea")
            
            col_save, col_new = st.columns([1, 1])
            with col_save:
                if st.button("💾 Save Changes to Sandbox", type="primary", use_container_width=True):
                    write_local_file.invoke({"file_path": selected_file, "content": edited_content})
                    st.success(f"Saved `{selected_file}` successfully!")
                    st.rerun()
            with col_new:
                if st.button("🗑️ Delete Selected File", type="secondary", use_container_width=True):
                    os.remove(file_path)
                    st.warning(f"Deleted `{selected_file}`.")
                    st.rerun()
