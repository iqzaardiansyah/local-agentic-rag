import streamlit as st
import os
import sys
import difflib
import time

# Ensure the root directory is in the path to import src
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
sys.path.append(os.path.dirname(__file__))

from src.agent.graph import LLM_MODEL, LLM_BASE_URL
from src.agent.health import check_llm_endpoint, format_health_badge
from src.tools.coding_tools import (
    list_workspace_files,
    clean_workspace,
    WORKSPACE_DIR,
    write_local_file,
    execute_terminal_command,
    IS_WINDOWS,
)
from src.rag.vectorstore import (
    save_and_ingest_uploaded_files,
    get_knowledge_base_stats,
    clear_vectorstore,
    reindex_all_data
)
from src.rag.hybrid_search import get_bm25_status, rebuild_bm25_index
from src.rag.graph_rag import get_kg_engine
from src.memory.episodic_memory import (
    list_all_memories,
    clear_all_memories,
    save_memory
)
from src.memory import chat_sessions
from src.memory.session_context import build_agent_messages
from src.memory.auto_memory import maybe_autocapture_preferences
from src.rag.citations import get_citations, clear_citations
from src.agent.runner import run_agent_stream
from src.agent.metrics import format_metrics
from citation_view import render_citations

st.set_page_config(page_title="Local Agentic RAG", page_icon="🤖", layout="wide")

# --- Sidebar: Workspace & Artifacts Controls ---
with st.sidebar:
    st.header("⚙️ Agent & Knowledge Base")

    st.markdown(f"**🤖 Model:** `{LLM_MODEL}`")
    st.markdown(f"**🔗 Endpoint:** `{LLM_BASE_URL}`")

    # --- LLM endpoint health (cached ~30s to avoid hammering) ---
    if "llm_health" not in st.session_state or "llm_health_ts" not in st.session_state:
        st.session_state.llm_health = None
        st.session_state.llm_health_ts = 0.0

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        age = time.time() - st.session_state.llm_health_ts
        if st.session_state.llm_health is None or age > 30:
            with st.spinner("Checking LLM endpoint..."):
                st.session_state.llm_health = check_llm_endpoint(LLM_BASE_URL)
                st.session_state.llm_health_ts = time.time()
        hh = st.session_state.llm_health or {}
        if hh.get("ok"):
            st.success(format_health_badge(hh))
        else:
            st.error(format_health_badge(hh))
            st.caption("Start Ollama / your ngrok notebook, then hit ↻.")
    with col_h2:
        if st.button("↻", key="recheck_llm", help="Re-check LLM endpoint"):
            st.session_state.llm_health = check_llm_endpoint(LLM_BASE_URL)
            st.session_state.llm_health_ts = time.time()
            st.rerun()

    st.divider()

    # 1. Document Upload & Knowledge Base Management
    st.subheader("📚 Knowledge Base Manager")
    kb_stats = get_knowledge_base_stats()
    kg_stats = get_kg_engine().get_graph_stats()
    bm25_stats = get_bm25_status()

    col_k1, col_k2 = st.columns(2)
    col_k1.metric("Chroma Chunks", kb_stats["total_chunks"])
    col_k2.metric("KG Entities", kg_stats["total_entities"])

    st.caption(f"🕸️ **GraphRAG:** `{kg_stats['total_relationships']}` relational triples indexed.")

    bm25_state = "fresh" if not bm25_stats.get("stale") else "stale"
    bm25_color = "🟢" if bm25_state == "fresh" else "🟡"
    st.caption(
        f"{bm25_color} **BM25:** {bm25_stats.get('documents', 0)} docs "
        f"(chroma={bm25_stats.get('chroma_count', '?')}) · {bm25_state}"
    )
    if bm25_stats.get("stale") and bm25_stats.get("reason"):
        st.caption(f"Reason: {bm25_stats['reason']}")
    if st.button("🔁 Rebuild BM25 Index", use_container_width=True):
        with st.spinner("Rebuilding BM25 from Chroma..."):
            res = rebuild_bm25_index()
        if res.get("ok"):
            st.success(f"BM25 rebuilt from {res.get('documents')} chunks.")
        else:
            st.error(f"BM25 rebuild failed: {res.get('error')}")
        st.rerun()
    
    with st.expander("📤 Upload & Ingest Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "csv", "json", "py"],
            accept_multiple_files=True,
            key="kb_uploader"
        )
        if uploaded_files:
            if st.button("📥 Index Uploaded Files", use_container_width=True, type="primary"):
                with st.spinner("Embedding and adding to ChromaDB & GraphRAG..."):
                    res = save_and_ingest_uploaded_files(uploaded_files)
                    if res.get("success"):
                        st.success(f"Indexed {res['total_documents']} document(s) into {res['chunks']} chunks!")
                        st.rerun()
                    else:
                        st.error(res.get("message", "Ingestion failed."))

    with st.expander("🌐 Ingest URL into Knowledge Base", expanded=False):
        url_to_ingest = st.text_input(
            "Page URL",
            placeholder="https://example.com/docs/page",
            key="kb_url_input",
        )
        if st.button("🌐 Fetch & Index URL", use_container_width=True, type="primary"):
            if not url_to_ingest.strip():
                st.warning("Enter a URL first.")
            else:
                from src.rag.ingest_url import ingest_url_to_kb

                with st.spinner("Scraping, embedding, extracting triples..."):
                    try:
                        res = ingest_url_to_kb(url_to_ingest.strip())
                    except Exception as e:
                        res = {"success": False, "message": str(e)}
                if res.get("success"):
                    st.success(
                        f"Indexed `{res.get('title') or res.get('file')}` — "
                        f"{res.get('chunks')} chunks, {res.get('triples')} triples."
                    )
                    st.rerun()
                else:
                    st.error(res.get("message", "Ingest failed."))

    with st.expander("🧪 Offline RAG Eval", expanded=False):
        st.caption("Generate Hit@k cases from files in `data/` (no LLM needed).")
        col_g1, col_g2 = st.columns(2)
        if col_g1.button("🧬 Generate eval cases", use_container_width=True):
            from src.eval.auto_eval_set import write_eval_set

            info = write_eval_set(replace=False)
            st.success(
                f"{info['total']} cases in `{info['path']}` "
                f"({info['generated']} new this run)."
            )
        if col_g2.button("📊 Run eval metrics", use_container_width=True):
            from src.eval.rag_eval import evaluate_all

            with st.spinner("Running hybrid retrieval eval..."):
                report = evaluate_all(top_k=3)
            if report.get("n"):
                st.metric("Hit Rate@3", f"{report['hit_rate_at_k']:.3f}")
                st.metric("MRR", f"{report['mrr']:.3f}")
                if report.get("avg_keyword_rate") is not None:
                    st.metric("Avg keyword", f"{report['avg_keyword_rate']:.3f}")
            else:
                st.warning("Eval set empty — generate cases first.")

    with st.expander("🕸️ Explore GraphRAG Triples", expanded=False):
        st.markdown("**Top Connected Entity Hubs:**")
        for hub in kg_stats.get("top_hubs", []):
            st.markdown(f"- **`{hub['entity']}`** ({hub['connections']} connections)")

    if kb_stats["files"]:
        with st.expander("📄 View Knowledge Base Files", expanded=False):
            for f in kb_stats["files"]:
                st.caption(f"• `{f}`")

    col_reindex, col_clear_kb = st.columns(2)
    if col_reindex.button("🔄 Re-Index", use_container_width=True):
        with st.spinner("Rebuilding ChromaDB & GraphRAG..."):
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

    if "auto_memory_enabled" not in st.session_state:
        st.session_state.auto_memory_enabled = True
    st.session_state.auto_memory_enabled = st.toggle(
        "✨ Auto-capture preferences from chat",
        value=st.session_state.auto_memory_enabled,
        help="Heuristically store phrases like 'I prefer…', 'always…', 'remember that…' into long-term memory.",
        key="auto_memory_toggle",
    )

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
    st.subheader("💬 Chat Sessions")
    all_sessions = chat_sessions.list_sessions()
    session_options = {s["id"]: f"{s['title']} · {s['updated_at'][:16]}" for s in all_sessions}

    if "active_session_id" not in st.session_state or st.session_state.active_session_id not in session_options:
        st.session_state.active_session_id = chat_sessions.ensure_active_session()
        st.session_state.messages = chat_sessions.load_messages(st.session_state.active_session_id)

    if session_options:
        selected = st.selectbox(
            "Active session",
            options=list(session_options.keys()),
            format_func=lambda sid: session_options[sid],
            index=list(session_options.keys()).index(st.session_state.active_session_id)
            if st.session_state.active_session_id in session_options
            else 0,
            key="session_picker",
        )
        if selected != st.session_state.active_session_id:
            st.session_state.active_session_id = selected
            st.session_state.messages = chat_sessions.load_messages(selected)
            st.rerun()

    col_new, col_del = st.columns(2)
    if col_new.button("➕ New Chat Session", use_container_width=True):
        new_s = chat_sessions.create_session("New Chat")
        st.session_state.active_session_id = new_s["id"]
        st.session_state.messages = []
        st.success("Started a new chat session.")
        st.rerun()

    if len(all_sessions) > 1 and col_del.button("🗑️ Delete Session", use_container_width=True):
        chat_sessions.delete_session(st.session_state.active_session_id)
        st.session_state.active_session_id = chat_sessions.ensure_active_session()
        st.session_state.messages = chat_sessions.load_messages(st.session_state.active_session_id)
        st.warning("Session deleted.")
        st.rerun()

    # --- Cross-session search ---
    with st.expander("🔎 Search all sessions", expanded=False):
        search_q = st.text_input(
            "Find text in any chat",
            key="xsession_search_q",
            placeholder="e.g. hybrid search, pandas, user prefers…",
        )
        if search_q.strip():
            hits = chat_sessions.search_messages(search_q.strip(), limit=20)
            if not hits:
                st.caption("No matches.")
            else:
                st.caption(f"{len(hits)} hit(s)")
                for i, h in enumerate(hits):
                    title = h.get("session_title") or "Untitled"
                    role = "🧑" if h.get("role") == "user" else "🤖"
                    st.markdown(f"**{role} `{title}`** · {h.get('created_at', '')[:16]}")
                    st.caption(h.get("snippet", ""))
                    if st.button(
                        "Open session",
                        key=f"xhit_{i}_{h['session_id'][:8]}",
                        use_container_width=False,
                    ):
                        st.session_state.active_session_id = h["session_id"]
                        st.session_state.messages = chat_sessions.load_messages(h["session_id"])
                        st.rerun()

    md_export = chat_sessions.export_session_markdown(st.session_state.active_session_id)
    st.download_button(
        "📥 Export Session as Markdown",
        data=md_export,
        file_name=f"chat-session-{st.session_state.active_session_id[:8]}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    if st.button("🗑️ Clear Current Chat History", use_container_width=True, type="secondary"):
        chat_sessions.clear_messages(st.session_state.active_session_id)
        st.session_state.messages = []
        st.success("Chat history cleared for this session.")
        st.rerun()


# --- Main App Header ---
st.title("🤖 Local Agentic RAG Portfolio Project")


def _run_agent_turn_ui(prompt: str) -> None:
    """Stream one agent turn into the chat UI (shared by send + regenerate)."""
    final_response = ""
    tool_steps = []
    clear_citations()
    last_metrics = {}

    status_box = st.status("🤖 Agent is analyzing & executing...", expanded=True)
    response_placeholder = st.empty()
    status_closed = False

    try:
        hh = st.session_state.get("llm_health") or {}
        if not hh.get("ok"):
            st.warning(
                "LLM endpoint looks offline. The agent may fail. "
                "Check Ollama / ngrok in the sidebar (↻)."
            )

        agent_messages = build_agent_messages(
            prompt,
            session_id=st.session_state.active_session_id,
            include_cross_session=True,
        )

        for event in run_agent_stream(
            prompt,
            session_id=st.session_state.active_session_id,
            history=agent_messages,
        ):
            etype = event.get("type")

            if etype == "meta":
                with status_box:
                    st.caption(
                        f"Context: {event.get('message_count', 0)} messages · "
                        f"session `{(event.get('session_id') or '')[:8]}`"
                    )

            elif etype == "tool_call":
                tool_name = event.get("tool", "tool")
                tool_args = event.get("args", {})
                with status_box:
                    st.markdown(f"🧠 **Model decided to use:** `{tool_name}`")
                    with st.expander(f"📥 Input to `{tool_name}`", expanded=False):
                        st.json(tool_args)
                tool_steps.append({"type": "call", "tool": tool_name, "args": tool_args})

            elif etype == "token":
                if not status_closed and len(tool_steps) > 0:
                    status_box.update(
                        label="✅ Tools executed. Streaming final answer...",
                        state="complete",
                        expanded=False,
                    )
                    status_closed = True
                final_response += event.get("token", "")
                response_placeholder.markdown(final_response + "▌")

            elif etype == "tool_result":
                tool_name = event.get("tool", "tool")
                tool_content = event.get("output", "") or ""
                with status_box:
                    st.markdown(f"🛠️ **Tool Executed:** `{tool_name}`")
                    with st.expander(f"📤 Output from `{tool_name}`", expanded=False):
                        st.code(
                            tool_content if len(tool_content) <= 1500
                            else tool_content[:1500] + "\n...[truncated for display]"
                        )
                tool_steps.append({
                    "type": "result",
                    "tool": tool_name,
                    "output": tool_content[:1000]
                })

            elif etype == "grade":
                label = event.get("label", "")
                detail = event.get("detail", "")
                with status_box:
                    if label.startswith("crag_high"):
                        st.markdown(f"🎯 **CRAG Grader:** `{detail}`")
                    elif label.startswith("crag_low"):
                        st.markdown(f"⚠️ **CRAG Grader:** `{detail}`")
                    elif label.startswith("reflexion"):
                        st.markdown(f"🔧 **Reflexion:** `{detail}`")

            elif etype == "subagent_start":
                with status_box:
                    st.markdown(
                        f"⚡ **Subagent started:** `{event.get('name')}` "
                        f"({event.get('role')}) · {event.get('index', 0)+1}/{event.get('total', '?')}"
                    )
                    st.caption(event.get("task", "")[:180])

            elif etype == "subagent_done":
                status = event.get("status", "")
                icon = "✅" if status == "success" else "❌"
                with status_box:
                    st.markdown(
                        f"{icon} **Subagent finished:** `{event.get('name')}` "
                        f"({status}) · {event.get('elapsed_ms', '?')} ms"
                    )
                    st.caption(event.get("result_preview", "")[:200])

            elif etype == "subagents_all_done":
                with status_box:
                    st.markdown(
                        f"⚡ **Parallel fan-out complete:** "
                        f"{event.get('success', 0)} ok / {event.get('error', 0)} error "
                        f"of {event.get('count', 0)}"
                    )

            elif etype == "error":
                last_metrics = event.get("metrics") or {}
                status_box.update(
                    label="❌ Error occurred during execution",
                    state="error",
                    expanded=True,
                )
                st.error(f"Error during agent execution: {event.get('error')}")
                st.markdown("**Tip:** Ensure the ngrok URL in `.env` is reachable and Ollama is active.")
                final_response = final_response or f"Error: {event.get('error')}"
                break

            elif etype == "done":
                last_metrics = event.get("metrics") or {}

        if not status_closed:
            status_box.update(
                label="✅ Agent finished reasoning and executing",
                state="complete",
                expanded=False,
            )

        citations = get_citations()
        if final_response:
            response_placeholder.markdown(final_response)
        else:
            st.warning("Agent completed execution without generating a textual response.")

        if last_metrics:
            st.caption("⏱️ " + format_metrics(last_metrics))

        render_citations(citations, key_prefix="live", expanded=bool(citations))

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_response,
            "steps": tool_steps,
            "citations": citations,
            "metrics": last_metrics,
        })
        chat_sessions.save_message(
            st.session_state.active_session_id,
            "assistant",
            final_response,
            steps=tool_steps,
            citations=citations,
        )

    except Exception as e:
        status_box.update(label="❌ Error occurred during execution", state="error", expanded=True)
        st.error(f"Error during agent execution: {str(e)}")
        st.markdown("**Tip:** Ensure the ngrok URL in `.env` is reachable and Ollama is active.")


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

    # Regenerate pending (set by button; runs after history render)
    regenerate_prompt = st.session_state.pop("regenerate_prompt", None)

    # Display chat messages with step history, metrics, fork, and citations
    last_assistant_idx = None
    for msg_idx, message in enumerate(st.session_state.messages):
        if message["role"] == "assistant":
            last_assistant_idx = msg_idx
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
            if message.get("metrics"):
                st.caption("⏱️ " + format_metrics(message["metrics"]))
            if message.get("citations"):
                render_citations(
                    message["citations"],
                    key_prefix=f"hist_{msg_idx}",
                    expanded=False,
                )

            col_a, col_b = st.columns([1, 1])
            msg_db_id = message.get("id")
            with col_a:
                if msg_db_id is not None and st.button(
                    "🔀 Fork here",
                    key=f"fork_{msg_idx}_{msg_db_id}",
                    help="Create a new session with history up to this message",
                ):
                    forked = chat_sessions.fork_session(
                        st.session_state.active_session_id,
                        up_to_message_id=msg_db_id,
                    )
                    st.session_state.active_session_id = forked["id"]
                    st.session_state.messages = chat_sessions.load_messages(forked["id"])
                    st.success(f"Forked to `{forked['title']}` ({forked['message_count']} messages).")
                    st.rerun()
            with col_b:
                if (
                    message["role"] == "assistant"
                    and msg_idx == last_assistant_idx
                    and st.button(
                        "🔄 Regenerate",
                        key=f"regen_{msg_idx}",
                        help="Delete this reply and re-run the last user message",
                    )
                ):
                    chat_sessions.delete_last_assistant_reply(
                        st.session_state.active_session_id
                    )
                    last_user = chat_sessions.last_user_message(
                        st.session_state.active_session_id
                    )
                    st.session_state.messages = chat_sessions.load_messages(
                        st.session_state.active_session_id
                    )
                    if last_user:
                        st.session_state.regenerate_prompt = last_user
                    st.rerun()

    # Handle regenerate (user message already in history; do not save user again)
    if regenerate_prompt:
        with st.chat_message("assistant"):
            _run_agent_turn_ui(regenerate_prompt)
        st.rerun()

    # Accept user input
    if prompt := st.chat_input("Ask a question, request data analysis, web research, or code execution..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_sessions.save_message(st.session_state.active_session_id, "user", prompt)

        # Auto-capture durable preferences (free heuristics, no extra LLM call)
        auto_saved = maybe_autocapture_preferences(
            prompt,
            session_id=st.session_state.active_session_id,
            enabled=st.session_state.get("auto_memory_enabled", True),
        )
        if auto_saved:
            with st.chat_message("assistant"):
                st.caption(
                    "🧠 Auto-saved to long-term memory: "
                    + "; ".join(f"`[{a['category']}]` {a['fact']}" for a in auto_saved)
                )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            _run_agent_turn_ui(prompt)

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
                value=(
                    f"python {selected_file}" if ext == "py"
                    else (f"Get-Content {selected_file}" if IS_WINDOWS else f"cat {selected_file}")
                ),
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
