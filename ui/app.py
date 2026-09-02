import streamlit as st
import os
import sys

# Ensure the root directory is in the path to import src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agent.graph import app as agent_app, LLM_MODEL, LLM_BASE_URL
from src.tools.coding_tools import list_workspace_files, clean_workspace, WORKSPACE_DIR
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

st.set_page_config(page_title="Local Agentic RAG", page_icon="🤖", layout="wide")

# --- Sidebar: Workspace & Artifacts Controls ---
with st.sidebar:
    st.header("⚙️ Agent & Workspace Controls")
    
    st.markdown(f"**🤖 Model:** `{LLM_MODEL}`")
    st.markdown(f"**🔗 Endpoint:** `{LLM_BASE_URL}`")
    st.divider()
    
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

    # Display assistant response with Live Observability
    with st.chat_message("assistant"):
        final_response = ""
        tool_steps = []
        
        with st.status("🤖 Agent is analyzing & executing...", expanded=True) as status:
            try:
                # Format messages for LangGraph
                formatted_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        formatted_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        formatted_messages.append(AIMessage(content=msg["content"]))
                
                inputs = {"messages": formatted_messages}
                
                # Stream events from LangGraph
                for event in agent_app.stream(inputs, stream_mode="updates"):
                    for node_name, node_output in event.items():
                        if node_name == "agent":
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        tool_name = tc.get("name", "tool")
                                        tool_args = tc.get("args", {})
                                        st.markdown(f"🧠 **Model decided to use:** `{tool_name}`")
                                        with st.expander(f"📥 Input to `{tool_name}`", expanded=False):
                                            st.json(tool_args)
                                        tool_steps.append({
                                            "type": "call",
                                            "tool": tool_name,
                                            "args": tool_args
                                        })
                                elif getattr(msg, "content", None):
                                    final_response = msg.content
                                    
                        elif node_name == "action":
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                tool_name = getattr(msg, "name", "tool")
                                tool_content = getattr(msg, "content", "")
                                st.markdown(f"🛠️ **Tool Executed:** `{tool_name}`")
                                with st.expander(f"📤 Output from `{tool_name}`", expanded=False):
                                    st.code(tool_content if len(tool_content) <= 1500 else tool_content[:1500] + "\n...[truncated for display]")
                                tool_steps.append({
                                    "type": "result",
                                    "tool": tool_name,
                                    "output": tool_content[:1000]
                                })
                
                status.update(label="✅ Agent finished reasoning and executing tools", state="complete", expanded=False)
                
                if final_response:
                    st.markdown(final_response)
                else:
                    st.warning("Agent completed execution without generating a textual response.")
                    
                # Add assistant response with recorded steps to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "steps": tool_steps
                })
                
            except Exception as e:
                status.update(label="❌ Error occurred during agent execution", state="error", expanded=True)
                st.error(f"Error running agent: {e}")
                st.markdown("**Tip:** Ensure the ngrok URL in `.env` is reachable and Ollama is active.")

