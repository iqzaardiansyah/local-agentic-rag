import streamlit as st
import os
import sys

# Ensure the root directory is in the path to import src
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.agent.graph import app as agent_app
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Local Agentic RAG", page_icon="🤖", layout="wide")

st.title("🤖 Local Agentic RAG Portfolio Project")
st.markdown("""
This is a fully local, free-to-run AI Agent demonstrating:
- **LangGraph** for stateful orchestration
- **Data Analysis (ChatGPT-style)**: Built-in Python execution REPL
- **Agentic Coding (Claude Code-style)**: Can read/write files and execute universal terminal commands (Node, Bash, etc.)
- **Model Context Protocol (MCP)** via a custom local data server
- **RAG** using ChromaDB and HuggingFace Embeddings
- **Web Browsing**: DuckDuckGo search + Beautifulsoup Scraper
- Powered by a remote **Ollama** server via ngrok!
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the documents or employee database..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            try:
                # Format messages for LangGraph
                formatted_messages = []
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        formatted_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        formatted_messages.append(AIMessage(content=msg["content"]))
                
                # Run the graph
                inputs = {"messages": formatted_messages}
                final_state = agent_app.invoke(inputs)
                
                # Get the last message content
                response = final_state["messages"][-1].content
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"Error running agent: {e}")
                st.markdown("**Tip:** Ensure the ngrok URL in `.env` is correct and the Ollama server is running. Also check if the MCP server is running on port 8000.")
