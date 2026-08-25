# 🤖 Local Agentic RAG: A Free, Agentic RAG System

A portfolio-ready AI Engineer project demonstrating cutting-edge agentic workflows, Information Retrieval (RAG), and Model Context Protocol (MCP) concepts—all completely **free to run** using a remote Ollama server via ngrok.

## 🌟 Key Features
- **LangGraph Orchestration:** Replaces rigid LangChain chains with a cyclic, stateful agent graph capable of reasoning and routing.
- **Agentic Coding Workspace (Claude Code / Antigravity style):** The agent can read/write files and execute universal terminal commands (Bash, Node.js, compiling C++, etc.) directly on your local system.
- **Python Code Execution (ChatGPT Data Analysis style):** Built-in Python REPL tool allows the agent to write and execute code locally to solve math, process data, or test logic safely.
- **Web Scraping & Search:** Includes DuckDuckGo integration and a Beautifulsoup-based Web Scraper to browse and read the live internet.
- **Model Context Protocol (MCP) Integration:** Features a local FastAPI server simulating an MCP-compliant data source (employee database) that the agent can dynamically query using tool calls.
- **Local RAG (Retrieval-Augmented Generation):** Uses local HuggingFace embeddings (`all-MiniLM-L6-v2`) and a local ChromaDB instance to search documents without exposing data to external APIs.
- **Cost-Free LLM Hosting:** Designed to connect to a free Colab/Kaggle notebook running an Ollama instance exposed via ngrok.

## 📁 Repository Structure
- `src/agent/`: The LangGraph state machine definition (`graph.py`).
- `src/rag/`: Document ingestion and ChromaDB vector store initialization.
- `src/mcp_server/`: A local FastAPI server exposing structured data (simulating an MCP server).
- `src/tools/`: The client tools the agent uses to interact with RAG and the MCP server.
- `ui/`: The Streamlit chat interface.

## 🚀 How to Run

### 1. Start the Remote LLM (Free)
1. Upload the `notebooks/lm-server.ipynb` notebook to a free Kaggle or Google Colab instance.
2. Add your ngrok Auth Token to the notebook secrets.
3. Run the notebook. It will output a public ngrok URL (e.g., `https://1234-abcd.ngrok-free.app/v1`).

### 2. Setup the Local App
Clone this repository and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment
Rename `.env.example` to `.env` and paste your ngrok URL:
```env
LLM_BASE_URL=https://<your-ngrok-url>.ngrok-free.app/v1
LLM_MODEL=qwen3.6:latest
LLM_API_KEY=ollama
```

### 4. Ingest Documents (Optional)
Put some `.txt` or `.pdf` files into the `data/` folder, then run:
```bash
python src/rag/vectorstore.py
```

### 5. Start the MCP Server
In a new terminal window, start the local data server:
```bash
python src/mcp_server/server.py
```

### 6. Run the App!
In another terminal, launch the UI:
```bash
streamlit run ui/app.py
```
