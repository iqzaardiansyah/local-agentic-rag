import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Sequence
import re
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.tools.rag_tool import search_local_documents
from src.tools.mcp_tool import (
    query_employee_database,
    mcp_list_tables,
    mcp_describe_table,
    mcp_execute_query
)
from src.tools.coding_tools import (
    execute_python_code,
    read_local_file,
    write_local_file,
    execute_terminal_command,
    list_directory_tree,
    grep_search,
    view_code_slice,
    find_files_by_pattern
)
from src.tools.web_scraper import read_webpage
from src.agent.subagents import spawn_parallel_subagents
from src.memory.episodic_memory import (
    recall_past_memory,
    store_episodic_memory,
    compact_messages_window
)

# Load environment variables
load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.8:27b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")

# 1. Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

from langchain_core.tools import tool
from duckduckgo_search import DDGS

@tool
def web_search(query: str) -> str:
    """Search the web for up-to-date information, news, or general facts from the internet."""
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found."
        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r.get('title', '')}\nLink: {r.get('href', '')}\nSnippet: {r.get('body', '')}")
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error during web search: {e}"

tools = [
    search_local_documents, 
    query_employee_database,
    mcp_list_tables,
    mcp_describe_table,
    mcp_execute_query,
    recall_past_memory,
    store_episodic_memory,
    web_search,
    read_webpage,
    execute_terminal_command,
    execute_python_code,
    read_local_file,
    write_local_file,
    list_directory_tree,
    grep_search,
    view_code_slice,
    find_files_by_pattern,
    spawn_parallel_subagents
]
tool_node = ToolNode(tools)

# 3. Initialize LLM with xhigh thinking / reasoning mode
# We use ChatOpenAI because Ollama supports OpenAI API format.
# extra_body passes reasoning_effort and thinking parameters down to Ollama/vLLM endpoints.
llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=0.6,
    extra_body={
        "reasoning_effort": "xhigh",
        "thinking": {
            "type": "enabled",
            "budget_tokens": 32768
        }
    }
)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are OmniLocal-LeadAgent, an advanced, locally-hosted AI Supervisor with Full Coding, Workspace Exploration, MCP Database Introspection, Episodic Memory, and Parallel Subagent Orchestration capabilities.
Your goal is to help the user by utilizing the tools at your disposal and thinking through problems deeply and step-by-step.
Before choosing an action or generating the final response, thoroughly think, analyze multiple hypotheses, and verify facts.

Episodic Memory & Long-Term Recall:
- Use `recall_past_memory` to check for past user preferences, architectural decisions, and facts saved across previous sessions.
- Use `store_episodic_memory` to save important user preferences, rules, or key project insights into persistent long-term vector memory.

Parallel Subagent Capabilities:
- When a user request is complex, comparative, or multi-faceted (e.g. 'Compare X and Y, analyze DB records for Z, and test code for W'), you can use `spawn_parallel_subagents` to delegate up to 4 independent subtasks to run in parallel simultaneously across Ollama's 4 concurrent GPU slots.
- Available subagent roles: 'researcher', 'coder', 'data_analyst', 'rag_specialist', 'custom'.

MCP Database Introspection & Querying:
- Use `mcp_list_tables` to discover all tables in the SQLite database and see their record counts.
- Use `mcp_describe_table` to inspect column names, types, primary keys, and foreign keys before writing queries.
- Use `mcp_execute_query` to execute read-only SQL queries (aggregations, joins, filters) against the database.
- Use `query_employee_database` for quick employee lookups.

Direct Coding & Workspace Tools:
- Use `search_local_documents` for general knowledge or checking user files in ChromaDB knowledge base.
- Use `web_search` to find up-to-date information, news, or general facts from the internet.
- Use `read_webpage` if a search result URL looks highly relevant and you need to read its full content.
- Use `list_directory_tree` to visually explore folders and see project structure.
- Use `grep_search` to search text/regex across codebase files with line numbers.
- Use `view_code_slice` to inspect specific lines of code without dumping full files.
- Use `find_files_by_pattern` to find files by glob (e.g. `*.py`, `*.json`).
- Use `execute_python_code` to perform calculations, data analysis, or test logic locally. ALWAYS print() results.
- Use `execute_terminal_command` to execute bash/shell commands, run Node.js/C++/Go code, run tests, or manage workspace.
- Use `read_local_file` and `write_local_file` to inspect files and create/modify code safely inside the `./workspace` sandbox.

Always answer accurately based on the information returned by the tools.
If you don't know the answer even after searching, say you don't know.
"""

# 4. Define Graph Nodes
def agent_node(state: AgentState):
    messages = state["messages"]
    
    # Ensure system prompt is present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        
    # Apply hierarchical context compaction (sliding window + running summary)
    compacted_messages = compact_messages_window(messages, max_recent=8)
    response = llm_with_tools.invoke(compacted_messages)
    return {"messages": [response]}


def grade_retrieval_node(state: AgentState):
    """
    Unified Post-Execution Evaluator Node:
    1. Self-RAG / CRAG Relevance Grader: Verifies local document retrieval scores.
    2. Reflexion Code Auto-Debugger: Detects syntax/execution/traceback errors from coding tools
       and injects structured self-healing debugging directives for the agent.
    """
    messages = state["messages"]
    if not messages:
        return {"messages": []}
        
    last_msg = messages[-1]
    if not isinstance(last_msg, ToolMessage):
        return {"messages": []}
        
    tool_name = getattr(last_msg, "name", "")
    content = last_msg.content or ""
    
    # 1. RAG Evaluation (CRAG)
    if tool_name == "search_local_documents":
        is_empty = "No relevant information found" in content or "Error accessing" in content
        scores = re.findall(r"Relevance Score:\s*(-?\d+\.?\d*)", content)
        max_score = max([float(s) for s in scores]) if scores else None
        
        if is_empty or (max_score is not None and max_score < 0.0):
            feedback = (
                "\n\n⚠️ [CRAG Verification: Low Local Document Relevance]\n"
                "The local knowledge base does not contain strong matches for this query. "
                "Guidance: Acknowledge the absence of local data or use `web_search` to verify on the internet."
            )
        else:
            feedback = f"\n\n✅ [CRAG Verification: High Confidence Match (Score: {max_score:.3f})]"
            
        updated_msg = ToolMessage(
            content=content + feedback,
            name=last_msg.name,
            tool_call_id=last_msg.tool_call_id
        )
        return {"messages": [updated_msg]}

    # 2. Reflexion Code Auto-Debugger
    elif tool_name in ["execute_python_code", "execute_terminal_command", "write_local_file"]:
        error_indicators = [
            "Traceback (most recent call last)",
            "SyntaxError",
            "NameError",
            "TypeError",
            "IndexError",
            "ImportError",
            "ModuleNotFoundError",
            "AttributeError",
            "ZeroDivisionError",
            "command not found",
            "Errors:"
        ]
        
        has_error = any(err in content for err in error_indicators)
        exit_code_match = re.search(r"Exit code\s*([1-9]\d*)", content)
        if exit_code_match:
            has_error = True
            
        if has_error:
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            error_line = lines[-1] if lines else "Unknown runtime error"
            
            reflexion_directive = (
                f"\n\n⚠️ [Reflexion Auto-Debugger: Execution Failure Detected]\n"
                f"• Diagnostic Output: `{error_line[:120]}`\n"
                f"• Self-Correction Directive:\n"
                f"  1. Analyze the root cause of this failure.\n"
                f"  2. Use `view_code_slice` or `read_local_file` to inspect the source if helpful.\n"
                f"  3. Write a corrected script using `write_local_file` or run the fixed command.\n"
                f"  4. Re-execute to confirm the code runs cleanly with exit code 0."
            )
            updated_msg = ToolMessage(
                content=content + reflexion_directive,
                name=last_msg.name,
                tool_call_id=last_msg.tool_call_id
            )
            return {"messages": [updated_msg]}
        else:
            success_tag = "\n\n✅ [Reflexion Auto-Debugger: Execution verified with zero errors]"
            updated_msg = ToolMessage(
                content=content + success_tag,
                name=last_msg.name,
                tool_call_id=last_msg.tool_call_id
            )
            return {"messages": [updated_msg]}
            
    return {"messages": []}


# 5. Define Graph Edges
def should_continue(state: AgentState):
    """Determine whether to use a tool or return to the user."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, go to the tool node
    if last_message.tool_calls:
        return "continue"
    # Otherwise, we are finished
    return "end"

# 6. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("action", tool_node)
workflow.add_node("grade_retrieval", grade_retrieval_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END
    }
)

workflow.add_edge("action", "grade_retrieval")
workflow.add_edge("grade_retrieval", "agent")

# Compile the graph
app = workflow.compile()

# Helper function to run the agent
def run_agent(query: str):
    inputs = {"messages": [HumanMessage(content=query)]}
    for event in app.stream(inputs, stream_mode="values"):
        last_message = event["messages"][-1]
    return last_message.content

