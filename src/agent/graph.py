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
from src.tools.mcp_tool import query_employee_database
from src.tools.coding_tools import execute_python_code, read_local_file, write_local_file, execute_terminal_command
from src.tools.web_scraper import read_webpage

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
    web_search,
    read_webpage,
    execute_terminal_command,
    execute_python_code,
    read_local_file,
    write_local_file
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

SYSTEM_PROMPT = """You are OmniLocal-Agent, an advanced, locally-hosted AI assistant.
Your goal is to help the user by utilizing the tools at your disposal and thinking through problems deeply and step-by-step.
Before choosing an action or generating the final response, thoroughly think, analyze multiple hypotheses, and verify facts.

Tool Usage Guide:
- Use `search_local_documents` for general knowledge or checking user files.
- Use `query_employee_database` to look up staff, departments, and salaries.
- Use `web_search` to find up-to-date information, news, or general facts from the internet.
- Use `read_webpage` if a search result URL looks highly relevant and you need to read its full content.
- Use `execute_python_code` (like ChatGPT Advanced Data Analysis) to perform calculations, data analysis, or test logic locally. ALWAYS print() your results inside the code so you can read them.
- Use `execute_terminal_command` (like Claude Code/Antigravity) to execute bash/shell commands, run Node.js/C++/Go code, run tests, or manage the operating system.
- Use `read_local_file` and `write_local_file` to act as a coding assistant and modify the user's workspace directly when they ask you to write code or read their files.

Always answer accurately based on the information returned by the tools.
If you don't know the answer even after searching, say you don't know.
"""

# 4. Define Graph Nodes
def agent_node(state: AgentState):
    messages = state["messages"]
    
    # Ensure system prompt is present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def grade_retrieval_node(state: AgentState):
    """
    Self-RAG / Corrective RAG (CRAG) Grader Node.
    Evaluates retrieved local documents for relevance quality based on Cross-Encoder scores.
    If local retrieval is insufficient or negative, injects corrective guidance to prevent hallucinations.
    """
    messages = state["messages"]
    if not messages:
        return {"messages": []}
        
    last_msg = messages[-1]
    
    # Check if the last action was from search_local_documents
    if isinstance(last_msg, ToolMessage) and getattr(last_msg, "name", "") == "search_local_documents":
        content = last_msg.content or ""
        
        # Check for error or empty conditions
        is_empty = "No relevant information found" in content or "Error accessing" in content
        
        # Extract Cross-Encoder relevance scores
        scores = re.findall(r"Relevance Score:\s*(-?\d+\.?\d*)", content)
        max_score = max([float(s) for s in scores]) if scores else None
        
        # If no results or all scores are negative, trigger corrective feedback
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

