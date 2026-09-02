import os
import concurrent.futures
from typing import List, Dict, Any, Annotated, Sequence, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Import toolkits
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
from src.tools.graph_tool import query_knowledge_graph
from duckduckgo_search import DDGS

@tool
def subagent_web_search(query: str) -> str:
    """Search the web for up-to-date facts and articles."""
    try:
        results = DDGS().text(query, max_results=4)
        if not results:
            return "No web results found."
        return "\n\n".join([f"Title: {r.get('title')}\nSnippet: {r.get('body')}" for r in results])
    except Exception as e:
        return f"Web search error: {e}"

# Tool profiles per subagent role
ROLE_TOOLS = {
    "researcher": [query_knowledge_graph, subagent_web_search, read_webpage, search_local_documents],
    "coder": [execute_python_code, execute_terminal_command, read_local_file, write_local_file, grep_search, view_code_slice, list_directory_tree, find_files_by_pattern],
    "data_analyst": [query_knowledge_graph, mcp_list_tables, mcp_describe_table, mcp_execute_query, query_employee_database, execute_python_code, read_local_file, search_local_documents],
    "rag_specialist": [query_knowledge_graph, search_local_documents, read_local_file, subagent_web_search],
    "custom": [query_knowledge_graph, mcp_list_tables, mcp_describe_table, mcp_execute_query, search_local_documents, query_employee_database, subagent_web_search, read_webpage, execute_python_code, execute_terminal_command, read_local_file, write_local_file]
}



ROLE_PROMPTS = {
    "researcher": "You are a specialized Research Subagent. Your mission is to gather accurate facts and summarize findings using search and scraping tools. Be factual and concise.",
    "coder": "You are a specialized Coding & Software Engineering Subagent. Your mission is to write, inspect, and test code strictly inside the ./workspace sandbox.",
    "data_analyst": "You are a specialized Data Analyst Subagent. Your mission is to query databases, analyze tabular data, perform math in Python, and produce clear insights.",
    "rag_specialist": "You are a specialized Knowledge Base Subagent. Your mission is to search internal documents, verify retrieved facts, and extract relevant knowledge.",
    "custom": "You are an autonomous Task Subagent. Focus on solving the specific objective given to you using the available tools."
}

class SubagentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_subagent_runner(role: str):
    """Factory to build an isolated single-loop subagent execution graph."""
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_model = os.getenv("LLM_MODEL", "qwen3.8:27b")
    llm_api_key = os.getenv("LLM_API_KEY", "ollama")
    
    subagent_llm = ChatOpenAI(
        model=llm_model,
        base_url=llm_base_url,
        api_key=llm_api_key,
        temperature=0.4,
        extra_body={
            "reasoning_effort": "high",
            "thinking": {"type": "enabled", "budget_tokens": 8192}
        }
    )
    
    tools = ROLE_TOOLS.get(role, ROLE_TOOLS["custom"])
    bound_llm = subagent_llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    system_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["custom"])
    
    def subagent_node(state: SubagentState):
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        response = bound_llm.invoke(messages)
        return {"messages": [response]}
        
    def should_continue(state: SubagentState):
        last_message = state["messages"][-1]
        return "continue" if last_message.tool_calls else "end"
        
    builder = StateGraph(SubagentState)
    builder.add_node("agent", subagent_node)
    builder.add_node("action", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"continue": "action", "end": END})
    builder.add_edge("action", "agent")
    
    return builder.compile()


def execute_single_subagent(subtask: Dict[str, str]) -> Dict[str, Any]:
    """Execute a single subagent task to completion."""
    role = subtask.get("role", "custom").lower()
    task = subtask.get("task", "")
    subagent_name = subtask.get("name") or f"{role.capitalize()}-Agent"
    
    try:
        runner = create_subagent_runner(role)
        inputs = {"messages": [HumanMessage(content=f"Subtask Objective: {task}")]}
        result = runner.invoke(inputs)
        
        last_msg = result["messages"][-1]
        output_text = getattr(last_msg, "content", "Task completed.")
        return {
            "name": subagent_name,
            "role": role,
            "task": task,
            "status": "success",
            "result": output_text
        }
    except Exception as e:
        return {
            "name": subagent_name,
            "role": role,
            "task": task,
            "status": "error",
            "result": f"Subagent error: {str(e)}"
        }

def run_subagents_parallel(subtasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """
    Spawns multiple subagents in parallel using ThreadPoolExecutor.
    Leverages OLLAMA_NUM_PARALLEL=4 on Ollama server for concurrent multi-slot inference.
    """
    if not subtasks:
        return []
        
    # Cap parallel execution at 4 workers to match server capacity
    worker_count = min(len(subtasks), max_workers)
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_task = {executor.submit(execute_single_subagent, st): st for st in subtasks}
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                res = future.result()
                results.append(res)
            except Exception as exc:
                st = future_to_task[future]
                results.append({
                    "name": st.get("name", "Subagent"),
                    "role": st.get("role", "custom"),
                    "task": st.get("task", ""),
                    "status": "error",
                    "result": f"Execution exception: {exc}"
                })
                
    return results

@tool
def spawn_parallel_subagents(subtasks: List[Dict[str, str]]) -> str:
    """
    Spawn up to 4 specialized subagents to execute independent subtasks concurrently in parallel.
    Takes full advantage of multi-slot GPU inference (OLLAMA_NUM_PARALLEL=4).
    
    Args:
        subtasks: A list of dicts with keys:
            - 'role': One of ['researcher', 'coder', 'data_analyst', 'rag_specialist', 'custom']
            - 'task': Detailed prompt describing what this subagent should investigate or do.
            - 'name': (Optional) A descriptive label for this subagent.
            
    Example:
        spawn_parallel_subagents(subtasks=[
            {"role": "researcher", "task": "Search for latest developments on LangGraph v0.2"},
            {"role": "data_analyst", "task": "Query employee database for Engineering department staff"},
            {"role": "rag_specialist", "task": "Search local documents for project requirements"}
        ])
    """
    if not subtasks:
        return "Error: No subtasks provided."
        
    # Run in parallel
    results = run_subagents_parallel(subtasks, max_workers=4)
    
    formatted = [f"### ⚡ Parallel Subagent Execution Report ({len(results)} Subagents Finished)"]
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        formatted.append(
            f"\n#### {status_icon} [{r['name']} | Role: {r['role']}]\n"
            f"**Objective:** {r['task']}\n"
            f"**Findings / Result:**\n{r['result']}\n"
        )
        
    return "\n---\n".join(formatted)
