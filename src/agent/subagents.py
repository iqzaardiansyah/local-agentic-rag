import os
import time
import concurrent.futures
from typing import List, Dict, Any, Annotated, Optional, Sequence, TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agent.subagent_events import publish_subagent_event

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
    """Search the web for up-to-date facts and articles (with local result cache)."""
    try:
        from src.tools.web_cache import (
            get_cached_search,
            put_cached_search,
            format_results,
        )

        cached = get_cached_search(query)
        if cached is not None:
            return format_results(cached)
        results = DDGS().text(query, max_results=4)
        if not results:
            return "No web results found."
        results = list(results)
        put_cached_search(query, results)
        return format_results(results)
    except Exception as e:
        return f"Web search error: {e}"

ROLE_TOOLS = {
    "researcher": [query_knowledge_graph, subagent_web_search, read_webpage, search_local_documents],
    "coder": [execute_python_code, execute_terminal_command, read_local_file, write_local_file, grep_search, view_code_slice, list_directory_tree, find_files_by_pattern],
    "data_analyst": [query_knowledge_graph, mcp_list_tables, mcp_describe_table, mcp_execute_query, query_employee_database, execute_python_code, read_local_file, search_local_documents],
    "rag_specialist": [query_knowledge_graph, search_local_documents, read_local_file, subagent_web_search],
    "custom": [query_knowledge_graph, mcp_list_tables, mcp_describe_table, mcp_execute_query, search_local_documents, query_employee_database, subagent_web_search, read_webpage, execute_python_code, execute_terminal_command, read_local_file, write_local_file]
}



ROLE_PROMPTS = {
    "researcher": (
        "You are a Research Subagent. Gather facts with search/scrape tools "
        "and return a short structured brief."
    ),
    "coder": (
        "You are a Coding Subagent working only in the project sandbox.\n"
        "- Write files with write_local_file using sandbox-relative paths "
        "(e.g. `todo_app/app.py`) — never `workspace/todo_app/app.py`.\n"
        "- Work in phases; run a quick check (py_compile/pytest/python script) after writing.\n"
        "- Finish with a brief summary of files and how to run them."
    ),
    "data_analyst": (
        "You are a Data Analyst Subagent. Query data, compute stats in Python, "
        "and return compact insights."
    ),
    "rag_specialist": (
        "You are a Knowledge Base Subagent. Search internal docs and return a tight summary."
    ),
    "custom": (
        "You are a Task Subagent. Solve only the assigned objective. "
        "Prefer tools over long prose; if coding, write files incrementally and test."
    ),
}

class SubagentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_subagent_runner(role: str):
    """Factory to build an isolated single-loop subagent execution graph."""
    from src.agent.llm_config import load_llm_settings, extra_body_from_settings, install_k2_openai_patch

    install_k2_openai_patch()
    cfg = load_llm_settings()
    subagent_llm = ChatOpenAI(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        temperature=cfg["subagent_temperature"],
        streaming=True,
        max_tokens=cfg["subagent_max_tokens"],
        timeout=cfg.get("timeout_seconds", 600.0),
        extra_body=extra_body_from_settings(cfg),
    )
    
    tools = ROLE_TOOLS.get(role, ROLE_TOOLS["custom"])
    bound_llm = subagent_llm.bind_tools(tools)
    tool_node = ToolNode(tools)
    system_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["custom"])
    
    def subagent_node(state: SubagentState):
        from src.agent.llm_config import sanitize_messages_for_family

        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        has_user = any(isinstance(m, HumanMessage) and bool(str(m.content).strip()) for m in messages)
        if not has_user:
            messages.append(HumanMessage(content="Please execute the assigned task."))

        messages = sanitize_messages_for_family(messages)
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


def _episodic_brief_for_task(task: str, top_k: int = 3) -> str:
    """
    Pull top episodic memories related to the subtask.
    Returns '' when nothing relevant (or store unavailable) so the agent
    is never blocked by memory failures.
    """
    if not task or len(task.strip()) < 4:
        return ""
    try:
        from src.memory.episodic_memory import recall_memories

        mems = recall_memories(task, top_k=top_k)
    except Exception:
        return ""
    if not mems:
        return ""
    lines = ["Long-term user/project memories relevant to this subtask:"]
    for m in mems:
        cat = m.get("category", "general")
        fact = m.get("fact", "")
        if fact:
            lines.append(f"- [{cat}] {fact}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def execute_single_subagent(subtask: Dict[str, str], index: int = 0, total: int = 1) -> Dict[str, Any]:
    """Execute a single subagent task to completion and publish progress events."""
    role = subtask.get("role", "custom").lower()
    task = (subtask.get("task") or "").strip()
    if not task:
        task = f"Execute subagent objective for role '{role}'."
    subagent_name = subtask.get("name") or f"{role.capitalize()}-Agent"

    publish_subagent_event({
        "type": "subagent_start",
        "name": subagent_name,
        "role": role,
        "task": task[:300],
        "index": index,
        "total": total,
    })
    started = time.perf_counter()

    try:
        runner = create_subagent_runner(role)
        objective = f"Subtask Objective: {task}"
        brief = _episodic_brief_for_task(task)
        if brief:
            objective += f"\n\n{brief}\n\nPrefer user-stated preferences above when they apply."
        inputs = {"messages": [HumanMessage(content=objective)]}
        result = runner.invoke(inputs)

        last_msg = result["messages"][-1]
        output_text = getattr(last_msg, "content", "Task completed.")
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        publish_subagent_event({
            "type": "subagent_done",
            "name": subagent_name,
            "role": role,
            "status": "success",
            "result_preview": str(output_text)[:240],
            "elapsed_ms": elapsed_ms,
            "index": index,
            "total": total,
        })
        return {
            "name": subagent_name,
            "role": role,
            "task": task,
            "status": "success",
            "result": output_text,
            "elapsed_ms": elapsed_ms,
            "used_memory": bool(brief),
        }
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        publish_subagent_event({
            "type": "subagent_done",
            "name": subagent_name,
            "role": role,
            "status": "error",
            "result_preview": str(e)[:240],
            "elapsed_ms": elapsed_ms,
            "index": index,
            "total": total,
        })
        return {
            "name": subagent_name,
            "role": role,
            "task": task,
            "status": "error",
            "result": f"Subagent error: {str(e)}",
            "elapsed_ms": elapsed_ms,
        }


def run_subagents_parallel(subtasks: List[Dict[str, str]], max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run subagents in parallel (capped by LLM_MAX_PARALLEL / server capacity).
    """
    from src.agent.llm_config import get_max_parallel

    if not subtasks:
        return []

    limit = max_workers if max_workers is not None else get_max_parallel()
    limit = max(1, min(limit, len(subtasks)))
    total = len(subtasks)
    results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=limit) as executor:
        future_to_task = {
            executor.submit(execute_single_subagent, st, idx, total): (st, idx)
            for idx, st in enumerate(subtasks)
        }
        for future in concurrent.futures.as_completed(future_to_task):
            st, _idx = future_to_task[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as exc:
                results.append({
                    "name": st.get("name", "Subagent"),
                    "role": st.get("role", "custom"),
                    "task": st.get("task", ""),
                    "status": "error",
                    "result": f"Execution exception: {exc}",
                })

    success = sum(1 for r in results if r.get("status") == "success")
    publish_subagent_event({
        "type": "subagents_all_done",
        "count": len(results),
        "success": success,
        "error": len(results) - success,
    })
    return results

@tool
def spawn_parallel_subagents(subtasks: List[Dict[str, str]]) -> str:
    """
    Run specialized subagents in parallel (researcher, coder, data_analyst,
    rag_specialist, custom). Parallelism is capped by server capacity
    (LLM_MAX_PARALLEL, default 4). Keep each subtask small and file-oriented.

    Example:
        spawn_parallel_subagents(subtasks=[
            {"role": "researcher", "task": "List requirements and file layout for a Flask todo API"},
            {"role": "coder", "task": "Create todo_app/app.py skeleton only"},
            {"role": "coder", "task": "Create todo_app/tests/test_app.py with 3 pytest cases"}
        ])
    """
    from src.agent.llm_config import get_max_parallel

    if not subtasks:
        return "Error: No subtasks provided."

    limit = get_max_parallel()
    if len(subtasks) > limit:
        subtasks = subtasks[:limit]

    results = run_subagents_parallel(subtasks, max_workers=limit)

    formatted = [f"### ⚡ Parallel Subagent Execution Report ({len(results)} Subagents Finished)"]
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        formatted.append(
            f"\n#### {status_icon} [{r['name']} | Role: {r['role']}]\n"
            f"**Objective:** {r['task']}\n"
            f"**Findings / Result:**\n{r['result']}\n"
        )

    return "\n---\n".join(formatted)
