import json
from langchain_core.tools import tool
from src.mcp_server.sqlite_server import handle_mcp_request

def call_mcp_tool(tool_name: str, arguments: dict = None) -> dict:
    """Helper to dispatch tool calls via MCP JSON-RPC 2.0 protocol."""
    request = {
        "jsonrpc": "2.0",
        "id": "agent-call-1",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments or {}
        }
    }
    response = handle_mcp_request(request)
    if "error" in response:
        return {"error": response["error"].get("message", "MCP Tool Error")}
        
    try:
        content_items = response.get("result", {}).get("content", [])
        if content_items and content_items[0].get("type") == "text":
            return json.loads(content_items[0].get("text", "{}"))
    except Exception:
        pass
    return response.get("result", {})

@tool
def mcp_list_tables() -> str:
    """
    Introspect the internal SQLite database via MCP.
    Returns a list of all existing database tables and their total record counts.
    Use this first before querying to see what data is stored in the database.
    """
    try:
        data = call_mcp_tool("list_tables")
        if isinstance(data, list):
            formatted = ["📊 **Database Tables:**"]
            for item in data:
                formatted.append(f"- `{item['table_name']}` ({item['row_count']} records)")
            return "\n".join(formatted)
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"MCP Error listing tables: {str(e)}"

@tool
def mcp_describe_table(table_name: str) -> str:
    """
    Inspect the schema and column definitions of a specific SQLite database table via MCP.
    Returns column names, types (INTEGER, TEXT, etc.), primary keys, and foreign keys.
    Args:
        table_name: The table name to describe (e.g. 'employees', 'departments', 'projects').
    """
    try:
        data = call_mcp_tool("describe_table", {"table_name": table_name})
        if "error" in data:
            return f"Error: {data['error']}"
            
        columns = data.get("columns", [])
        formatted = [f"📋 **Schema for Table `{table_name}`:**"]
        for c in columns:
            pk = " (PRIMARY KEY)" if c.get("is_primary_key") else ""
            req = " [NOT NULL]" if c.get("notnull") else ""
            formatted.append(f"- **{c['name']}**: `{c['type']}`{pk}{req}")
            
        fks = data.get("foreign_keys", [])
        if fks:
            formatted.append("\n🔗 **Foreign Keys:**")
            for fk in fks:
                formatted.append(f"- `{fk['from']}` -> `{fk['table']}.{fk['to']}`")
                
        return "\n".join(formatted)
    except Exception as e:
        return f"MCP Error describing table: {str(e)}"

@tool
def mcp_execute_query(query: str) -> str:
    """
    Execute a read-only SQL query against the SQLite database via MCP.
    Only SELECT and WITH statements are allowed.
    Returns results in a clear markdown table.
    Args:
        query: The SQL query to execute (e.g. 'SELECT name, salary FROM employees WHERE salary > 100000').
    """
    try:
        data = call_mcp_tool("execute_read_query", {"query": query, "max_rows": 50})
        if not data.get("success"):
            return f"SQL Error: {data.get('error', 'Unknown query execution error')}"
            
        rows = data.get("rows", [])
        columns = data.get("columns", [])
        if not rows:
            return "Query executed successfully. 0 rows returned."
            
        # Format as Markdown table
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        data_rows = []
        for r in rows:
            data_rows.append("| " + " | ".join([str(r.get(c, "")) for c in columns]) + " |")
            
        return f"**Results ({len(rows)} rows):**\n" + "\n".join([header, separator] + data_rows)
    except Exception as e:
        return f"MCP Error executing query: {str(e)}"

@tool
def query_employee_database(department: str = None) -> str:
    """
    Query the internal company employee database (via SQLite MCP Server).
    Use this to find employee names, departments, titles, and salaries.
    You can specify a department to filter, or leave empty for all employees.
    """
    if department and department.strip():
        sql = f"SELECT name, department, title, salary, hire_date FROM employees WHERE department LIKE '%{department.strip()}%'"
    else:
        sql = "SELECT name, department, title, salary, hire_date FROM employees"
    return mcp_execute_query.invoke({"query": sql})
