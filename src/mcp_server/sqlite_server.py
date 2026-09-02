import json
import os
import sqlite3
import sys
from typing import Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "mock_data.db")

def init_database():
    """Seed comprehensive mock tables for employees, departments, and projects."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if old schema exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='employees'")
    if cursor.fetchone():
        cursor.execute("PRAGMA table_info(employees)")
        cols = [c[1] for c in cursor.fetchall()]
        if "title" not in cols:
            cursor.execute("DROP TABLE IF EXISTS employees")
            cursor.execute("DROP TABLE IF EXISTS departments")
            cursor.execute("DROP TABLE IF EXISTS projects")
            
    # 1. Employees Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            title TEXT NOT NULL,
            salary INTEGER NOT NULL,
            hire_date TEXT NOT NULL
        )
    """)
    
    # 2. Departments Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dept_name TEXT UNIQUE NOT NULL,
            budget INTEGER NOT NULL,
            head_of_dept TEXT NOT NULL
        )
    """)
    
    # 3. Projects Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            lead_dept TEXT NOT NULL,
            budget_allocated INTEGER NOT NULL,
            status TEXT NOT NULL
        )
    """)
    
    # Seed data if empty
    cursor.execute("SELECT count(*) FROM employees")
    if cursor.fetchone()[0] == 0:
        employees_data = [
            ("Alice Smith", "Engineering", "Principal AI Architect", 145000, "2022-03-15"),
            ("Bob Jones", "Engineering", "Senior DevOps Engineer", 120000, "2023-01-10"),
            ("Charlie Brown", "Product", "Lead Product Manager", 115000, "2021-08-01"),
            ("Diana Prince", "Marketing", "Growth Marketing Director", 95000, "2023-06-20"),
            ("Evan Wright", "HR", "People Operations Lead", 75000, "2022-11-05"),
            ("Fiona Gallagher", "Engineering", "ML Systems Engineer", 130000, "2024-02-01")
        ]
        cursor.executemany("INSERT INTO employees (name, department, title, salary, hire_date) VALUES (?, ?, ?, ?, ?)", employees_data)
        
        dept_data = [
            ("Engineering", 1500000, "Alice Smith"),
            ("Product", 600000, "Charlie Brown"),
            ("Marketing", 400000, "Diana Prince"),
            ("HR", 200000, "Evan Wright")
        ]
        cursor.executemany("INSERT INTO departments (dept_name, budget, head_of_dept) VALUES (?, ?, ?)", dept_data)
        
        projects_data = [
            ("Local AI Agent Hub", "Engineering", 350000, "Active"),
            ("Enterprise Semantic Search", "Engineering", 200000, "Completed"),
            ("Q3 Global Marketing Push", "Marketing", 120000, "Active"),
            ("HR Automation Suite", "HR", 50000, "Planning")
        ]
        cursor.executemany("INSERT INTO projects (project_name, lead_dept, budget_allocated, status) VALUES (?, ?, ?, ?)", projects_data)
        
        conn.commit()
    conn.close()


init_database()

# --- Core Introspection Engine ---

def list_tables() -> List[Dict[str, Any]]:
    """Returns all tables in the SQLite database with their live row count."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]
    
    table_info = []
    for t in tables:
        cursor.execute(f"SELECT count(*) FROM {t}")
        count = cursor.fetchone()[0]
        table_info.append({"table_name": t, "row_count": count})
    conn.close()
    return table_info

def describe_table(table_name: str) -> Dict[str, Any]:
    """Returns column names, types, primary keys, and nullable flags for a table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        if not columns:
            return {"error": f"Table '{table_name}' does not exist."}
            
        col_list = []
        for col in columns:
            col_list.append({
                "cid": col[0],
                "name": col[1],
                "type": col[2],
                "notnull": bool(col[3]),
                "default_value": col[4],
                "is_primary_key": bool(col[5])
            })
            
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        fk_list = [{"id": f[0], "table": f[2], "from": f[3], "to": f[4]} for f in cursor.fetchall()]
        
        return {
            "table_name": table_name,
            "columns": col_list,
            "foreign_keys": fk_list
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()

def execute_read_query(query: str, max_rows: int = 100) -> Dict[str, Any]:
    """
    Safely executes a read-only SQL query against the SQLite database.
    Rejects any non-SELECT statements.
    """
    cleaned = query.strip()
    # Guard against destructive keywords
    disallowed_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REPLACE"]
    first_word = cleaned.split()[0].upper() if cleaned else ""
    
    if first_word != "SELECT" and first_word != "WITH":
        return {
            "success": False,
            "error": f"Security Error: Only SELECT queries are permitted (received '{first_word}')."
        }
        
    for kw in disallowed_keywords:
        if f" {kw} " in f" {cleaned.upper()} ":
            return {
                "success": False,
                "error": f"Security Error: Disallowed keyword '{kw}' detected in query."
            }
            
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(cleaned)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows)
        results = [dict(zip(columns, row)) for row in rows]
        
        return {
            "success": True,
            "columns": columns,
            "row_count": len(results),
            "rows": results
        }
    except Exception as e:
        return {"success": False, "error": f"SQL Execution Error: {str(e)}"}
    finally:
        conn.close()

# --- Model Context Protocol (MCP) JSON-RPC 2.0 Handler ---

MCP_TOOLS_MANIFEST = [
    {
        "name": "list_tables",
        "description": "List all tables available in the SQLite database along with total row counts.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "describe_table",
        "description": "Inspect the schema of a specific table, including columns, data types, and primary keys.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "table_name": {"type": "string", "description": "The name of the table to describe."}
            },
            "required": ["table_name"]
        }
    },
    {
        "name": "execute_read_query",
        "description": "Execute a read-only SELECT SQL query to query tables, aggregate data, or filter records.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The SQL SELECT statement to execute."},
                "max_rows": {"type": "integer", "description": "Maximum number of rows to return (default 100).", "default": 100}
            },
            "required": ["query"]
        }
    }
]

def handle_mcp_request(request_json: Dict[str, Any]) -> Dict[str, Any]:
    """Processes JSON-RPC 2.0 requests per Model Context Protocol specification."""
    req_id = request_json.get("id")
    method = request_json.get("method")
    params = request_json.get("params", {})
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "sqlite-introspection-mcp-server",
                    "version": "1.0.0"
                },
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False}
                }
            }
        }
        
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": MCP_TOOLS_MANIFEST
            }
        }
        
    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "list_tables":
            data = list_tables()
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(data, indent=2)}]
                }
            }
        elif tool_name == "describe_table":
            table_name = arguments.get("table_name", "")
            data = describe_table(table_name)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(data, indent=2)}]
                }
            }
        elif tool_name == "execute_read_query":
            sql_query = arguments.get("query", "")
            max_rows = arguments.get("max_rows", 100)
            data = execute_read_query(sql_query, max_rows)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(data, indent=2)}]
                }
            }
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Tool '{tool_name}' not found."}
            }
            
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not recognized."}
        }

if __name__ == "__main__":
    # Stdio transport loop for MCP standard
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = handle_mcp_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_res = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err_res) + "\n")
            sys.stdout.flush()
