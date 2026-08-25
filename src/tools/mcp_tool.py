import httpx
from langchain_core.tools import tool

@tool
def query_employee_database(department: str = None) -> str:
    """
    Query the internal company employee database (via MCP Server).
    Use this to find employee names, departments, and salaries.
    You can specify a department to filter, or leave empty for all employees.
    """
    mcp_url = "http://localhost:8000/api/employees"
    params = {}
    if department and department.strip():
        params["department"] = department.strip()
        
    try:
        response = httpx.get(mcp_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("results", [])
        if isinstance(results, str):
            return results # "No employees found."
            
        formatted = "\n".join([f"- Name: {r['name']}, Dept: {r['department']}, Salary: ${r['salary']}" for r in results])
        return f"Employee Data:\n{formatted}"
    except httpx.RequestError as e:
        return f"Error contacting the MCP database server. Ensure it is running on port 8000. Detail: {e}"
