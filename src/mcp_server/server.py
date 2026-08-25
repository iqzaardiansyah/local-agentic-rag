from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import os

app = FastAPI(title="Local Data MCP Server", description="A Model Context Protocol server exposing local data.")

DB_PATH = os.path.join(os.path.dirname(__file__), "mock_data.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY,
            name TEXT,
            department TEXT,
            salary INTEGER
        )
    ''')
    # Insert mock data if empty
    cursor.execute("SELECT count(*) FROM employees")
    if cursor.fetchone()[0] == 0:
        employees = [
            ("Alice Smith", "Engineering", 120000),
            ("Bob Jones", "Marketing", 85000),
            ("Charlie Brown", "HR", 70000)
        ]
        cursor.executemany("INSERT INTO employees (name, department, salary) VALUES (?, ?, ?)", employees)
        conn.commit()
    conn.close()

init_db()

class QueryRequest(BaseModel):
    department: str = None

@app.get("/api/employees")
def get_employees(department: str = None):
    """Retrieve employees, optionally filtered by department."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if department:
        cursor.execute("SELECT name, department, salary FROM employees WHERE department LIKE ?", (f"%{department}%",))
    else:
        cursor.execute("SELECT name, department, salary FROM employees")
        
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return {"results": "No employees found."}
    
    results = [{"name": r[0], "department": r[1], "salary": r[2]} for r in rows]
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    print("Starting Mock MCP Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
