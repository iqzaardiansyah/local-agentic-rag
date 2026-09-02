import os
import shutil
import subprocess
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# Dedicated workspace directory to contain all agent-generated artifacts
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKSPACE_DIR = os.path.join(ROOT_DIR, "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

repl = PythonREPL()

def get_safe_path(file_path: str) -> str:
    """Resolve relative paths to the sandboxed workspace directory."""
    if not os.path.isabs(file_path):
        target = os.path.abspath(os.path.join(WORKSPACE_DIR, file_path))
    else:
        target = os.path.abspath(file_path)
    return target

def list_workspace_files():
    """List all artifacts currently stored in the workspace directory."""
    if not os.path.exists(WORKSPACE_DIR):
        return []
    files = []
    for root, _, filenames in os.walk(WORKSPACE_DIR):
        for f in filenames:
            rel = os.path.relpath(os.path.join(root, f), WORKSPACE_DIR)
            files.append(rel)
    return files

def clean_workspace():
    """Wipe all agent-generated artifacts inside the workspace directory."""
    if os.path.exists(WORKSPACE_DIR):
        for item in os.listdir(WORKSPACE_DIR):
            item_path = os.path.join(WORKSPACE_DIR, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

@tool
def execute_terminal_command(command: str) -> str:
    """
    Execute a generic terminal/shell command inside the sandboxed workspace directory.
    Use this to run non-Python code (e.g., `node script.js`, `gcc -o app app.c`), run bash scripts, or manage files.
    WARNING: Only run safe commands.
    """
    try:
        # Run command inside WORKSPACE_DIR to contain all created files
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30, cwd=WORKSPACE_DIR)
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return f"Command executed in ./workspace (Exit code {result.returncode}).\nOutput:\n{output}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code in a local REPL environment.
    Use this to perform complex math calculations, data analysis, or logic testing.
    The code runs in a stateful environment. Print your final results so you can see them.
    WARNING: Only run safe code.
    """
    try:
        result = repl.run(code)
        return f"Code executed successfully.\nOutput:\n{result}"
    except Exception as e:
        return f"Error executing Python code: {str(e)}"

@tool
def read_local_file(file_path: str) -> str:
    """
    Read the contents of a file on the local filesystem.
    Relative paths will be searched inside ./workspace or the project root.
    """
    try:
        resolved_path = get_safe_path(file_path)
        # If not in workspace, check root directory
        if not os.path.exists(resolved_path):
            alt_path = os.path.abspath(os.path.join(ROOT_DIR, file_path))
            if os.path.exists(alt_path):
                resolved_path = alt_path
            else:
                return f"Error: File '{file_path}' does not exist in workspace or root."
                
        with open(resolved_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_local_file(file_path: str, content: str) -> str:
    """
    Write content to a file inside the sandboxed workspace directory (./workspace).
    This allows you to act as a coding assistant and generate files for the user safely.
    """
    try:
        resolved_path = get_safe_path(file_path)
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        with open(resolved_path, 'w', encoding='utf-8') as f:
            f.write(content)
        rel_display = os.path.relpath(resolved_path, ROOT_DIR)
        return f"Successfully wrote to '{rel_display}' (contained in sandbox)."
    except Exception as e:
        return f"Error writing to file: {str(e)}"

