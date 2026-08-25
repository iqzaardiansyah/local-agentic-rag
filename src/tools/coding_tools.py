import os
import subprocess
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

repl = PythonREPL()

@tool
def execute_terminal_command(command: str) -> str:
    """
    Execute a generic terminal/shell command. 
    Use this to run non-Python code (e.g., `node script.js`, `gcc -o app app.c`), run bash scripts, or manage the OS.
    WARNING: Only run safe commands.
    """
    try:
        # Run command and capture output
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return f"Command executed (Exit code {result.returncode}).\nOutput:\n{output}"
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
    Provide the absolute or relative path to the file.
    """
    try:
        if not os.path.exists(file_path):
            return f"Error: File '{file_path}' does not exist."
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_local_file(file_path: str, content: str) -> str:
    """
    Write content to a file on the local filesystem.
    This allows you to act as a coding assistant and create or modify files for the user.
    """
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to '{file_path}'."
    except Exception as e:
        return f"Error writing to file: {str(e)}"
