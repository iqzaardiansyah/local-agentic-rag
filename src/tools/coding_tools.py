import fnmatch
import os
import re
import shutil
import subprocess
from typing import List
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# Dedicated workspace directory to contain all agent-generated artifacts
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKSPACE_DIR = os.path.join(ROOT_DIR, "workspace")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

repl = PythonREPL()

def get_safe_workspace_path(file_path: str) -> str:
    """Resolve paths strictly inside the sandboxed workspace directory."""
    if not os.path.isabs(file_path):
        target = os.path.abspath(os.path.join(WORKSPACE_DIR, file_path))
    else:
        target = os.path.abspath(file_path)
        
    # Prevent directory traversal attacks out of root or workspace
    if not (target.startswith(WORKSPACE_DIR) or target.startswith(ROOT_DIR)):
        raise PermissionError("Access denied: Target path is outside project root.")
    return target

def resolve_readable_path(file_path: str) -> str:
    """Resolve path for reading, checking project root and workspace directory."""
    if not file_path or file_path == ".":
        return ROOT_DIR
    if os.path.isabs(file_path):
        return file_path
    # Check project root first for exploration tools
    root_path = os.path.abspath(os.path.join(ROOT_DIR, file_path))
    if os.path.exists(root_path):
        return root_path
    # Check workspace directory
    ws_path = os.path.abspath(os.path.join(WORKSPACE_DIR, file_path))
    if os.path.exists(ws_path):
        return ws_path
    return root_path


def list_workspace_files() -> List[str]:
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
def list_directory_tree(path: str = ".", max_depth: int = 2) -> str:
    """
    Explore the file and directory hierarchy of the workspace or project.
    Generates a visual directory tree structure.
    Args:
        path: Directory path to inspect (defaults to current project).
        max_depth: Maximum recursion depth (defaults to 2).
    """
    target_dir = resolve_readable_path(path)
    if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
        return f"Error: Directory '{path}' not found."
        
    ignored_patterns = {".git", "__pycache__", "venv", ".pytest_cache", "node_modules", "chroma_db", ".ipynb_checkpoints"}
    
    tree_lines = [f"📁 {os.path.basename(target_dir) or 'root'}/"]
    
    def _build_tree(curr_dir: str, prefix: str = "", current_depth: int = 0):
        if current_depth >= max_depth:
            return
            
        try:
            entries = sorted(os.listdir(curr_dir))
        except Exception:
            return
            
        filtered = [e for e in entries if e not in ignored_patterns and not e.endswith(".pyc")]
        for idx, entry in enumerate(filtered):
            is_last = (idx == len(filtered) - 1)
            connector = "└── " if is_last else "├── "
            entry_path = os.path.join(curr_dir, entry)
            
            if os.path.isdir(entry_path):
                tree_lines.append(f"{prefix}{connector}📁 {entry}/")
                new_prefix = prefix + ("    " if is_last else "│   ")
                _build_tree(entry_path, new_prefix, current_depth + 1)
            else:
                size_kb = os.path.getsize(entry_path) / 1024.0
                tree_lines.append(f"{prefix}{connector}📄 {entry} ({size_kb:.1f} KB)")
                
    _build_tree(target_dir, current_depth=0)
    return "\n".join(tree_lines)

@tool
def grep_search(query: str, path: str = ".", is_regex: bool = False, file_pattern: str = "*") -> str:
    """
    Search for exact text or regex patterns across files in the codebase.
    Returns matching filenames, line numbers, and code snippets.
    Args:
        query: String or regex pattern to search for.
        path: Base directory to search (defaults to project root).
        is_regex: Whether to treat query as a regular expression.
        file_pattern: Glob pattern to filter files (e.g. '*.py', '*.md').
    """
    target_dir = resolve_readable_path(path)
    if not os.path.exists(target_dir):
        return f"Error: Path '{path}' not found."
        
    ignored_dirs = {".git", "__pycache__", "venv", "chroma_db", "node_modules", ".ipynb_checkpoints"}
    matches = []
    
    try:
        pattern = re.compile(query, re.IGNORECASE) if is_regex else None
    except Exception as e:
        return f"Invalid regex pattern: {e}"
        
    for root, dirs, files in os.walk(target_dir):
        # Prune ignored directories in-place
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        
        for file in files:
            if not fnmatch.fnmatch(file, file_pattern):
                continue
                
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, ROOT_DIR)
            
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_num, line in enumerate(f, start=1):
                        matched = bool(pattern.search(line)) if is_regex else (query.lower() in line.lower())
                        if matched:
                            matches.append(f"{rel_path}:{line_num}: {line.strip()[:150]}")
                            if len(matches) >= 30:
                                matches.append("... [capped at 30 matches]")
                                return "\n".join(matches)
            except Exception:
                continue
                
    if not matches:
        return f"No matches found for '{query}'."
    return "\n".join(matches)

@tool
def view_code_slice(file_path: str, start_line: int = 1, end_line: int = 60) -> str:
    """
    View specific line ranges of a code or text file with 1-indexed line numbers.
    Args:
        file_path: Path to the file.
        start_line: Starting line number (1-indexed, inclusive).
        end_line: Ending line number (inclusive).
    """
    resolved_path = resolve_readable_path(file_path)
    if not os.path.exists(resolved_path) or os.path.isdir(resolved_path):
        return f"Error: File '{file_path}' not found."
        
    try:
        with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        total_lines = len(lines)
        if total_lines == 0:
            return f"File '{file_path}' is empty."
            
        start_idx = max(1, start_line)
        end_idx = min(total_lines, end_line)
        
        if start_idx > total_lines:
            return f"Error: start_line {start_idx} exceeds total lines ({total_lines})."
            
        output = [f"File: {os.path.relpath(resolved_path, ROOT_DIR)} (Lines {start_idx}-{end_idx} of {total_lines}):"]
        for i in range(start_idx, end_idx + 1):
            output.append(f"{i:4d}: {lines[i - 1].rstrip()}")
            
        return "\n".join(output)
    except Exception as e:
        return f"Error viewing code slice: {e}"

@tool
def find_files_by_pattern(pattern: str = "*.py", path: str = ".") -> str:
    """
    Find files matching a glob pattern (e.g., '*.py', '*.json', 'test_*').
    Args:
        pattern: Glob pattern to look for.
        path: Directory to start search from.
    """
    target_dir = resolve_readable_path(path)
    if not os.path.exists(target_dir):
        return f"Error: Path '{path}' not found."
        
    ignored_dirs = {".git", "__pycache__", "venv", "chroma_db", "node_modules"}
    matched_files = []
    
    for root, dirs, files in os.walk(target_dir):
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                rel = os.path.relpath(os.path.join(root, f), ROOT_DIR)
                matched_files.append(rel)
                if len(matched_files) >= 50:
                    matched_files.append("... [capped at 50 results]")
                    return "\n".join(matched_files)
                    
    if not matched_files:
        return f"No files matched pattern '{pattern}'."
    return "\n".join(matched_files)

@tool
def execute_terminal_command(command: str) -> str:
    """
    Execute a shell/terminal command strictly inside the sandboxed workspace directory (./workspace).
    Use this to run Node.js/Python scripts, build projects, or manage workspace files.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=WORKSPACE_DIR
        )
        output = result.stdout
        if result.stderr:
            output += f"\nErrors:\n{result.stderr}"
        return f"Executed in ./workspace (Exit code {result.returncode}).\nOutput:\n{output}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"

@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code in a stateful local REPL environment.
    Use this for data analysis, math calculations, or testing algorithms.
    Always use print() to see results.
    """
    try:
        result = repl.run(code)
        return f"Code executed successfully.\nOutput:\n{result}"
    except Exception as e:
        return f"Error executing Python code: {str(e)}"

@tool
def read_local_file(file_path: str) -> str:
    """
    Read the entire contents of a file. Checks ./workspace first, then project root.
    """
    try:
        resolved_path = resolve_readable_path(file_path)
        if not os.path.exists(resolved_path) or os.path.isdir(resolved_path):
            return f"Error: File '{file_path}' does not exist in workspace or root."
                
        with open(resolved_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_local_file(file_path: str, content: str) -> str:
    """
    Write content to a file strictly inside the sandboxed workspace directory (./workspace).
    Protects project source files and contains all generated code.
    """
    try:
        resolved_path = get_safe_workspace_path(file_path)
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        with open(resolved_path, 'w', encoding='utf-8') as f:
            f.write(content)
        rel_display = os.path.relpath(resolved_path, ROOT_DIR)
        return f"Successfully wrote to '{rel_display}' (contained in sandbox)."
    except Exception as e:
        return f"Error writing to file: {str(e)}"


