"""Git status/log/diff tools for the project and workspace."""

import os
import subprocess
from typing import List

from langchain_core.tools import tool

from src.tools.coding_tools import ROOT_DIR, WORKSPACE_DIR


def _run_git(args: List[str], cwd: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=cwd,
        )
        out = result.stdout or ""
        if result.stderr:
            out += ("\n" if out else "") + result.stderr
        if not out.strip():
            out = "(no output)"
        return out.strip()
    except FileNotFoundError:
        return "Error: git is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return "Error: git command timed out."
    except Exception as e:
        return f"Error running git: {e}"


@tool
def git_status() -> str:
    """Show git status of the project repository (branch, staged/unstaged/untracked)."""
    return _run_git(["status", "-sb"], ROOT_DIR)


@tool
def git_log(limit: int = 10) -> str:
    """
    Show recent git commit history (one-line format).
    Args:
        limit: Number of commits to show (default 10).
    """
    n = max(1, min(int(limit), 50))
    return _run_git(["log", f"-{n}", "--oneline", "--decorate"], ROOT_DIR)


@tool
def git_diff(path: str = "", staged: bool = False) -> str:
    """
    Show a git diff of the project. Optionally limit to a path and/or staged changes.
    Args:
        path: Optional file or directory path filter.
        staged: If true, show staged (cached) diff only.
    """
    args = ["diff"]
    if staged:
        args.append("--cached")
    if path:
        args.append("--")
        args.append(path)
    out = _run_git(args, ROOT_DIR)
    if len(out) > 8000:
        return out[:8000] + "\n...[diff truncated]"
    return out


@tool
def git_workspace_status() -> str:
    """Show git status of the ./workspace sandbox if it is a git repo; otherwise report that."""
    if not os.path.isdir(os.path.join(WORKSPACE_DIR, ".git")):
        return (
            "workspace/ is not a standalone git repo. "
            "Project-level git tools still work (git_status, git_log, git_diff). "
            "Use git_status to see untracked workspace files if they are not gitignored."
        )
    return _run_git(["status", "-sb"], WORKSPACE_DIR)
