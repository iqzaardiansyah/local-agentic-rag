"""
Local .env / runtime config validation. Free, no cloud calls.

Surfaces missing keys, bad URLs, and placeholder values before a long agent turn.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
ENV_EXAMPLE_PATH = os.path.join(ROOT_DIR, ".env.example")

REQUIRED_KEYS = ("LLM_BASE_URL", "LLM_MODEL")
OPTIONAL_KEYS = ("LLM_API_KEY",)


def _read_env_file(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not os.path.isfile(path):
        return data
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                data[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return data


def validate_config(
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate LLM endpoint config.

    Returns:
      {
        ok: bool,
        issues: [{level: error|warning, key, message}],
        values: {LLM_BASE_URL, LLM_MODEL, LLM_API_KEY},
        env_exists: bool,
        env_path: str,
      }
    """
    file_vals = _read_env_file(ENV_PATH)
    values = {
        "LLM_BASE_URL": base_url if base_url is not None else os.getenv("LLM_BASE_URL") or file_vals.get("LLM_BASE_URL", ""),
        "LLM_MODEL": model if model is not None else os.getenv("LLM_MODEL") or file_vals.get("LLM_MODEL", ""),
        "LLM_API_KEY": api_key if api_key is not None else os.getenv("LLM_API_KEY") or file_vals.get("LLM_API_KEY", ""),
    }

    issues: List[Dict[str, str]] = []
    env_exists = os.path.isfile(ENV_PATH)

    if not env_exists:
        issues.append({
            "level": "warning",
            "key": ".env",
            "message": f"No `.env` file at {ENV_PATH}. Using process env / defaults. Copy `.env.example` to get started.",
        })

    url = (values.get("LLM_BASE_URL") or "").strip()
    if not url:
        issues.append({
            "level": "error",
            "key": "LLM_BASE_URL",
            "message": "LLM_BASE_URL is empty. Expected e.g. http://localhost:11434/v1",
        })
    else:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            issues.append({
                "level": "error",
                "key": "LLM_BASE_URL",
                "message": f"LLM_BASE_URL must be http(s). Got scheme={parsed.scheme!r}.",
            })
        if not parsed.netloc:
            issues.append({
                "level": "error",
                "key": "LLM_BASE_URL",
                "message": "LLM_BASE_URL is missing a host.",
            })
        if "example" in url.lower() or "xxxx" in url.lower():
            issues.append({
                "level": "warning",
                "key": "LLM_BASE_URL",
                "message": "LLM_BASE_URL looks like a placeholder/example URL.",
            })
        if "ngrok" in url.lower() and not url.startswith("https://"):
            issues.append({
                "level": "warning",
                "key": "LLM_BASE_URL",
                "message": "ngrok endpoints usually require https://.",
            })

    model = (values.get("LLM_MODEL") or "").strip()
    if not model:
        issues.append({
            "level": "error",
            "key": "LLM_MODEL",
            "message": "LLM_MODEL is empty. Expected e.g. qwen3:8b or llama3.2:latest",
        })
    elif re.search(r"\s", model):
        issues.append({
            "level": "warning",
            "key": "LLM_MODEL",
            "message": "LLM_MODEL contains whitespace — usually unexpected.",
        })

    key = (values.get("LLM_API_KEY") or "").strip()
    if not key:
        issues.append({
            "level": "warning",
            "key": "LLM_API_KEY",
            "message": "LLM_API_KEY is empty. OpenAI clients usually need any non-empty string for local Ollama (e.g. 'ollama').",
        })

    # Compare with example keys (informational)
    if env_exists and os.path.isfile(ENV_EXAMPLE_PATH):
        example_keys = set(_read_env_file(ENV_EXAMPLE_PATH).keys())
        present = set(file_vals.keys())
        missing_optional = example_keys - present - set(REQUIRED_KEYS)
        if missing_optional:
            issues.append({
                "level": "warning",
                "key": ",".join(sorted(missing_optional)),
                "message": f"Optional keys from .env.example not set: {', '.join(sorted(missing_optional))}",
            })

    has_error = any(i["level"] == "error" for i in issues)
    return {
        "ok": not has_error,
        "issues": issues,
        "values": {
            "LLM_BASE_URL": values.get("LLM_BASE_URL") or "",
            "LLM_MODEL": values.get("LLM_MODEL") or "",
            "LLM_API_KEY": ("***" if values.get("LLM_API_KEY") else ""),
        },
        "env_exists": env_exists,
        "env_path": ENV_PATH,
        "error_count": sum(1 for i in issues if i["level"] == "error"),
        "warning_count": sum(1 for i in issues if i["level"] == "warning"),
    }


def format_config_report(report: Dict[str, Any]) -> str:
    """Human-readable multi-line config report."""
    lines = []
    if report.get("ok"):
        lines.append("✅ Config looks valid.")
    else:
        lines.append(f"❌ Config has {report.get('error_count', 0)} error(s).")
    lines.append(f"- `.env`: {report.get('env_path')} ({'exists' if report.get('env_exists') else 'missing'})")
    for i in report.get("issues") or []:
        icon = "❌" if i.get("level") == "error" else "⚠️"
        lines.append(f"{icon} `{i.get('key')}`: {i.get('message')}")
    return "\n".join(lines)
