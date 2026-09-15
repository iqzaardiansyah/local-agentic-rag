"""LLM endpoint health checks (OpenAI-compatible / Ollama)."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx


def _base_models_url(base_url: str) -> str:
    """
    Normalize an OpenAI-style base URL to a /models endpoint.
    Accepts:
      http://localhost:11434/v1
      http://localhost:11434
      https://xxxx.ngrok-free.app/v1
    """
    base = (base_url or "").rstrip("/")
    if not base:
        return ""
    if base.endswith("/v1"):
        return f"{base}/models"
    # Ollama native API also has /api/tags; prefer OpenAI-compatible /v1/models first.
    return f"{base}/v1/models"


def check_llm_endpoint(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 4.0,
) -> Dict[str, Any]:
    """
    Ping the configured LLM endpoint.

    Returns:
      {
        ok: bool,
        base_url: str,
        models_url: str,
        latency_ms: float | None,
        models: [str, ...],
        model_count: int,
        error: str | None,
        checked_at: iso str,
      }
    """
    if base_url is None:
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    if api_key is None:
        api_key = os.getenv("LLM_API_KEY", "ollama")

    models_url = _base_models_url(base_url)
    started = time.perf_counter()
    result: Dict[str, Any] = {
        "ok": False,
        "base_url": base_url,
        "models_url": models_url,
        "latency_ms": None,
        "models": [],
        "model_count": 0,
        "error": None,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": urlparse(base_url).netloc or base_url,
    }

    if not models_url:
        result["error"] = "LLM_BASE_URL is empty"
        return result

    headers = {"Authorization": f"Bearer {api_key or 'ollama'}"}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(models_url, headers=headers)
        latency = (time.perf_counter() - started) * 1000.0
        result["latency_ms"] = round(latency, 1)

        if resp.status_code >= 400:
            # Fallback: native Ollama /api/tags
            alt = base_url.rstrip("/")
            if alt.endswith("/v1"):
                alt = alt[: -len("/v1")]
            alt_url = f"{alt}/api/tags"
            try:
                with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                    alt_resp = client.get(alt_url, headers=headers)
                if alt_resp.status_code < 400:
                    data = alt_resp.json() or {}
                    names = [m.get("name") or m.get("model") or "" for m in data.get("models") or []]
                    result["ok"] = True
                    result["models"] = [n for n in names if n][:20]
                    result["model_count"] = len(result["models"])
                    result["models_url"] = alt_url
                    return result
            except Exception:
                pass
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return result

        data = resp.json() or {}
        items = data.get("data") or data.get("models") or []
        names = []
        for item in items:
            if isinstance(item, dict):
                names.append(item.get("id") or item.get("name") or "")
            elif isinstance(item, str):
                names.append(item)
        result["ok"] = True
        result["models"] = [n for n in names if n][:20]
        result["model_count"] = len(result["models"])
        return result

    except httpx.TimeoutException:
        result["error"] = f"Timeout after {timeout}s — endpoint unreachable?"
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def format_health_badge(status: Dict[str, Any]) -> str:
    """Short human-readable status line for UI captions."""
    if status.get("ok"):
        ms = status.get("latency_ms")
        ms_txt = f" · {ms} ms" if ms is not None else ""
        n = status.get("model_count") or 0
        return f"ONLINE{ms_txt} · {n} model(s)"
    err = (status.get("error") or "unreachable")[:80]
    return f"OFFLINE · {err}"
