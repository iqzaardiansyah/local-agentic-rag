"""LLM settings loaded from environment / .env."""

from __future__ import annotations

import os
from typing import Any, Dict


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def load_llm_settings() -> Dict[str, Any]:
    """Read all model knobs from the environment."""
    enable_thinking = _env_bool("LLM_ENABLE_THINKING", False)
    thinking_budget = max(256, _env_int("LLM_THINKING_BUDGET", 4096))
    max_tokens = max(256, _env_int("LLM_MAX_TOKENS", 4096))
    if enable_thinking:
        min_tokens = thinking_budget + 2048
        if max_tokens < min_tokens:
            max_tokens = min_tokens

    return {
        "base_url": os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        "model": os.getenv("LLM_MODEL", "qwen3.8:27b"),
        "api_key": os.getenv("LLM_API_KEY", "ollama"),
        "enable_thinking": enable_thinking,
        "thinking_budget": thinking_budget,
        "max_tokens": max_tokens,
        "temperature": _env_float("LLM_TEMPERATURE", 0.7),
        "top_p": _env_float("LLM_TOP_P", 1.0),
        "top_k": _env_int("LLM_TOP_K", 20),
        "min_p": _env_float("LLM_MIN_P", 0.0),
        "repetition_penalty": _env_float("LLM_REPETITION_PENALTY", 1.0),
        "subagent_temperature": _env_float("SUBAGENT_TEMPERATURE", 0.4),
        "subagent_max_tokens": max(256, _env_int("SUBAGENT_MAX_TOKENS", max_tokens)),
        "streaming": True,
    }


def extra_body_from_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI-compatible extra_body for llama.cpp / Ollama."""
    body: Dict[str, Any] = {
        "top_k": settings["top_k"],
        "min_p": settings["min_p"],
        "repetition_penalty": settings["repetition_penalty"],
        "chat_template_kwargs": {"enable_thinking": bool(settings["enable_thinking"])},
    }
    if settings["enable_thinking"]:
        body["thinking"] = {
            "type": "enabled",
            "budget_tokens": settings["thinking_budget"],
        }
    return body
