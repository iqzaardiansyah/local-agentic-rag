"""LLM settings loaded from environment / .env."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional


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


def _env_str(name: str, default: str = "") -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def detect_model_family(model: str) -> str:
    m = (model or "").lower()
    if "k2-horizon" in m or "k2_horizon" in m or "k2horizon" in m:
        return "k2"
    if "qwen" in m:
        return "qwen"
    return "generic"


def _probe_server_max_parallel(base_url: str, timeout: float = 2.0) -> Optional[int]:
    """Try common server endpoints for max concurrency; None if unknown."""
    try:
        import httpx
        from urllib.parse import urlparse

        base = (base_url or "").rstrip("/")
        if not base:
            return None
        # Prefer the OpenAI root (strip trailing /v1)
        root = base[:-3] if base.endswith("/v1") else base
        candidates = [
            f"{root}/v1/models",
            f"{root}/metrics",
            f"{root}/props",  # llama.cpp
        ]
        headers = {"Authorization": "Bearer ollama"}
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            for url in candidates:
                try:
                    r = client.get(url, headers=headers)
                    if r.status_code >= 400:
                        continue
                    text = r.text
                    # llama.cpp /props: {"total_slots": N} or "n_parallel"
                    if url.endswith("/props"):
                        data = r.json() or {}
                        for key in ("total_slots", "n_parallel", "parallel", "max_parallel"):
                            if isinstance(data.get(key), int) and data[key] > 0:
                                return int(data[key])
                    # JSON body may include max_num_seqs
                    try:
                        data = r.json()
                        if isinstance(data, dict):
                            for key in ("max_num_seqs", "n_parallel", "total_slots"):
                                if isinstance(data.get(key), int) and data[key] > 0:
                                    return int(data[key])
                            # nested in some vLLM payloads
                            nested = data.get("data") or {}
                            if isinstance(nested, dict):
                                for key in ("max_num_seqs", "n_parallel"):
                                    if isinstance(nested.get(key), int) and nested[key] > 0:
                                        return int(nested[key])
                    except Exception:
                        pass
                    # Prometheus-style metrics
                    for line in text.splitlines():
                        if "max_num_seqs" in line or "n_parallel" in line or "total_slots" in line:
                            m = re.search(r"(\d+)\s*$", line.strip())
                            if m:
                                n = int(m.group(1))
                                if 1 <= n <= 64:
                                    return n
                except Exception:
                    continue
    except Exception:
        return None
    return None


_max_parallel_cache: Optional[int] = None


def get_max_parallel() -> int:
    """
    Max concurrent LLM requests (LLM_MAX_PARALLEL → server probe → default 4).
    """
    global _max_parallel_cache
    env = _env_int("LLM_MAX_PARALLEL", 0)
    if env > 0:
        _max_parallel_cache = min(env, 32)
        return _max_parallel_cache
    if _max_parallel_cache is not None:
        return _max_parallel_cache
    base = _env_str("LLM_BASE_URL", "")
    probed = _probe_server_max_parallel(base)
    _max_parallel_cache = min(probed, 32) if probed and probed > 0 else 4
    return _max_parallel_cache


def load_llm_settings() -> Dict[str, Any]:
    """Read all model knobs from the environment (with family-aware defaults)."""
    model = _env_str("LLM_MODEL", "qwen3.8:27b")
    family = detect_model_family(model)

    if family == "k2":
        def_temp, def_top_p, def_max_tokens = 1.0, 0.95, 8192
        def_reasoning = "high"
        def_tool_format = "xml"
    else:
        def_temp, def_top_p, def_max_tokens = 0.7, 1.0, 4096
        def_reasoning = "none"
        def_tool_format = "xml"

    enable_thinking = _env_bool("LLM_ENABLE_THINKING", family == "k2")
    thinking_budget = max(256, _env_int("LLM_THINKING_BUDGET", 4096))

    reasoning_effort = _env_str("LLM_REASONING_EFFORT", "").lower()
    if not reasoning_effort:
        if enable_thinking and family == "k2":
            reasoning_effort = def_reasoning
        elif enable_thinking:
            reasoning_effort = "high"
        else:
            reasoning_effort = "none"
    if reasoning_effort not in {"none", "low", "medium", "high"}:
        reasoning_effort = def_reasoning

    tool_call_format = _env_str("LLM_TOOL_CALL_FORMAT", "").lower()
    if tool_call_format not in {"json", "xml", "xml_typed"}:
        tool_call_format = def_tool_format

    max_tokens = max(256, _env_int("LLM_MAX_TOKENS", def_max_tokens))
    reasoning_on = reasoning_effort not in {"none", ""} or enable_thinking
    if reasoning_on and family != "k2":
        min_tokens = thinking_budget + 2048
        if max_tokens < min_tokens:
            max_tokens = min_tokens

    sub_temp_default = 1.0 if family == "k2" else 0.4

    return {
        "base_url": _env_str("LLM_BASE_URL", "http://localhost:11434/v1"),
        "model": model,
        "api_key": _env_str("LLM_API_KEY", "ollama"),
        "family": family,
        "enable_thinking": reasoning_on,
        "reasoning_effort": reasoning_effort,
        "tool_call_format": tool_call_format,
        "thinking_budget": thinking_budget,
        "max_tokens": max_tokens,
        "temperature": _env_float("LLM_TEMPERATURE", def_temp),
        "top_p": _env_float("LLM_TOP_P", def_top_p),
        "top_k": _env_int("LLM_TOP_K", 20),
        "min_p": _env_float("LLM_MIN_P", 0.0),
        "repetition_penalty": _env_float("LLM_REPETITION_PENALTY", 1.0),
        "subagent_temperature": _env_float("SUBAGENT_TEMPERATURE", sub_temp_default),
        "subagent_max_tokens": max(256, _env_int("SUBAGENT_MAX_TOKENS", max_tokens)),
        "max_parallel": get_max_parallel(),
        "timeout_seconds": _env_float("LLM_TIMEOUT_SECONDS", 600.0),
        "streaming": True,
    }


def sanitize_messages_for_family(messages, family: Optional[str] = None):
    """Add missing thinking fields on assistant messages when the model requires them."""
    if family is None:
        family = detect_model_family(_env_str("LLM_MODEL", ""))
    if family != "k2":
        return list(messages)

    from langchain_core.messages import AIMessage

    thinking_keys = (
        "think",
        "reasoning",
        "reasoning_content",
        "think_fast",
        "think_faster",
    )
    out = []
    for m in messages:
        if isinstance(m, AIMessage):
            ak = dict(getattr(m, "additional_kwargs", None) or {})
            if not any(str(ak.get(k) or "").strip() for k in thinking_keys):
                ak["reasoning_content"] = ""
            new = AIMessage(content=m.content)
            new.additional_kwargs = ak
            if getattr(m, "tool_calls", None):
                new.tool_calls = m.tool_calls
            if getattr(m, "invalid_tool_calls", None):
                new.invalid_tool_calls = m.invalid_tool_calls
            out.append(new)
        else:
            out.append(m)
    return out


_K2_PATCH_INSTALLED = False


def install_k2_openai_patch() -> bool:
    """Add reasoning_content to assistant messages in the OpenAI payload when needed."""
    global _K2_PATCH_INSTALLED
    if _K2_PATCH_INSTALLED:
        return True
    family = detect_model_family(_env_str("LLM_MODEL", ""))
    if family != "k2":
        return False

    import langchain_openai.chat_models.base as _openai_base
    from langchain_core.messages import AIMessage

    orig = _openai_base._convert_message_to_dict
    thinking_keys = (
        "think",
        "reasoning",
        "reasoning_content",
        "think_fast",
        "think_faster",
    )

    def _convert_message_to_dict_k2(message, api="chat/completions"):
        d = orig(message, api=api)
        if isinstance(message, AIMessage) and d.get("role") == "assistant":
            if not any(d.get(k) for k in thinking_keys):
                d["reasoning_content"] = " "
        return d

    _openai_base._convert_message_to_dict = _convert_message_to_dict_k2
    _K2_PATCH_INSTALLED = True
    return True


def extra_body_from_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI-compatible extra_body for llama.cpp / Ollama / K2 Horizon."""
    family = settings.get("family") or "generic"
    kwargs: Dict[str, Any] = {}

    if family == "k2":
        fmt = settings.get("tool_call_format") or "xml"
        if fmt not in {"json", "xml", "xml_typed"}:
            fmt = "xml"
        kwargs["reasoning_effort"] = settings.get("reasoning_effort") or "high"
        kwargs["tool_call_format"] = fmt
        return {"chat_template_kwargs": kwargs}

    kwargs["enable_thinking"] = bool(settings.get("enable_thinking"))
    body: Dict[str, Any] = {
        "top_k": settings.get("top_k", 20),
        "min_p": settings.get("min_p", 0.0),
        "repetition_penalty": settings.get("repetition_penalty", 1.0),
        "chat_template_kwargs": kwargs,
    }
    if settings.get("enable_thinking"):
        body["thinking"] = {
            "type": "enabled",
            "budget_tokens": settings.get("thinking_budget", 4096),
        }
    return body
