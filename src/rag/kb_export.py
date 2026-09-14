"""
Full knowledge-base Markdown export. Free, local.

Combines data/ files, GraphRAG stats + top triples, Chroma/BM25 stats,
and episodic memory count into one downloadable report.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DEFAULT_OUT = os.path.join(ROOT_DIR, "workspace", "kb_export.md")

_SKIP_NAMES = {
    "chat_sessions.db",
    "web_search_cache.db",
    "eval_set.json",
    "knowledge_graph.json",
}
_TEXT_EXTS = {
    ".txt", ".md", ".py", ".json", ".csv", ".html", ".js", ".yml", ".yaml", ".ts",
}


def _list_data_files() -> List[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    names = []
    for name in sorted(os.listdir(DATA_DIR)):
        if name.startswith(".") or name in _SKIP_NAMES:
            continue
        path = os.path.join(DATA_DIR, name)
        if os.path.isfile(path):
            names.append(name)
    return names


def _read_text_file(path: str, max_chars: int = 4000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_chars)
        if os.path.getsize(path) > max_chars:
            text += "\n\n…[truncated for export]"
        return text
    except OSError as e:
        return f"*(could not read: {e})*"


def _kg_summary(max_triples: int = 40) -> Dict[str, Any]:
    try:
        from src.rag.graph_rag import get_kg_engine

        kg = get_kg_engine()
        stats = kg.get_graph_stats()
        triples = []
        for s, o, _k, data in list(kg.graph.edges(keys=True, data=True))[:max_triples]:
            triples.append(
                {
                    "subject": s,
                    "predicate": data.get("relation", _k),
                    "object": o,
                    "source": data.get("source", ""),
                }
            )
        return {"stats": stats, "triples": triples, "ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e), "stats": {}, "triples": []}


def _vector_stats() -> Dict[str, Any]:
    try:
        from src.rag.vectorstore import get_knowledge_base_stats
        from src.rag.hybrid_search import get_bm25_status

        return {
            "ok": True,
            "chroma": get_knowledge_base_stats(),
            "bm25": get_bm25_status(),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _memory_count() -> int:
    try:
        from src.memory.episodic_memory import list_all_memories

        return len(list_all_memories())
    except Exception:
        return 0


def export_knowledge_base_markdown(
    output_path: Optional[str] = None,
    include_file_bodies: bool = True,
    max_file_chars: int = 4000,
) -> Dict[str, Any]:
    """
    Write a full KB Markdown report and return path + counts.
    """
    out_path = output_path or DEFAULT_OUT
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    files = _list_data_files()
    kg = _kg_summary()
    vec = _vector_stats()
    mem_n = _memory_count()
    ts = datetime.now().isoformat(timespec="seconds")

    lines: List[str] = [
        "# Knowledge Base Export",
        "",
        f"- **Generated:** {ts}",
        f"- **Source:** `data/` + GraphRAG + Chroma stats",
        f"- **Files included:** {len(files)}",
        f"- **Episodic memories:** {mem_n}",
        "",
        "## Index Stats",
        "",
    ]
    if vec.get("ok"):
        chroma = vec.get("chroma") or {}
        bm25 = vec.get("bm25") or {}
        lines.append(f"- Chroma chunks: **{chroma.get('total_chunks', 0)}**")
        lines.append(f"- Data files on disk: **{chroma.get('total_files', 0)}**")
        lines.append(
            f"- BM25: **{'fresh' if not bm25.get('stale') else 'stale'}** "
            f"({bm25.get('documents', 0)} docs)"
        )
    else:
        lines.append(f"- Vectorstore stats unavailable: {vec.get('error')}")
    lines.append("")

    lines.extend(["## Knowledge Graph", ""])
    if kg.get("ok"):
        st = kg.get("stats") or {}
        lines.append(f"- Entities: **{st.get('total_entities', 0)}**")
        lines.append(f"- Relationships: **{st.get('total_relationships', 0)}**")
        hubs = st.get("top_hubs") or []
        if hubs:
            lines.append("- Top hubs:")
            for h in hubs[:5]:
                lines.append(f"  - `{h.get('entity')}` ({h.get('connections')} connections)")
        lines.append("")
        triples = kg.get("triples") or []
        if triples:
            lines.append("### Sample Triples")
            lines.append("")
            lines.append("| Subject | Predicate | Object | Source |")
            lines.append("| --- | --- | --- | --- |")
            for t in triples:
                lines.append(
                    f"| {t['subject']} | `{t['predicate']}` | {t['object']} | {t.get('source') or '—'} |"
                )
            lines.append("")
    else:
        lines.append(f"*(KG unavailable: {kg.get('error')})*")
        lines.append("")

    lines.extend(["## Documents", ""])
    if not files:
        lines.append("*No source files found in `data/`.*")
        lines.append("")
    for name in files:
        path = os.path.join(DATA_DIR, name)
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        ext = os.path.splitext(name)[1].lower()
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"- Size: {size:,} bytes")
        lines.append("")
        if include_file_bodies and ext in _TEXT_EXTS and size <= 2_000_000:
            body = _read_text_file(path, max_chars=max_file_chars)
            lang = ext.lstrip(".") or "text"
            lines.append(f"```{lang}")
            lines.append(body.rstrip())
            lines.append("```")
            lines.append("")
        else:
            lines.append("*(binary or large file — body omitted)*")
            lines.append("")

    lines.extend(
        [
            "---",
            f"*Exported from Local Agentic RAG at {ts}*",
            "",
        ]
    )

    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    return {
        "success": True,
        "path": out_path,
        "bytes": len(content.encode("utf-8")),
        "files": len(files),
        "triples": len(kg.get("triples") or []),
        "memories": mem_n,
    }
