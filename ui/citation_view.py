"""Streamlit citation rendering helpers — clickable source open/preview/download."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from src.rag.citations import find_snippet_lines, read_source_content


def render_citations(
    citations: Optional[List[Dict[str, Any]]],
    key_prefix: str = "cite",
    expanded: bool = False,
) -> None:
    """
    Render an interactive Sources & Citations panel.

    For each citation:
      - shows source name + score + relative path
      - "Open source" expander with matched snippet + file preview
      - Download button when the file exists on disk
    """
    if not citations:
        return

    with st.expander(f"📎 Sources & Citations ({len(citations)})", expanded=expanded):
        for i, c in enumerate(citations, 1):
            src = c.get("source") or "Unknown"
            score = c.get("score")
            score_txt = f" · `{score:.3f}`" if isinstance(score, (int, float)) else ""
            rel = c.get("rel_path") or ""
            exists = bool(c.get("exists"))
            readable = bool(c.get("readable"))
            path = c.get("path")

            header = f"**{i}.** `{src}`{score_txt}"
            if rel:
                header += f"  \n📁 `{rel}`"
            else:
                header += "  \n📁 *file not found on disk*"
            st.markdown(header)

            preview = c.get("preview") or ""
            if preview:
                st.caption(preview)

            col_open, col_dl = st.columns([1, 1])
            open_label = "🔍 Open source" if exists else "🔍 Source unavailable"
            with col_open:
                show = st.checkbox(
                    open_label,
                    key=f"{key_prefix}_open_{i}_{src}",
                    disabled=not exists,
                )
            with col_dl:
                if exists and path and readable:
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                        import os as _os

                        st.download_button(
                            "📥 Download",
                            data=data,
                            file_name=_os.path.basename(path),
                            key=f"{key_prefix}_dl_{i}_{src}",
                            use_container_width=True,
                        )
                    except OSError:
                        st.caption("Download failed")
                elif exists and path:
                    st.caption("Binary file — open outside the app")

            if show and exists and path:
                snippet_lines = find_snippet_lines(path, preview)
                if snippet_lines:
                    st.markdown("**Matched context:**")
                    st.code("\n".join(snippet_lines), language="text")
                if readable:
                    body = read_source_content(path, max_chars=12000)
                    st.markdown("**File preview:**")
                    st.code(body, language=_guess_lang(path))
                else:
                    st.info("This file is binary or too large to preview inline.")


def _guess_lang(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "json": "json",
        "md": "markdown",
        "sql": "sql",
        "csv": "csv",
        "sh": "bash",
        "yml": "yaml",
        "yaml": "yaml",
        "html": "html",
        "css": "css",
    }.get(ext, "text")


def render_citation_footer_markdown(citations: Optional[List[Dict[str, Any]]]) -> str:
    """Compact markdown footer for chat bubble text (already includes paths)."""
    from src.rag.citations import format_citation_footer

    if not citations:
        return ""
    return format_citation_footer(citations)
