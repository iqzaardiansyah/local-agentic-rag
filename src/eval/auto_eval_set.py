"""Generate RAG eval cases from headings/sentences/keywords in data/."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
DEFAULT_EVAL_SET = os.path.join(DATA_DIR, "eval_set.json")

_TEXT_EXTS = {".txt", ".md", ".py", ".json", ".csv", ".html", ".js", ".yml", ".yaml"}
_SKIP_NAMES = {"eval_set.json", "knowledge_graph.json", "chat_sessions.db"}


def _is_candidate_file(name: str) -> bool:
    if name in _SKIP_NAMES or name.startswith("."):
        return False
    ext = os.path.splitext(name)[1].lower()
    return ext in _TEXT_EXTS


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text)


def _top_keywords(text: str, n: int = 5) -> List[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "you", "your", "all", "can", "has", "have", "will", "not", "but", "its",
        "into", "when", "what", "how", "why", "use", "using", "used", "each",
        "than", "then", "them", "they", "their", "there", "here", "also", "more",
        "some", "any", "one", "two", "out", "our", "his", "her", "him", "she",
        "get", "got", "set", "via", "per", "may", "new", "old", "see", "add",
    }
    counts: Dict[str, int] = {}
    for tok in _tokenize(text.lower()):
        if tok in stop or len(tok) < 3:
            continue
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:n]]


def _heading_lines(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or (len(s) < 90 and (s.endswith(":") or s.isupper())):
            cleaned = s.lstrip("#").strip()
            if 4 <= len(cleaned) <= 80:
                lines.append(cleaned)
    return lines[:8]


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    out = []
    for p in parts:
        p = " ".join(p.split()).strip()
        if 30 <= len(p) <= 180:
            out.append(p)
    return out[:40]


def generate_cases_for_file(path: str, max_cases: int = 2) -> List[Dict[str, Any]]:
    filename = os.path.basename(path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(80000)
    except OSError:
        return []
    if len(text.strip()) < 80:
        return []

    keywords = _top_keywords(text, n=6)
    if not keywords:
        return []

    cases: List[Dict[str, Any]] = []

    # Case from a heading (or first strong sentence)
    headings = _heading_lines(text)
    if headings:
        h = headings[0]
        q = f"What does the document say about {h.rstrip(':')}?"
        cases.append({
            "query": q,
            "expected_source_substring": filename,
            "answer_keywords": keywords[:4],
            "auto_generated": True,
            "derived_from": "heading",
        })
    else:
        sents = _sentences(text)
        if sents:
            # Turn first sentence into a cloze-ish query using its keywords
            sent_kw = _top_keywords(sents[0], n=3)
            if sent_kw:
                q = f"Tell me about {sent_kw[0]}"
                if len(sent_kw) > 1:
                    q = f"What is {sent_kw[0]} related to {sent_kw[1]}?"
                cases.append({
                    "query": q,
                    "expected_source_substring": filename,
                    "answer_keywords": keywords[:4],
                    "auto_generated": True,
                    "derived_from": "sentence",
                })

    # Second case: keyword combination query
    if len(keywords) >= 2 and len(cases) < max_cases:
        cases.append({
            "query": f"How are {keywords[0]} and {keywords[1]} related in {filename}?",
            "expected_source_substring": filename,
            "answer_keywords": keywords[:5],
            "auto_generated": True,
            "derived_from": "keywords",
        })

    return cases[:max_cases]


def generate_eval_cases(
    data_dir: Optional[str] = None,
    max_per_file: int = 2,
    max_total: int = 30,
) -> List[Dict[str, Any]]:
    root = data_dir or DATA_DIR
    if not os.path.isdir(root):
        return []
    cases: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(root)):
        if not _is_candidate_file(name):
            continue
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        cases.extend(generate_cases_for_file(path, max_cases=max_per_file))
        if len(cases) >= max_total:
            break
    return cases[:max_total]


def merge_eval_sets(
    existing: List[Dict[str, Any]],
    generated: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep existing cases; append generated ones not already present by query."""
    seen = {(c.get("query") or "").strip().lower() for c in existing}
    merged = list(existing)
    for g in generated:
        key = (g.get("query") or "").strip().lower()
        if key and key not in seen:
            merged.append(g)
            seen.add(key)
    return merged


def write_eval_set(
    path: Optional[str] = None,
    data_dir: Optional[str] = None,
    max_per_file: int = 2,
    replace: bool = False,
) -> Dict[str, Any]:
    """
    Generate cases from data/ and write eval_set.json.

    replace=False → merge with existing cases.
    replace=True  → overwrite with only generated cases.
    """
    out_path = path or DEFAULT_EVAL_SET
    generated = generate_eval_cases(data_dir=data_dir, max_per_file=max_per_file)

    existing: List[Dict[str, Any]] = []
    if not replace and os.path.isfile(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = []

    final = generated if replace else merge_eval_sets(existing, generated)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return {
        "path": out_path,
        "generated": len(generated),
        "existing": len(existing),
        "total": len(final),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Auto-generate RAG eval cases from data/")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=None, help="Output eval_set.json path")
    parser.add_argument("--max-per-file", type=int, default=2)
    parser.add_argument("--replace", action="store_true", help="Overwrite instead of merge")
    args = parser.parse_args()

    info = write_eval_set(
        path=args.out,
        data_dir=args.data_dir,
        max_per_file=args.max_per_file,
        replace=args.replace,
    )
    print(
        f"Wrote {info['total']} cases to {info['path']} "
        f"({info['generated']} new, {info['existing']} existing kept)"
    )


if __name__ == "__main__":
    main()
