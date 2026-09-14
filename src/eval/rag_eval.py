"""Offline RAG evaluation harness. Fully local, free, no paid APIs."""

import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DEFAULT_EVAL_SET = os.path.join(ROOT_DIR, "data", "eval_set.json")


def load_eval_set(path: Optional[str] = None) -> List[Dict[str, Any]]:
    eval_path = path or DEFAULT_EVAL_SET
    if not os.path.isfile(eval_path):
        raise FileNotFoundError(
            f"Eval set not found at {eval_path}. Create data/eval_set.json with a list of "
            '{"query", "expected_source_substring", "answer_keywords"} objects.'
        )
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("eval_set.json must be a JSON list of cases.")
    return data


def retrieve_for_query(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    from src.rag.hybrid_search import hybrid_search
    from src.rag.reranker import rerank_documents

    candidates = hybrid_search(query, top_k=12)
    if not candidates:
        return []
    reranked = rerank_documents(query, candidates, top_k=top_k)
    return [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "score": float(score),
            "content": doc.page_content,
        }
        for doc, score in reranked
    ]


def evaluate_case(case: Dict[str, Any], top_k: int = 3) -> Dict[str, Any]:
    query = case["query"]
    expected = (case.get("expected_source_substring") or "").lower()
    results = retrieve_for_query(query, top_k=top_k)

    sources = [r["source"].lower() for r in results]
    hit = bool(expected) and any(expected in s for s in sources)

    rr = 0.0
    if expected:
        for rank, s in enumerate(sources, start=1):
            if expected in s:
                rr = 1.0 / rank
                break

    keywords = [k.lower() for k in case.get("answer_keywords") or []]
    joined = " ".join(r["content"].lower() for r in results)
    keyword_hits = sum(1 for k in keywords if k in joined)
    keyword_rate = (keyword_hits / len(keywords)) if keywords else None

    return {
        "query": query,
        "hit": hit,
        "reciprocal_rank": rr,
        "keyword_rate": keyword_rate,
        "retrieved_sources": sources,
        "top_score": results[0]["score"] if results else None,
    }


def evaluate_all(
    eval_path: Optional[str] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    cases = load_eval_set(eval_path)
    if not cases:
        return {"error": "Empty eval set.", "n": 0}

    per_case = [evaluate_case(c, top_k=top_k) for c in cases]
    n = len(per_case)
    hits = sum(1 for c in per_case if c["hit"])
    mrr = sum(c["reciprocal_rank"] for c in per_case) / n
    kw_rates = [c["keyword_rate"] for c in per_case if c["keyword_rate"] is not None]
    avg_kw = sum(kw_rates) / len(kw_rates) if kw_rates else None

    summary = {
        "n": n,
        "hit_rate_at_k": hits / n,
        "mrr": mrr,
        "avg_keyword_rate": avg_kw,
        "top_k": top_k,
        "cases": per_case,
    }
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Offline RAG eval (Hit@k, MRR, keyword rate)")
    parser.add_argument("--eval-set", default=None, help="Path to eval_set.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Auto-generate/merge eval cases from data/ before evaluating",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only write auto-generated eval cases; do not run retrieval metrics",
    )
    parser.add_argument("--replace", action="store_true", help="With --generate: overwrite eval set")
    args = parser.parse_args()

    if args.generate or args.generate_only:
        from src.eval.auto_eval_set import write_eval_set

        info = write_eval_set(path=args.eval_set, replace=args.replace)
        print(
            f"Eval set: {info['total']} cases at {info['path']} "
            f"({info['generated']} generated this run)"
        )
        if args.generate_only:
            return

    report = evaluate_all(eval_path=args.eval_set, top_k=args.top_k)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("=== Offline RAG Evaluation ===")
    print(f"Cases: {report['n']}   top_k={report['top_k']}")
    print(f"Hit Rate@k : {report['hit_rate_at_k']:.3f}")
    print(f"MRR        : {report['mrr']:.3f}")
    if report.get("avg_keyword_rate") is not None:
        print(f"Avg keyword: {report['avg_keyword_rate']:.3f}")
    print()
    for c in report["cases"]:
        mark = "HIT " if c["hit"] else "MISS"
        print(f"[{mark}] rr={c['reciprocal_rank']:.3f}  {c['query'][:70]}")
        print(f"       sources: {c['retrieved_sources']}")


if __name__ == "__main__":
    main()
