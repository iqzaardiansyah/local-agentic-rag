import os
import re
import json
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from networkx.readwrite import json_graph

GRAPH_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "knowledge_graph.json")

# Stop-word tokens that should not be treated as entities.
_ENTITY_STOP = {
    "the", "this", "that", "these", "those", "there", "here", "and", "but",
    "for", "with", "from", "into", "onto", "over", "under", "about", "after",
    "before", "between", "through", "using", "used", "use", "also", "very",
    "more", "most", "some", "any", "each", "every", "all", "both", "such",
    "when", "where", "while", "which", "who", "whom", "whose", "what", "how",
    "why", "because", "therefore", "however", "although", "unless", "until",
    "please", "note", "example", "examples", "section", "chapter", "page",
    "figure", "table", "item", "items", "list", "number", "value", "values",
    "file", "files", "line", "lines", "code", "text", "content", "data",
    "user", "users", "system", "systems", "project", "projects", "name",
    "names", "type", "types", "set", "get", "run", "make", "take", "give",
}


def _clean_entity(raw: str) -> str:
    """Trim punctuation/whitespace and collapse internal spacing."""
    s = re.sub(r"\s+", " ", (raw or "").strip())
    s = s.strip(" \t\n\r.,;:!?()[]{}\"'`“”‘’")
    # Drop trailing verbs/adverbs that regex sometimes swallows
    s = re.sub(
        r"\s+(?:is|are|was|were|be|been|being|to|of|in|on|at|by|for|and|or|the|a|an)$",
        "",
        s,
        flags=re.I,
    ).strip(" .,;:")
    return s


def _entity_ok(ent: str) -> bool:
    """Reject noisy / non-entity strings so the graph does not flood."""
    if not ent or len(ent) < 2 or len(ent) > 48:
        return False
    # Must contain at least one letter
    if not re.search(r"[A-Za-z]", ent):
        return False
    # Reject if entirely stopwords
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", ent)]
    if tokens and all(t in _ENTITY_STOP for t in tokens):
        return False
    # Reject sentence-like fragments (too many lowercase function words)
    if len(tokens) >= 6:
        return False
    # Reject if mostly non-alphanumeric
    alnum = sum(1 for c in ent if c.isalnum() or c in " _-")
    if alnum / max(len(ent), 1) < 0.7:
        return False
    return True


def _looks_like_proper_or_technical(ent: str) -> bool:
    """Prefer capitalized heads, digits, or known multiword proper names."""
    head = ent.strip().split()[0] if ent.strip() else ""
    if head[:1].isupper():
        return True
    if any(ch.isdigit() for ch in ent):
        return True
    # CamelCase / ALLCAPS tech tokens
    if re.search(r"[A-Z]{2,}", ent) or re.search(r"[a-z]+[A-Z]", ent):
        return True
    return False


# Relation patterns. Each: (regex, predicate, require_proper_object)
# (regex, predicate, require_proper_object)
_KG_PATTERNS = [
    # X uses / utilizes / leverages / integrates / is built with Y
    (
        r"\b([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Z][A-Za-z0-9_.\-]{1,30}){0,3})\s+"
        r"(?:uses|utilizes|utilises|leverages|integrates|employs)\s+"
        r"([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Z][A-Za-z0-9_.\-]{0,30}){0,3})",
        "USES",
        True,
    ),
    # Y is used by X  →  X USES Y
    (
        r"\b([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{1,30}){0,3})\s+is\s+used\s+by\s+"
        r"([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{0,30}){0,3})",
        "USES",
        True,
    ),
    # X leads / manages / directs / heads / runs Y
    (
        r"\b([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Z][A-Za-z0-9_.\-]{1,30}){0,3})\s+"
        r"(?:leads|manages|directs|heads|oversees|runs)\s+"
        r"([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{0,30}){0,3})",
        "LEADS",
        True,
    ),
    # X works in / works for / belongs to / is part of / is a member of Y
    (
        r"\b([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{1,30}){0,3})\s+"
        r"(?:works\s+(?:in|for|at)|belongs\s+to|is\s+part\s+of|is\s+a\s+member\s+of|is\s+managed\s+by)\s+"
        r"([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{0,30}){0,3})",
        "BELONGS_TO",
        True,
    ),
    # X depends on / requires / relies on Y
    (
        r"\b([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{1,30}){0,3})\s+"
        r"(?:depends\s+on|requires|relies\s+on)\s+"
        r"([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{0,30}){0,3})",
        "DEPENDS_ON",
        True,
    ),
    # X includes / contains / features Y
    (
        r"\b([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{1,30}){0,3})\s+"
        r"(?:includes|contains|features)\s+"
        r"([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{0,30}){0,3})",
        "INCLUDES",
        True,
    ),
    # X is built with / powered by / based on Y
    (
        r"\b([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{1,30}){0,3})\s+"
        r"(?:is\s+built\s+with|is\s+powered\s+by|is\s+based\s+on|is\s+implemented\s+(?:in|with))\s+"
        r"([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{0,30}){0,3})",
        "BUILT_WITH",
        True,
    ),
    # X created / authored / wrote / developed Y
    (
        r"\b([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{1,30}){0,3})\s+"
        r"(?:created|authored|wrote|developed|founded|invented)\s+"
        r"([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{0,30}){0,3})",
        "CREATED",
        True,
    ),
    # X has a budget of / budget is $Y
    (
        r"\b([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{1,30}){0,3})\s+"
        r"(?:has\s+a\s+budget\s+of|budget\s+is)\s+(\$[0-9][0-9,\.]*)",
        "HAS_BUDGET",
        False,
    ),
    # X is located in / based in / headquartered in Y
    (
        r"\b([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{1,30}){0,3})\s+"
        r"(?:is\s+located\s+in|is\s+based\s+in|is\s+headquartered\s+in)\s+"
        r"([A-Z][A-Za-z0-9_.\-]{1,30}(?:\s+[A-Za-z0-9_.\-]{0,30}){0,3})",
        "LOCATED_IN",
        True,
    ),
    # Markdown-ish: **X** uses **Y**  (already covered by first pattern with case)
]


class KnowledgeGraphEngine:
    """
    Lightweight, in-memory & persistent GraphRAG engine powered by NetworkX.
    Extracts semantic triples (Subject, Relation, Object), stores multi-directed edges,
    and performs 1-hop and 2-hop relational traversals for multi-hop question answering.
    """
    def __init__(self, storage_path: str = GRAPH_DATA_PATH):
        self.storage_path = storage_path
        self.graph = nx.MultiDiGraph()
        self.load_graph()

    def load_graph(self):
        """Load graph from JSON file or seed with initial knowledge."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.graph = json_graph.node_link_graph(data, directed=True, multigraph=True)
                    return
            except Exception:
                pass
        self._seed_default_graph()
        self.save_graph()

    def save_graph(self):
        """Persist graph to JSON disk file."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        try:
            data = json_graph.node_link_data(self.graph)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving knowledge graph: {e}")

    def _seed_default_graph(self):
        """Seed foundational project relationships."""
        seed_triples = [
            ("Local Agentic RAG", "USES", "LangGraph", {"type": "architecture"}),
            ("Local Agentic RAG", "USES", "Ollama", {"type": "llm_backend"}),
            ("Local Agentic RAG", "USES", "ChromaDB", {"type": "vectorstore"}),
            ("Local Agentic RAG", "USES", "NetworkX", {"type": "knowledge_graph"}),
            ("Local Agentic RAG", "INCLUDES", "Reflexion Auto-Debugger", {"type": "feature"}),
            ("Local Agentic RAG", "INCLUDES", "Cross-Encoder Reranker", {"type": "feature"}),
            ("Local Agentic RAG", "INCLUDES", "Hybrid Search RRF", {"type": "feature"}),
            ("Local Agentic RAG", "INCLUDES", "Episodic Memory", {"type": "feature"}),
            ("Local Agentic RAG", "INCLUDES", "Parallel Subagents", {"type": "feature"}),

            # Enterprise Organization Graph
            ("Alice Smith", "LEADS", "Engineering", {"title": "Principal AI Architect"}),
            ("Alice Smith", "MANAGES", "Local AI Agent Hub", {"role": "Lead"}),
            ("Bob Jones", "WORKS_IN", "Engineering", {"title": "Senior DevOps Engineer"}),
            ("Fiona Gallagher", "WORKS_IN", "Engineering", {"title": "ML Systems Engineer"}),
            ("Charlie Brown", "LEADS", "Product", {"title": "Lead Product Manager"}),
            ("Diana Prince", "LEADS", "Marketing", {"title": "Growth Marketing Director"}),
            ("Evan Wright", "LEADS", "HR", {"title": "People Operations Lead"}),

            ("Engineering", "HAS_BUDGET", "$1,500,000", {"currency": "USD"}),
            ("Product", "HAS_BUDGET", "$600,000", {"currency": "USD"}),
            ("Marketing", "HAS_BUDGET", "$400,000", {"currency": "USD"}),
            ("HR", "HAS_BUDGET", "$200,000", {"currency": "USD"}),

            ("Local AI Agent Hub", "BELONGS_TO", "Engineering", {"status": "Active"}),
            ("Enterprise Semantic Search", "BELONGS_TO", "Engineering", {"status": "Completed"}),
            ("Q3 Global Marketing Push", "BELONGS_TO", "Marketing", {"status": "Active"}),
            ("HR Automation Suite", "BELONGS_TO", "HR", {"status": "Planning"})
        ]
        for subj, pred, obj, meta in seed_triples:
            self.add_triple(subj, pred, obj, meta, persist=False)

    def add_triple(self, subject: str, predicate: str, object_: str, metadata: Optional[Dict[str, Any]] = None, persist: bool = True):
        """Add a directed relation triple into the Knowledge Graph."""
        s = subject.strip()
        p = predicate.strip().upper().replace(" ", "_")
        o = object_.strip()

        if not s or not p or not o:
            return

        if not self.graph.has_node(s):
            self.graph.add_node(s, type="entity")
        if not self.graph.has_node(o):
            self.graph.add_node(o, type="entity")

        self.graph.add_edge(s, o, key=p, relation=p, **(metadata or {}))
        if persist:
            self.save_graph()

    def extract_and_ingest_text(
        self,
        text: str,
        source_doc: str = "doc",
        max_triples: int = 40,
    ) -> int:
        """
        Rule-based regex triple extractor with conservative entity validation.

        Improvements vs the original extractor:
        - More relations (BUILT_WITH, CREATED, LOCATED_IN, passive USES)
        - Entity hygiene (_clean_entity / _entity_ok) to stop junk nodes
        - Prefer proper/technical object names
        - Per-document cap (max_triples) so ingest cannot flood the graph
        - Deduplicates identical triples within one document
        """
        if not text or not text.strip():
            return 0

        # Work on a de-hyphenated / wrapped-line flattened copy for better matches.
        flat = re.sub(r"-\n", "", text)
        flat = re.sub(r"\s+", " ", flat)

        seen: set = set()
        extracted_count = 0
        for pat, rel, require_proper_obj in _KG_PATTERNS:
            if extracted_count >= max_triples:
                break
            for match in re.finditer(pat, flat):
                if extracted_count >= max_triples:
                    break
                subj = _clean_entity(match.group(1))
                obj = _clean_entity(match.group(2))
                if not _entity_ok(subj) or not _entity_ok(obj):
                    continue
                if subj.lower() == obj.lower():
                    continue
                if require_proper_obj and not _looks_like_proper_or_technical(obj):
                    continue
                # Subjects should at least look proper/technical when multi-token
                if len(subj.split()) >= 2 and not _looks_like_proper_or_technical(subj):
                    continue

                key = (subj.lower(), rel, obj.lower())
                if key in seen:
                    continue
                seen.add(key)

                self.add_triple(
                    subj,
                    rel,
                    obj,
                    {"source": source_doc, "extractor": "regex_v2"},
                    persist=False,
                )
                extracted_count += 1

        if extracted_count > 0:
            self.save_graph()
        return extracted_count

    def search_subgraph(self, query: str, max_hops: int = 2) -> Dict[str, Any]:
        """
        Performs multi-hop relational search across the knowledge graph.
        Returns matching entities, outgoing relations, incoming relations, and 2-hop connected paths.
        """
        q = query.lower().strip()
        matched_nodes = []
        
        # 1. Match Nodes by query substring
        for node in self.graph.nodes():
            if q in node.lower() or node.lower() in q:
                matched_nodes.append(node)
                
        if not matched_nodes:
            # Fallback: check token matches
            q_tokens = set(re.findall(r"\w+", q))
            for node in self.graph.nodes():
                node_tokens = set(re.findall(r"\w+", node.lower()))
                if q_tokens.intersection(node_tokens):
                    matched_nodes.append(node)
                    
        if not matched_nodes:
            return {"found": False, "message": f"No entities in Knowledge Graph match '{query}'."}
            
        results = {
            "found": True,
            "matched_entities": matched_nodes[:5],
            "direct_relations": [],
            "multi_hop_paths": []
        }
        
        visited_nodes = set(matched_nodes)
        
        # 2. Extract 1-Hop Relations
        for node in matched_nodes[:5]:
            # Outgoing edges
            for _, target, edge_key, data in self.graph.out_edges(node, keys=True, data=True):
                rel = data.get("relation", edge_key)
                results["direct_relations"].append({
                    "subject": node,
                    "predicate": rel,
                    "object": target,
                    "direction": "outgoing"
                })
                visited_nodes.add(target)
                
            # Incoming edges
            for source, _, edge_key, data in self.graph.in_edges(node, keys=True, data=True):
                rel = data.get("relation", edge_key)
                results["direct_relations"].append({
                    "subject": source,
                    "predicate": rel,
                    "object": node,
                    "direction": "incoming"
                })
                visited_nodes.add(source)
                
        # 3. Extract 2-Hop Relations (Multi-Hop) if max_hops >= 2
        if max_hops >= 2:
            for direct_item in results["direct_relations"][:10]:
                neighbor = direct_item["object"] if direct_item["direction"] == "outgoing" else direct_item["subject"]
                for _, target_2, edge_key_2, data_2 in self.graph.out_edges(neighbor, keys=True, data=True):
                    if target_2 not in matched_nodes:
                        rel_2 = data_2.get("relation", edge_key_2)
                        results["multi_hop_paths"].append({
                            "start": direct_item["subject"],
                            "hop1_predicate": direct_item["predicate"],
                            "intermediate": neighbor,
                            "hop2_predicate": rel_2,
                            "end": target_2
                        })
                        
        return results

    def get_graph_stats(self) -> Dict[str, Any]:
        """Returns node count, edge count, and top central hubs."""
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)[:5]
        top_hubs = [{"entity": node, "connections": deg} for node, deg in degrees]
        return {
            "total_entities": num_nodes,
            "total_relationships": num_edges,
            "top_hubs": top_hubs
        }

# Global singleton instance
_kg_engine = None

def get_kg_engine() -> KnowledgeGraphEngine:
    global _kg_engine
    if _kg_engine is None:
        _kg_engine = KnowledgeGraphEngine()
    return _kg_engine


def _format_triple_doc(relations: List[Dict[str, Any]], query: str) -> "Any":
    """Build a Document-like object from KG relations (lazy import to avoid cycles)."""
    from langchain_core.documents import Document

    lines = [f"Knowledge Graph context for query: {query}"]
    for r in relations:
        lines.append(f"{r['subject']} —[{r['predicate']}]→ {r['object']}")
    preview_entities = sorted({r["subject"] for r in relations} | {r["object"] for r in relations})
    return Document(
        page_content="\n".join(lines),
        metadata={
            "source": "knowledge_graph",
            "path": "data/knowledge_graph.json",
            "abs_path": GRAPH_DATA_PATH,
            "kind": "graphrag",
            "entities": ", ".join(preview_entities[:12]),
            "relation_count": len(relations),
            "query": query,
        },
    )


def graph_documents_for_query(query: str, max_docs: int = 3) -> List[Any]:
    """
    Convert Knowledge Graph subgraph hits into Documents for hybrid RRF fusion
    and the citation panel. Returns [] when the KG has no match for the query.

    Each Document is a structured triple listing, marked metadata.kind='graphrag'
    so downstream tools can treat it differently from file chunks.
    """
    try:
        kg = get_kg_engine()
        sub = kg.search_subgraph(query, max_hops=2)
    except Exception:
        return []
    if not sub.get("found"):
        return []

    direct = sub.get("direct_relations") or []
    multihop = sub.get("multi_hop_paths") or []
    matched = sub.get("matched_entities") or []
    if not direct and not multihop and not matched:
        return []

    docs = []

    # Doc 1: matched entities summary (always when found)
    if matched:
        from langchain_core.documents import Document

        docs.append(
            Document(
                page_content=(
                    f"Knowledge Graph matched entities for '{query}': "
                    + ", ".join(matched)
                ),
                metadata={
                    "source": "knowledge_graph",
                    "path": "data/knowledge_graph.json",
                    "abs_path": GRAPH_DATA_PATH,
                    "kind": "graphrag",
                    "entities": ", ".join(matched),
                    "relation_count": 0,
                    "query": query,
                },
            )
        )

    # Doc 2: 1-hop triples
    if direct:
        docs.append(_format_triple_doc(direct[:12], query))

    # Doc 3: 2-hop paths as synthetic relations
    if multihop:
        hop_rels = [
            {
                "subject": p["start"],
                "predicate": f"{p['hop1_predicate']}→{p['hop2_predicate']}",
                "object": p["end"],
            }
            for p in multihop[:8]
        ]
        docs.append(_format_triple_doc(hop_rels, query))

    return docs[:max_docs]
