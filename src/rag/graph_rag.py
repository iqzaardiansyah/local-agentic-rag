import os
import re
import json
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from networkx.readwrite import json_graph

GRAPH_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "knowledge_graph.json")

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
            
    def extract_and_ingest_text(self, text: str, source_doc: str = "doc"):
        """
        Rule-based NLP regex & pattern triple extractor for ingested documents.
        Extracts structural definitions, leadership, dependencies, and usages.
        """
        # Patterns for common relation structures
        patterns = [
            # X uses / leverages / utilizes Y
            (r"([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:uses|utilizes|leverages|integrates)\s+([A-Z][A-Za-z0-9_\s]{2,30})", "USES"),
            # X leads / manages / directs Y
            (r"([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:leads|manages|directs|heads)\s+([A-Z][A-Za-z0-9_\s]{2,30})", "LEADS"),
            # X belongs to / part of Y
            (r"([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:belongs to|is part of|is managed by)\s+([A-Z][A-Za-z0-9_\s]{2,30})", "BELONGS_TO"),
            # X depends on / requires Y
            (r"([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:depends on|requires|relies on)\s+([A-Z][A-Za-z0-9_\s]{2,30})", "DEPENDS_ON"),
            # X includes / contains Y
            (r"([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:includes|contains|features)\s+([A-Z][A-Za-z0-9_\s]{2,30})", "INCLUDES"),
            # X has budget of Y
            (r"([A-Z][A-Za-z0-9_\s]{2,30})\s+(?:has a budget of|budget is)\s+(\$[0-9,]+)", "HAS_BUDGET")
        ]
        
        extracted_count = 0
        for pat, rel in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                subj, obj = match.group(1).strip(), match.group(2).strip()
                if len(subj) > 2 and len(obj) > 1 and subj.lower() != obj.lower():
                    self.add_triple(subj, rel, obj, {"source": source_doc}, persist=False)
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
