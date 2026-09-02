from langchain_core.tools import tool
from src.rag.graph_rag import get_kg_engine

@tool
def query_knowledge_graph(entity_or_concept: str) -> str:
    """
    Query the internal GraphRAG Knowledge Graph to discover multi-hop relationships,
    organizational hierarchies, project owners, technology dependencies, and entity connections.
    Use this when a question asks about connections, ownership, dependencies, or multi-hop relationships.
    Args:
        entity_or_concept: The entity, person, project, department, or technology to explore (e.g. 'Alice Smith', 'Engineering', 'Local Agentic RAG').
    """
    kg = get_kg_engine()
    results = kg.search_subgraph(entity_or_concept, max_hops=2)
    
    if not results.get("found"):
        return f"Knowledge Graph: {results.get('message', 'No entities found.')}"
        
    matched = results.get("matched_entities", [])
    direct = results.get("direct_relations", [])
    multihop = results.get("multi_hop_paths", [])
    
    output = [f"🕸️ **Knowledge Graph Subgraph for:** `{', '.join(matched)}`\n"]
    
    # Direct Relations Table
    if direct:
        output.append("### 🔗 Direct Relationships (1-Hop):")
        output.append("| Subject | Relationship | Object |")
        output.append("| --- | --- | --- |")
        for r in direct:
            output.append(f"| **{r['subject']}** | `:{r['predicate']}:` | **{r['object']}** |")
        output.append("")
        
    # Multi-hop paths
    if multihop:
        output.append("### 🌐 Multi-Hop Connected Paths (2-Hop):")
        for p in multihop[:8]:
            output.append(f"- **{p['start']}** ──(`:{p['hop1_predicate']}:`)──▶ **{p['intermediate']}** ──(`:{p['hop2_predicate']}:`)──▶ **{p['end']}**")
        output.append("")
        
    return "\n".join(output)
