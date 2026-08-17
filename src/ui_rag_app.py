"""Streamlit UI for California Legal RAG (vector-only and vector+KG modes)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import chromadb
import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
from openai import OpenAI
from pyvis.network import Network
from sentence_transformers import SentenceTransformer

from doc_resolver import build_resolution_maps, resolve_where_params
from step5_rag_answer import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OPENAI_MODEL,
    build_where,
    format_context,
    kg_expand_chunk_ids,
)

# Node styling, roughly matching the report's own color scheme.
NODE_COLORS = {
    "Document": "#8e44ad",
    "Chunk (vector seed)": "#e74c3c",
    "Chunk (KG-expanded)": "#f39c12",
    "Article": "#16a085",
    "Section": "#1abc9c",
    "Topic": "#f1c40f",
    "Entity": "#7f8c8d",
}

PAGE_ICON = "⚖️"


def loading_indicator_html(message: str) -> str:
    """A self-contained (no external gif) law-themed 'thinking' animation --
    balancing scales rocking side to side, plus a flashing ellipsis, in the
    same spirit as ChatGPT/Claude's typing indicator. Colors are left to
    inherit from Streamlit's current theme so it reads correctly in both
    light and dark mode."""
    return f"""
    <div style="display:flex;align-items:center;gap:14px;padding:10px 0;">
        <span style="font-size:2.4rem;display:inline-block;transform-origin:50% 15%;
                      animation:kg-scale-balance 1.3s ease-in-out infinite;">&#9878;&#65039;</span>
        <span style="font-size:1.05rem;opacity:0.85;">
            {message}<span class="kg-dot" style="animation-delay:0s;">.</span><span class="kg-dot" style="animation-delay:0.2s;">.</span><span class="kg-dot" style="animation-delay:0.4s;">.</span>
        </span>
    </div>
    <style>
    @keyframes kg-scale-balance {{
        0%, 100% {{ transform: rotate(-10deg); }}
        50% {{ transform: rotate(10deg); }}
    }}
    @keyframes kg-dot-flash {{
        0%, 80%, 100% {{ opacity: 0; }}
        40% {{ opacity: 1; }}
    }}
    .kg-dot {{
        animation: kg-dot-flash 1.3s infinite;
        font-weight: bold;
    }}
    </style>
    """


@st.cache_resource
def get_embed_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


@st.cache_resource
def get_collection(chroma_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(name=collection_name)


@st.cache_resource
def get_resolution_maps(chunks_path: str):
    return build_resolution_maps(Path(chunks_path))


def run_query(
    query: str,
    top_k: int,
    chroma_dir: str,
    collection_name: str,
    embed_model_name: str,
    openai_model: str,
    doc_type: str | None,
    edcode_section: str | None,
    article: str | None,
    doc_id: str | None,
    use_kg: bool,
    kg_expand_k: int,
    kg_min_similarity: float,
    chunks_path: str = "data/chunks/chunks.jsonl",
    resolve_doc_id: bool = True,
) -> Tuple[str, List[Dict[str, Any]], str, List[str], Set[str]]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if not Path(chroma_dir).exists():
        raise FileNotFoundError(f"Missing Chroma dir: {chroma_dir}")

    collection = get_collection(chroma_dir, collection_name)
    embed_model = get_embed_model(embed_model_name)
    query_emb = embed_model.encode([query], normalize_embeddings=True)

    effective_article, effective_edcode_section, effective_doc_id = article, edcode_section, doc_id
    if resolve_doc_id and not doc_id and (article or edcode_section) and Path(chunks_path).exists():
        _, article_to_docs, edcode_section_to_docs = get_resolution_maps(chunks_path)
        effective_article, effective_edcode_section, effective_doc_id = resolve_where_params(
            doc_type, article, edcode_section, article_to_docs, edcode_section_to_docs
        )

    where = build_where(doc_type, effective_edcode_section, effective_article, effective_doc_id)
    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    seed_ids: Set[str] = set(ids)

    if use_kg and ids:
        extra_ids = kg_expand_chunk_ids(
            ids,
            kg_expand_k,
            query_embedding=query_emb,
            collection=collection,
            min_similarity=kg_min_similarity,
        )
        if extra_ids:
            extra = collection.get(ids=extra_ids, include=["documents", "metadatas"])
            extra_ids_returned = extra.get("ids", extra_ids)
            extra_docs = extra.get("documents", [])
            extra_metas = extra.get("metadatas", [])

            seen = set()
            merged_docs: List[str] = []
            merged_metas: List[Dict[str, Any]] = []
            merged_ids: List[str] = []
            for cid, d, m in list(zip(ids, docs, metas)) + list(zip(extra_ids_returned, extra_docs, extra_metas)):
                key = (
                    m.get("doc_id"),
                    m.get("chunk_index"),
                    m.get("start_char"),
                    m.get("end_char"),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged_docs.append(d)
                merged_metas.append(m)
                merged_ids.append(cid)
            docs, metas, ids = merged_docs, merged_metas, merged_ids

    context = format_context(docs, metas)

    system_prompt = (
        "You are a legal RAG assistant for California Education Code (Title 1) "
        "and the California Constitution. Answer strictly using the provided context. "
        "Cite sources as [#] using the numbered context blocks. If the answer is not "
        "in the context, say you don't have enough information."
    )

    client = OpenAI()
    response = client.responses.create(
        model=openai_model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
        ],
    )
    answer = (getattr(response, "output_text", None) or str(response)).strip()

    cited = set(int(x) for x in re.findall(r"\[(\d+)\]", answer))
    cited_sources: List[Dict[str, Any]] = []
    for i, meta in enumerate(metas, 1):
        if cited and i not in cited:
            continue
        cited_sources.append(
            {
                "idx": i,
                "label": meta.get("source_label", "source"),
                "rel_path": meta.get("rel_path", ""),
                "anchors": meta.get("anchors_json", ""),
            }
        )

    return answer, cited_sources, context, ids, seed_ids


def fetch_subgraph(
    chunk_ids: List[str], seed_ids: Set[str]
) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, str, str]]]:
    """
    Pull the Document/Article/Section/Entity/Topic neighborhood of the given
    chunk_ids straight out of Neo4j, for display as an interactive graph in
    the UI right after an answer is generated. Only looks at direct
    neighbors of the retrieved chunks (no further traversal), so this stays
    small and readable regardless of how "busy" any single entity node is
    elsewhere in the full KG.
    """
    if not chunk_ids:
        return {}, []

    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    if not (uri and user and password):
        raise RuntimeError("NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD are not set -- cannot fetch the graph view.")

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Tuple[str, str, str]] = []

    def add_node(node_id: str, label: str, group: str, title: str | None = None, size: int = 15) -> None:
        if node_id not in nodes:
            nodes[node_id] = {
                "label": label,
                "group": group,
                "color": NODE_COLORS.get(group, "#95a5a6"),
                "title": title or label,
                "size": size,
            }

    query = """
    UNWIND $chunk_ids AS cid
    MATCH (c:Chunk {chunk_id: cid})
    OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
    WITH c, d,
         [(c)-[:IN_ARTICLE]->(a) | a] AS articles,
         [(c)-[:MENTIONS_SECTION]->(s) | s] AS sections,
         [(c)-[:MENTIONS]->(e) | e] AS entities,
         [(c)-[:IN_TOPIC]->(t) | t] AS topics
    RETURN c.chunk_id AS chunk_id,
           d.doc_id AS doc_id, d.source_label AS doc_label,
           articles, sections, entities, topics
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            rows = session.run(query, chunk_ids=chunk_ids)
            for row in rows:
                cid = row["chunk_id"]
                is_seed = cid in seed_ids
                group = "Chunk (vector seed)" if is_seed else "Chunk (KG-expanded)"
                chunk_node_id = f"chunk:{cid}"
                add_node(chunk_node_id, label=f"Chunk {cid[:8]}", group=group, title=f"chunk_id: {cid}", size=18 if is_seed else 13)

                doc_id = row["doc_id"]
                if doc_id:
                    doc_label = row["doc_label"] or doc_id
                    doc_node_id = f"doc:{doc_id}"
                    add_node(doc_node_id, label=doc_label[:40], group="Document", title=doc_label, size=24)
                    edges.append((doc_node_id, chunk_node_id, "HAS_CHUNK"))

                for a in row["articles"]:
                    art = a["article"]
                    art_node_id = f"article:{art}"
                    add_node(art_node_id, label=f"Article {art}", group="Article", size=20)
                    edges.append((chunk_node_id, art_node_id, "IN_ARTICLE"))

                for s in row["sections"]:
                    sec_node_id = f"section:{s['number']}:{s['doc_type']}"
                    add_node(sec_node_id, label=f"§{s['number']}", group="Section", size=16)
                    edges.append((chunk_node_id, sec_node_id, "MENTIONS_SECTION"))

                for e in row["entities"]:
                    ent_node_id = f"entity:{e['name']}:{e['type']}"
                    add_node(ent_node_id, label=e["name"][:30], group="Entity", title=f"{e['name']} ({e['type']})", size=11)
                    edges.append((chunk_node_id, ent_node_id, "MENTIONS"))

                for t in row["topics"]:
                    topic_label = t.get("level2") or t.get("level1") or t["path"]
                    topic_node_id = f"topic:{t['path']}"
                    add_node(topic_node_id, label=str(topic_label), group="Topic", title=t["path"], size=16)
                    edges.append((chunk_node_id, topic_node_id, "IN_TOPIC"))
    finally:
        driver.close()

    return nodes, edges


def build_kg_html(nodes: Dict[str, Dict[str, Any]], edges: List[Tuple[str, str, str]], height_px: int = 600) -> str:
    net = Network(height=f"{height_px}px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-4000, spring_length=140)
    for node_id, props in nodes.items():
        net.add_node(
            node_id,
            label=props["label"],
            title=props["title"],
            color=props["color"],
            size=props["size"],
        )
    for src, dst, rel in edges:
        net.add_edge(src, dst, title=rel, arrows="to")
    return net.generate_html(notebook=False)


def main() -> None:
    st.set_page_config(page_title="California Legal RAG", page_icon=PAGE_ICON, layout="wide")
    st.title("California Legal RAG UI")

    with st.sidebar:
        st.subheader("Settings")
        use_kg = st.toggle(
            "Vector + Knowledge Graph Expansion",
            value=True,
            help="Off = vector search only. On = also expand results with graph-connected chunks from Neo4j.",
        )
        top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
        kg_expand_k = st.slider("KG Expand K", min_value=1, max_value=30, value=12)
        kg_min_similarity = st.slider(
            "KG Min Similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum cosine similarity to the query for a KG-expanded chunk to be kept.",
        )

        doc_type = st.selectbox(
            "Doc Type",
            options=["(all)", "ca_constitution", "ca_education_code"],
            index=0,
        )
        article = st.text_input("Article (optional, e.g., I, IX)", "")
        edcode_section = st.text_input("EdCode Section (optional)", "")
        doc_id = st.text_input(
            "Doc ID (optional)",
            "",
            help=(
                "Filter to one document's chunks by doc_id. More reliable than "
                "Article/EdCode Section filters, which only tag the chunk "
                "containing the section-header text -- doc_id is set on every "
                "chunk of a document regardless of anchor-extraction success."
            ),
        )

        chroma_dir = st.text_input("Chroma Dir", "data/chroma")
        collection_name = st.text_input("Collection", DEFAULT_COLLECTION)
        embed_model_name = st.text_input("Embed Model", DEFAULT_EMBED_MODEL)
        openai_model = st.text_input("OpenAI Model", DEFAULT_OPENAI_MODEL)
        chunks_path = st.text_input(
            "Chunks Path",
            "data/chunks/chunks.jsonl",
            help="Used to auto-resolve Article/EdCode Section into a reliable doc_id filter (see doc_resolver.py).",
        )

        st.divider()
        show_kg_graph = st.toggle(
            "Show Knowledge Graph after answering",
            value=True,
            help="Fetches the Document/Article/Section/Entity/Topic neighborhood of the retrieved chunks from Neo4j and renders it inline. Requires NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD to be set.",
        )

    query = st.text_area(
        "Question",
        value="What does Article IX say about education?",
        height=90,
    )

    if st.button("Ask", type="primary"):
        loading = st.empty()
        try:
            loading.markdown(
                loading_indicator_html("Consulting the California Constitution &amp; Education Code"),
                unsafe_allow_html=True,
            )
            answer, sources, context, final_chunk_ids, seed_ids = run_query(
                query=query.strip(),
                top_k=top_k,
                chroma_dir=chroma_dir.strip(),
                collection_name=collection_name.strip(),
                embed_model_name=embed_model_name.strip(),
                openai_model=openai_model.strip(),
                doc_type=None if doc_type == "(all)" else doc_type,
                edcode_section=edcode_section.strip() or None,
                article=article.strip().upper() or None,
                doc_id=doc_id.strip() or None,
                use_kg=use_kg,
                kg_expand_k=kg_expand_k,
                kg_min_similarity=kg_min_similarity,
                chunks_path=chunks_path.strip(),
            )

            graph_nodes: Dict[str, Dict[str, Any]] = {}
            graph_edges: List[Tuple[str, str, str]] = []
            graph_error: str | None = None
            if show_kg_graph:
                loading.markdown(
                    loading_indicator_html("Mapping the knowledge graph"),
                    unsafe_allow_html=True,
                )
                try:
                    graph_nodes, graph_edges = fetch_subgraph(final_chunk_ids, seed_ids)
                except RuntimeError as kg_exc:
                    graph_error = str(kg_exc)

            # Cache everything so the "simplify graph" toggle (rendered below,
            # next to the graph) can flip the view on its own rerun without
            # re-calling OpenAI or Neo4j.
            st.session_state["last_result"] = {
                "answer": answer,
                "sources": sources,
                "context": context,
                "seed_count": len(seed_ids),
                "kg_expanded_count": len([c for c in final_chunk_ids if c not in seed_ids]),
                "graph_fetch_attempted": show_kg_graph,
                "graph_error": graph_error,
                "graph_nodes": graph_nodes,
                "graph_edges": graph_edges,
            }
            st.session_state.pop("last_error", None)
        except Exception as exc:
            st.session_state["last_error"] = str(exc)
            st.session_state.pop("last_result", None)
        finally:
            loading.empty()

    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    result = st.session_state.get("last_result")
    if result:
        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        if result["sources"]:
            st.dataframe(result["sources"], use_container_width=True)
        else:
            st.info("No citations detected in answer.")

        with st.expander("Retrieved Context"):
            st.text(result["context"])

        if show_kg_graph:
            st.subheader("Knowledge Graph")
            if not result["graph_fetch_attempted"]:
                st.info("Knowledge Graph was off when this answer was generated -- click Ask again to fetch it.")
            elif result["graph_error"]:
                st.warning(f"Knowledge graph view unavailable: {result['graph_error']}")
            elif not result["graph_nodes"]:
                st.info("No graph nodes found for the retrieved chunks (they may predate the full KG build).")
            else:
                simplify_kg_graph = st.toggle(
                    "Simplify graph (hide Entity nodes)",
                    value=False,
                    key="simplify_kg_graph",
                    help="Entity nodes (gray) are usually the majority of nodes and make the graph look busy. "
                    "Toggle this on for a cleaner Document -> Chunk -> Article/Section/Topic view -- same "
                    "underlying data, just Entities and their edges filtered out before rendering.",
                )

                nodes, edges = result["graph_nodes"], result["graph_edges"]
                full_node_count, full_edge_count = len(nodes), len(edges)
                if simplify_kg_graph:
                    nodes = {nid: props for nid, props in nodes.items() if props["group"] != "Entity"}
                    edges = [(s, d, r) for (s, d, r) in edges if s in nodes and d in nodes]

                shown_groups = {props["group"] for props in nodes.values()}
                legend = "  ".join(
                    f"<span style='color:{color}'>&#9679;</span> {group}"
                    for group, color in NODE_COLORS.items()
                    if group in shown_groups
                )
                st.markdown(legend, unsafe_allow_html=True)
                html = build_kg_html(nodes, edges)
                components.html(html, height=620, scrolling=True)

                if simplify_kg_graph:
                    st.caption(
                        f"Simplified view: {len(nodes)} nodes, {len(edges)} edges "
                        f"(full graph: {full_node_count} nodes, {full_edge_count} edges, "
                        f"{full_node_count - len(nodes)} Entity nodes hidden) -- "
                        f"{result['seed_count']} vector-seed chunk(s), "
                        f"{result['kg_expanded_count']} KG-expanded chunk(s)."
                    )
                else:
                    st.caption(
                        f"{len(nodes)} nodes, {len(edges)} edges -- "
                        f"{result['seed_count']} vector-seed chunk(s), "
                        f"{result['kg_expanded_count']} KG-expanded chunk(s)."
                    )


if __name__ == "__main__":
    main()
