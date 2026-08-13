"""
STEP 5 — RAG answer generation with OpenAI.

Retrieves top-k chunks from Chroma, then asks an OpenAI model to answer
using the provided context, with citations.

Example:
  python src/step5_rag_answer.py --query "What does Article IX say about education?" --doc-type ca_constitution
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from neo4j import GraphDatabase


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "ca_legal_chunks"
DEFAULT_OPENAI_MODEL = "gpt-5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG answer generation with OpenAI")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--doc-type", default=None, help="Filter: ca_constitution or ca_education_code")
    parser.add_argument("--edcode-section", default=None, help="Filter by Education Code section number (e.g., 48900)")
    parser.add_argument("--article", default=None, help="Filter by Constitution article (e.g., IX)")
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Filter to a single document's chunks by doc_id (reliable even when article/section anchors are missing)",
    )
    parser.add_argument("--use-kg", action="store_true", help="Expand retrieved chunks with Neo4j graph neighbors")
    parser.add_argument("--kg-expand-k", type=int, default=12, help="Max neighbor chunk ids to add from KG")
    parser.add_argument(
        "--kg-min-similarity",
        type=float,
        default=0.25,
        help="Min cosine similarity (to the query) for a KG-expanded chunk to be kept, 0-1",
    )
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context blocks before the answer")
    return parser.parse_args()


def build_where(
    doc_type: str | None,
    edcode_section: str | None,
    article: str | None,
    doc_id: str | None = None,
) -> Dict[str, Any] | None:
    clauses = []
    if doc_type:
        clauses.append({"doc_type": doc_type})
    if edcode_section:
        clauses.append({"edcode_section": str(edcode_section)})
    if article:
        clauses.append({"article": str(article).upper()})
    if doc_id:
        # More reliable than "article"/"edcode_section" anchors, which are
        # only tagged on the chunk containing the literal header text (a
        # sliding-window chunking artifact) -- doc_id is set on every chunk
        # of a document regardless of anchor-extraction success, so this
        # guarantees complete recall for "give me this whole document" needs.
        clauses.append({"doc_id": doc_id})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def format_context(docs: List[str], metas: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        label = meta.get("source_label", "source")
        rel = meta.get("rel_path", "")
        anchors = meta.get("anchors_json", "")
        blocks.append(
            f"[{i}] {label}\n"
            f"rel_path: {rel}\n"
            f"anchors: {anchors}\n"
            f"text:\n{doc}\n"
        )
    return "\n---\n".join(blocks)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set.")
    return value


def kg_expand_chunk_ids(
    seed_chunk_ids: List[str],
    expand_k: int,
    query_embedding: Any = None,
    collection: Any = None,
    min_similarity: float = 0.25,
) -> List[str]:
    """
    Expand retrieval set using Neo4j graph proximity.
    Traversal paths:
      Chunk -> Article <- Chunk
      Chunk -> Section <- Chunk
      Chunk -> Entity <- Chunk
      Chunk -> Topic <- Chunk

    Legacy behavior (query_embedding/collection omitted): candidates are
    ranked purely by how many graph paths connect them to the seed chunks.
    This is noisy in practice — generic, high-degree entities/sections (e.g.
    "Legislature", "State") connect a chunk to huge numbers of topically
    unrelated neighbors, so path-count alone tends to surface irrelevant
    chunks just because they share a common bureaucratic term with a seed.

    When query_embedding and collection ARE provided, this pulls a wider
    structural candidate pool from the graph (still ranked by path-count,
    just a bigger pool: expand_k * 5), then re-ranks that pool by cosine
    similarity between each candidate's stored Chroma embedding and the
    query embedding, keeping only chunks at or above min_similarity. This
    filters expansion down to chunks that are both graph-connected AND
    actually about the same thing as the query — candidates that are only
    connected via generic entities get dropped instead of padding out the
    result to expand_k regardless of relevance.
    """
    if not seed_chunk_ids or expand_k <= 0:
        return []

    uri = require_env("NEO4J_URI")
    user = require_env("NEO4J_USER")
    password = require_env("NEO4J_PASSWORD")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    rerank_by_similarity = query_embedding is not None and collection is not None
    pool_size = max(expand_k * 5, 20) if rerank_by_similarity else expand_k

    # Keep a lightweight scoring so chunks seen via multiple paths rank higher
    # within the structural candidate pool.
    scores: Dict[str, int] = {}
    query = """
    UNWIND $seed_ids AS sid
    MATCH (s:Chunk {chunk_id: sid})
    OPTIONAL MATCH (s)-[:IN_ARTICLE]->(:Article)<-[:IN_ARTICLE]-(c1:Chunk)
    OPTIONAL MATCH (s)-[:MENTIONS_SECTION]->(:Section)<-[:MENTIONS_SECTION]-(c2:Chunk)
    OPTIONAL MATCH (s)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(c3:Chunk)
    OPTIONAL MATCH (s)-[:IN_TOPIC]->(:Topic)<-[:IN_TOPIC]-(c4:Chunk)
    WITH sid,
         collect(DISTINCT c1.chunk_id) + collect(DISTINCT c2.chunk_id) +
         collect(DISTINCT c3.chunk_id) + collect(DISTINCT c4.chunk_id) AS nbrs
    UNWIND nbrs AS nid
    WITH nid WHERE nid IS NOT NULL AND NOT nid IN $seed_ids
    RETURN nid AS chunk_id, count(*) AS score
    ORDER BY score DESC
    LIMIT $limit
    """
    try:
        with driver.session() as session:
            rows = session.run(query, seed_ids=seed_chunk_ids, limit=pool_size)
            for row in rows:
                cid = row.get("chunk_id")
                if cid:
                    scores[cid] = int(row.get("score", 1))
    finally:
        driver.close()

    if not scores:
        return []

    if not rerank_by_similarity:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:expand_k]]

    candidate_ids = list(scores.keys())
    fetched = collection.get(ids=candidate_ids, include=["embeddings"])
    fetched_ids = fetched.get("ids", []) or []
    fetched_embs = fetched.get("embeddings", [])
    if fetched_embs is None:
        fetched_embs = []

    query_vec = np.asarray(query_embedding[0], dtype=float)
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    similarities: List[tuple] = []
    for cid, emb in zip(fetched_ids, fetched_embs):
        if emb is None:
            continue
        vec = np.asarray(emb, dtype=float)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        sim = float(np.dot(query_vec, vec))
        if sim >= min_similarity:
            similarities.append((cid, sim))

    # Ties broken by original graph path-count as a secondary signal.
    similarities.sort(key=lambda x: (x[1], scores.get(x[0], 0)), reverse=True)
    return [cid for cid, _ in similarities[:expand_k]]


def main() -> None:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running this script.")

    chroma_dir = Path(args.chroma_dir)
    if not chroma_dir.exists():
        raise FileNotFoundError(f"Missing Chroma dir: {chroma_dir}")

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name=args.collection)

    embed_model = SentenceTransformer(args.embed_model)
    query_emb = embed_model.encode([args.query], normalize_embeddings=True)

    where = build_where(args.doc_type, args.edcode_section, args.article, args.doc_id)

    results = collection.query(
        query_embeddings=query_emb,
        n_results=args.top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if args.use_kg and ids:
        extra_ids = kg_expand_chunk_ids(
            ids,
            args.kg_expand_k,
            query_embedding=query_emb,
            collection=collection,
            min_similarity=args.kg_min_similarity,
        )
        if extra_ids:
            extra = collection.get(ids=extra_ids, include=["documents", "metadatas"])
            extra_docs = extra.get("documents", [])
            extra_metas = extra.get("metadatas", [])
            # Append only valid rows, deduplicating by rel_path+start/end where available.
            seen = set()
            merged_docs: List[str] = []
            merged_metas: List[Dict[str, Any]] = []
            for d, m in list(zip(docs, metas)) + list(zip(extra_docs, extra_metas)):
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
            docs, metas = merged_docs, merged_metas

    context = format_context(docs, metas)
    if args.show_context:
        print("Retrieved Context:\n")
        print(context)
        print("\n====================\n")

    system = (
        "You are a legal RAG assistant for California Education Code (Title 1) "
        "and the California Constitution. Answer strictly using the provided context. "
        "Cite sources as [#] using the numbered context blocks. If the answer is not "
        "in the context, say you don't have enough information."
    )

    user = f"Question: {args.query}\n\nContext:\n{context}"

    openai_client = OpenAI()
    response = openai_client.responses.create(
        model=args.openai_model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    answer = (getattr(response, "output_text", None) or str(response)).strip()

    # Only show sources that are actually cited like [1], [2], etc.
    cited = set(int(x) for x in re.findall(r"\[(\d+)\]", answer))

    print("Answer:\n")
    print(answer)
    print("\nSources:")
    for i, meta in enumerate(metas, 1):
        if cited and i not in cited:
            continue
        label = meta.get("source_label", "source")
        rel = meta.get("rel_path", "")
        anchors = meta.get("anchors_json", "")
        print(f"[{i}] {label} | {rel} | {anchors}")


if __name__ == "__main__":
    main()
