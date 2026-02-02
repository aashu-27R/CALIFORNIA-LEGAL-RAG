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
from pathlib import Path
from typing import Dict, Any, List

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI


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
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context blocks before the answer")
    return parser.parse_args()


def build_where(doc_type: str | None, edcode_section: str | None, article: str | None) -> Dict[str, Any] | None:
    clauses = []
    if doc_type:
        clauses.append({"doc_type": doc_type})
    if edcode_section:
        clauses.append({"edcode_section": str(edcode_section)})
    if article:
        clauses.append({"article": str(article).upper()})

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

    where = build_where(args.doc_type, args.edcode_section, args.article)

    results = collection.query(
        query_embeddings=query_emb,
        n_results=args.top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

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
    cited = set(int(x) for x in __import__("re").findall(r"\\[(\\d+)\\]", answer))

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
