"""
STEP 4 — Query Chroma index for CA legal RAG.

Example:
  python src/step4_query_rag.py --query "What does Ed Code say about suspension for defiance?" --top-k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "ca_legal_chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Chroma index for CA legal RAG")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--doc-type", default=None, help="Filter: ca_constitution or ca_education_code")
    parser.add_argument("--edcode-section", default=None, help="Filter by Education Code section number (e.g., 48900)")
    parser.add_argument("--article", default=None, help="Filter by Constitution article (e.g., IX)")
    parser.add_argument(
        "--must-contain",
        default=None,
        help="Comma-separated keywords that must appear in the document (e.g., '48900,defiance,suspension')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chroma_dir = Path(args.chroma_dir)
    if not chroma_dir.exists():
        raise FileNotFoundError(f"Missing Chroma dir: {chroma_dir}")

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name=args.collection)

    model = SentenceTransformer(args.model)
    query_emb = model.encode([args.query], normalize_embeddings=True)

    where_clauses = []
    if args.doc_type:
        where_clauses.append({"doc_type": args.doc_type})
    if args.edcode_section:
        where_clauses.append({"edcode_section": str(args.edcode_section)})
    if args.article:
        where_clauses.append({"article": str(args.article).upper()})

    if not where_clauses:
        where = None
    elif len(where_clauses) == 1:
        where = where_clauses[0]
    else:
        where = {"$and": where_clauses}

    where_document = None
    if args.must_contain:
        terms = [t.strip() for t in args.must_contain.split(",") if t.strip()]
        if terms:
            if len(terms) == 1:
                where_document = {"$contains": terms[0]}
            else:
                where_document = {"$and": [{"$contains": t} for t in terms]}

    results = collection.query(
        query_embeddings=query_emb,
        n_results=args.top_k,
        where=where,
        where_document=where_document,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    print("Query:", args.query)
    print(f"Top-{args.top_k} results:")
    print("")

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        anchors = meta.get("anchors_json", "")
        anchors_obj: Dict[str, Any] = {}
        if anchors:
            try:
                anchors_obj = json.loads(anchors)
            except Exception:
                anchors_obj = {"raw": anchors}

        snippet = doc[:500].replace("\n", " ").strip()
        if len(doc) > 500:
            snippet += " ..."

        print(f"{i}. {meta.get('source_label')}")
        print(f"   rel_path: {meta.get('rel_path')}")
        print(f"   doc_type: {meta.get('doc_type')}  distance: {dist:.4f}")
        if anchors_obj:
            print(f"   anchors: {anchors_obj}")
        print(f"   snippet: {snippet}")
        print("")


if __name__ == "__main__":
    main()
