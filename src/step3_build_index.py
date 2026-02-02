"""
STEP 3 — Build Chroma vector index from chunked legal docs.

Input : data/chunks/chunks.jsonl (from Step 2)
Output: data/chroma/ (persistent Chroma DB)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "ca_legal_chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma index for CA legal RAG")
    parser.add_argument("--in-path", default="data/chunks/chunks.jsonl")
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reset", action="store_true", help="Delete existing collection before indexing")
    return parser.parse_args()


def to_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    anchors = doc.get("anchors") or {}
    edcode_sections = anchors.get("edcode_sections") or []
    edcode_section = edcode_sections[0] if edcode_sections else None
    return {
        "doc_id": doc.get("doc_id"),
        "doc_type": doc.get("doc_type"),
        "rel_path": doc.get("rel_path"),
        "source_label": doc.get("source_label"),
        "chunk_index": doc.get("chunk_index"),
        "start_char": doc.get("start_char"),
        "end_char": doc.get("end_char"),
        "edcode_section": edcode_section,
        "anchors_json": json.dumps(anchors, ensure_ascii=False),
    }


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path)
    chroma_dir = Path(args.chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    client = chromadb.PersistentClient(path=str(chroma_dir))

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"Deleted collection: {args.collection}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(args.model)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    total = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids.append(rec["chunk_id"])
            docs.append(rec["text"])
            metas.append(to_metadata(rec))

            if len(ids) >= args.batch_size:
                embeddings = model.encode(docs, batch_size=args.batch_size, normalize_embeddings=True)
                collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
                total += len(ids)
                print(f"Indexed {total} chunks...")
                ids, docs, metas = [], [], []

    if ids:
        embeddings = model.encode(docs, batch_size=args.batch_size, normalize_embeddings=True)
        collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
        total += len(ids)

    print("✅ Step 3 complete")
    print(f"Collection: {args.collection}")
    print(f"Indexed chunks: {total}")
    print(f"Chroma dir: {chroma_dir}")


if __name__ == "__main__":
    main()
