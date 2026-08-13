"""
Batch validation runner for California Legal RAG.

Reads a CSV of questions plus optional retrieval filters, runs the existing
RAG pipeline in vector and/or KG modes, and writes detailed outputs to disk.

Example:
  python src/run_validation_batch.py \
    --questions-csv data/eval/validation_questions.csv \
    --modes both \
    --top-k 3 \
    --kg-expand-k 4 \
    --openai-model gpt-5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from step5_rag_answer import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OPENAI_MODEL,
    build_where,
    format_context,
    kg_expand_chunk_ids,
)


VALID_MODES = {"vector", "kg", "both"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch validation for California Legal RAG")
    parser.add_argument("--questions-csv", required=True, help="CSV with at least a 'question' column")
    parser.add_argument("--modes", default="both", choices=sorted(VALID_MODES))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--kg-expand-k", type=int, default=4)
    parser.add_argument(
        "--kg-min-similarity",
        type=float,
        default=0.25,
        help="Min cosine similarity (to the query) for a KG-expanded chunk to be kept, 0-1",
    )
    parser.add_argument("--chroma-dir", default="data/chroma")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--output-dir", default="data/eval/results")
    parser.add_argument("--run-name", default=None, help="Optional prefix for output files")
    return parser.parse_args()


def normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def load_questions(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions CSV not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError("Questions CSV must include a 'question' column.")

        rows: List[Dict[str, str]] = []
        for idx, row in enumerate(reader, 1):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            normalized = {k: (v or "").strip() for k, v in row.items() if k}
            normalized.setdefault("question_id", str(idx))
            normalized["question"] = question
            rows.append(normalized)

    if not rows:
        raise ValueError("No non-empty questions found in CSV.")
    return rows


def ensure_runtime_requirements(args: argparse.Namespace) -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running this script.")

    chroma_dir = Path(args.chroma_dir)
    if not chroma_dir.exists():
        raise FileNotFoundError(f"Missing Chroma dir: {chroma_dir}")


def mode_list(modes: str) -> List[str]:
    if modes == "both":
        return ["vector", "kg"]
    return [modes]


def query_collection(
    collection: Any,
    query_emb: Any,
    top_k: int,
    doc_type: str | None,
    edcode_section: str | None,
    article: str | None,
    doc_id: str | None = None,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    where = build_where(doc_type, edcode_section, article, doc_id)
    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return (
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("ids", [[]])[0],
    )


def dedupe_rows(
    docs: List[str],
    metas: List[Dict[str, Any]],
    ids: List[str],
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    seen = set()
    out_docs: List[str] = []
    out_metas: List[Dict[str, Any]] = []
    out_ids: List[str] = []

    for doc, meta, chunk_id in zip(docs, metas, ids):
        key = (
            chunk_id,
            meta.get("doc_id"),
            meta.get("chunk_index"),
            meta.get("start_char"),
            meta.get("end_char"),
        )
        if key in seen:
            continue
        seen.add(key)
        out_docs.append(doc)
        out_metas.append(meta)
        out_ids.append(chunk_id)

    return out_docs, out_metas, out_ids


def retrieve_context(
    collection: Any,
    embed_model: SentenceTransformer,
    question: str,
    category: str | None,
    top_k: int,
    doc_type: str | None,
    edcode_section: str | None,
    article: str | None,
    use_kg: bool,
    kg_expand_k: int,
    kg_min_similarity: float = 0.25,
    doc_id: str | None = None,
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    query_emb = embed_model.encode([question], normalize_embeddings=True)
    normalized_category = (category or "").strip().lower()

    # Cross-source questions benefit from guaranteed coverage from both corpora.
    if normalized_category == "cross_source":
        constitution_k = max(1, top_k // 2)
        education_k = max(1, top_k - constitution_k)

        c_docs, c_metas, c_ids = query_collection(
            collection=collection,
            query_emb=query_emb,
            top_k=constitution_k,
            doc_type="ca_constitution",
            edcode_section=None,
            article=article,
        )
        e_docs, e_metas, e_ids = query_collection(
            collection=collection,
            query_emb=query_emb,
            top_k=education_k,
            doc_type="ca_education_code",
            edcode_section=edcode_section,
            article=None,
        )
        docs, metas, ids = dedupe_rows(c_docs + e_docs, c_metas + e_metas, c_ids + e_ids)
    else:
        docs, metas, ids = query_collection(
            collection=collection,
            query_emb=query_emb,
            top_k=top_k,
            doc_type=doc_type,
            edcode_section=edcode_section,
            article=article,
            doc_id=doc_id,
        )

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
            extra_docs = extra.get("documents", [])
            extra_metas = extra.get("metadatas", [])
            docs, metas, ids = dedupe_rows(
                docs + extra_docs,
                metas + extra_metas,
                ids + extra_ids,
            )

    return docs, metas, ids


def generate_answer(
    openai_client: OpenAI,
    question: str,
    docs: List[str],
    metas: List[Dict[str, Any]],
    openai_model: str,
) -> Tuple[str, List[Dict[str, Any]], str]:
    context = format_context(docs, metas)
    system_prompt = (
        "You are a legal RAG assistant for California Education Code (Title 1) "
        "and the California Constitution. Answer strictly using the provided context. "
        "Cite sources as [#] using the numbered context blocks. If the answer is not "
        "in the context, say you don't have enough information."
    )

    response = openai_client.responses.create(
        model=openai_model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
        ],
    )
    answer = (getattr(response, "output_text", None) or str(response)).strip()

    cited = set(int(x) for x in re.findall(r"\[(\d+)\]", answer))
    sources: List[Dict[str, Any]] = []
    for idx, meta in enumerate(metas, 1):
        if cited and idx not in cited:
            continue
        sources.append(
            {
                "idx": idx,
                "label": meta.get("source_label", "source"),
                "rel_path": meta.get("rel_path", ""),
                "anchors": meta.get("anchors_json", ""),
            }
        )

    return answer, sources, context


def retrieve_context_with_debug(
    collection: Any,
    embed_model: SentenceTransformer,
    question: str,
    category: str | None,
    top_k: int,
    doc_type: str | None,
    edcode_section: str | None,
    article: str | None,
    use_kg: bool,
    kg_expand_k: int,
    kg_min_similarity: float = 0.25,
    doc_id: str | None = None,
) -> Tuple[List[str], List[Dict[str, Any]], List[str], Dict[str, Any]]:
    query_emb = embed_model.encode([question], normalize_embeddings=True)
    normalized_category = (category or "").strip().lower()
    retrieval_strategy = "single_query"

    if normalized_category == "cross_source":
        retrieval_strategy = "cross_source_split"
        constitution_k = max(1, top_k // 2)
        education_k = max(1, top_k - constitution_k)

        c_docs, c_metas, c_ids = query_collection(
            collection=collection,
            query_emb=query_emb,
            top_k=constitution_k,
            doc_type="ca_constitution",
            edcode_section=None,
            article=article,
        )
        e_docs, e_metas, e_ids = query_collection(
            collection=collection,
            query_emb=query_emb,
            top_k=education_k,
            doc_type="ca_education_code",
            edcode_section=edcode_section,
            article=None,
        )
        docs, metas, ids = dedupe_rows(c_docs + e_docs, c_metas + e_metas, c_ids + e_ids)
    else:
        docs, metas, ids = query_collection(
            collection=collection,
            query_emb=query_emb,
            top_k=top_k,
            doc_type=doc_type,
            edcode_section=edcode_section,
            article=article,
            doc_id=doc_id,
        )

    seed_ids = list(ids)
    kg_extra_ids: List[str] = []
    if use_kg and ids:
        kg_extra_ids = kg_expand_chunk_ids(
            ids,
            kg_expand_k,
            query_embedding=query_emb,
            collection=collection,
            min_similarity=kg_min_similarity,
        )
        if kg_extra_ids:
            extra = collection.get(ids=kg_extra_ids, include=["documents", "metadatas"])
            extra_docs = extra.get("documents", [])
            extra_metas = extra.get("metadatas", [])
            docs, metas, ids = dedupe_rows(
                docs + extra_docs,
                metas + extra_metas,
                ids + kg_extra_ids,
            )

    debug = {
        "retrieval_strategy": retrieval_strategy,
        "seed_chunk_ids": seed_ids,
        "kg_extra_chunk_ids": kg_extra_ids,
        "seed_chunk_count": len(seed_ids),
        "kg_extra_chunk_count": len(kg_extra_ids),
        "final_chunk_count": len(ids),
    }
    return docs, metas, ids, debug


def main() -> None:
    args = parse_args()
    ensure_runtime_requirements(args)

    questions_path = Path(args.questions_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_questions(questions_path)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or questions_path.stem
    run_prefix = f"{run_name}_{run_stamp}"

    client = chromadb.PersistentClient(path=str(Path(args.chroma_dir)))
    collection = client.get_collection(name=args.collection)
    embed_model = SentenceTransformer(args.embed_model)
    openai_client = OpenAI()

    results_path = output_dir / f"{run_prefix}_results.csv"
    jsonl_path = output_dir / f"{run_prefix}_results.jsonl"
    summary_path = output_dir / f"{run_prefix}_summary.json"

    modes = mode_list(args.modes)
    rows_out: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "run_name": run_name,
        "run_timestamp_local": run_stamp,
        "questions_csv": str(questions_path),
        "question_count": len(questions),
        "modes": modes,
        "top_k": args.top_k,
        "kg_expand_k": args.kg_expand_k,
        "kg_min_similarity": args.kg_min_similarity,
        "openai_model": args.openai_model,
        "results_csv": str(results_path),
        "results_jsonl": str(jsonl_path),
        "per_mode": {},
    }

    for mode in modes:
        summary["per_mode"][mode] = {"rows": 0, "errors": 0}

    for q_idx, question_row in enumerate(questions, 1):
        question = question_row["question"]
        category = normalize_optional(question_row.get("category"))
        doc_type = normalize_optional(question_row.get("doc_type"))
        edcode_section = normalize_optional(question_row.get("edcode_section"))
        article = normalize_optional(question_row.get("article"))
        doc_id = normalize_optional(question_row.get("doc_id"))

        print(f"[{q_idx}/{len(questions)}] {question}")

        for mode in modes:
            use_kg = mode == "kg"
            try:
                docs, metas, final_ids, debug = retrieve_context_with_debug(
                    collection=collection,
                    embed_model=embed_model,
                    question=question,
                    category=category,
                    top_k=args.top_k,
                    doc_type=doc_type,
                    edcode_section=edcode_section,
                    article=article,
                    doc_id=doc_id,
                    use_kg=use_kg,
                    kg_expand_k=args.kg_expand_k,
                    kg_min_similarity=args.kg_min_similarity,
                )
                answer, sources, context = generate_answer(
                    openai_client=openai_client,
                    question=question,
                    docs=docs,
                    metas=metas,
                    openai_model=args.openai_model,
                )
                error = ""
            except Exception as exc:
                docs, metas, final_ids, sources, context = [], [], [], [], ""
                answer = ""
                error = str(exc)
                debug = {
                    "retrieval_strategy": "error",
                    "seed_chunk_ids": [],
                    "kg_extra_chunk_ids": [],
                    "seed_chunk_count": 0,
                    "kg_extra_chunk_count": 0,
                    "final_chunk_count": 0,
                }
                summary["per_mode"][mode]["errors"] += 1

            output_row: Dict[str, Any] = dict(question_row)
            output_row.update(
                {
                    "mode": mode,
                    "top_k": args.top_k,
                    "kg_expand_k": args.kg_expand_k if use_kg else 0,
                    "retrieved_chunks": len(docs),
                    "retrieval_strategy": debug["retrieval_strategy"],
                    "seed_chunk_ids": json.dumps(debug["seed_chunk_ids"], ensure_ascii=False),
                    "kg_extra_chunk_ids": json.dumps(debug["kg_extra_chunk_ids"], ensure_ascii=False),
                    "seed_chunk_count": debug["seed_chunk_count"],
                    "kg_extra_chunk_count": debug["kg_extra_chunk_count"],
                    "final_chunk_ids": json.dumps(final_ids, ensure_ascii=False),
                    "answer": answer,
                    "sources_json": json.dumps(sources, ensure_ascii=False),
                    "context_chars": len(context),
                    "error": error,
                }
            )
            rows_out.append(output_row)
            summary["per_mode"][mode]["rows"] += 1

    fieldnames: List[str] = []
    seen = set()
    for row in rows_out:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("Validation batch complete")
    print(f"Questions: {len(questions)}")
    print(f"Modes: {', '.join(modes)}")
    print(f"Results CSV: {results_path}")
    print(f"Results JSONL: {jsonl_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
