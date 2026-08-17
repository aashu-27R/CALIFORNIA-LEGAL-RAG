"""
Compute Recall@k and MRR (Mean Reciprocal Rank) for existing validation
batch results, without re-running the RAG pipeline.

This reuses the doc_type/article/edcode_section columns already present in
the validation questions CSV as a lightweight "gold" relevance signal: for
each question, we know which source document(s) it *should* be about, and
we check whether the retrieved chunks (already saved in the results JSONL
from run_validation_batch.py) actually came from those document(s).

Ground-truth document identification deliberately avoids the per-chunk
"article" anchor field, which we found is only populated on the single
chunk containing a document's section-header text (a sliding-window
chunking artifact -- see step5_rag_answer.py's doc_id filter for the same
finding). Instead:
  - Constitution articles are identified from the PDF filename itself
    (e.g. "ARTICLE IX ECUCATION [...].pdf" -> "IX"), which has no such gap.
  - Education Code sections are identified from the "edcode_sections"
    anchor list, which is populated far more densely across chunks than
    "article" is.

Example:
  python src/compute_retrieval_metrics.py \
    --results-jsonl data/eval/results/validation_questions_20260811_182612_results.jsonl \
    --chunks-path data/chunks/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

from doc_resolver import build_resolution_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Recall@k / MRR from existing validation results")
    parser.add_argument("--results-jsonl", required=True, help="Path to a run_validation_batch.py *_results.jsonl file")
    parser.add_argument("--chunks-path", default="data/chunks/chunks.jsonl")
    parser.add_argument("--recall-ks", default="1,3,5", help="Comma-separated k values for Recall@k")
    parser.add_argument("--output", default=None, help="Optional path to write the summary JSON")
    return parser.parse_args()


def build_ground_truth_maps(chunks_path: Path):
    """Thin wrapper kept for backward compatibility -- see doc_resolver.build_resolution_maps."""
    return build_resolution_maps(chunks_path)


def gold_doc_ids_for_row(
    row: Dict[str, Any],
    article_to_docs: Dict[str, Set[str]],
    edcode_section_to_docs: Dict[str, Set[str]],
) -> Set[str]:
    doc_type = (row.get("doc_type") or "").strip()
    article = (row.get("article") or "").strip().upper()
    edcode_section = (row.get("edcode_section") or "").strip()

    if doc_type == "ca_constitution" and article:
        return set(article_to_docs.get(article, set()))
    if doc_type == "ca_education_code" and edcode_section:
        return set(edcode_section_to_docs.get(edcode_section, set()))
    return set()


def compute_metrics_for_row(
    final_chunk_ids: List[str],
    gold_docs: Set[str],
    chunk_to_doc: Dict[str, str],
    recall_ks: List[int],
) -> Dict[str, Any]:
    retrieved_docs_in_order = [chunk_to_doc.get(cid) for cid in final_chunk_ids]

    first_hit_rank = None
    for rank, doc_id in enumerate(retrieved_docs_in_order, start=1):
        if doc_id is not None and doc_id in gold_docs:
            first_hit_rank = rank
            break

    reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank else 0.0
    recall_at_k = {
        k: 1.0 if (first_hit_rank is not None and first_hit_rank <= k) else 0.0
        for k in recall_ks
    }
    return {
        "first_hit_rank": first_hit_rank,
        "reciprocal_rank": reciprocal_rank,
        "recall_at_k": recall_at_k,
    }


def main() -> None:
    args = parse_args()
    recall_ks = [int(x) for x in args.recall_ks.split(",") if x.strip()]

    chunk_to_doc, article_to_docs, edcode_section_to_docs = build_ground_truth_maps(Path(args.chunks_path))
    print(f"Ground truth coverage: {len(article_to_docs)} constitution articles, "
          f"{len(edcode_section_to_docs)} distinct Ed Code sections identified from filenames/anchors.")

    per_mode_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped_no_label = 0

    with Path(args.results_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                continue  # can't judge retrieval quality on a failed call

            gold_docs = gold_doc_ids_for_row(row, article_to_docs, edcode_section_to_docs)
            if not gold_docs:
                skipped_no_label += 1
                continue

            try:
                final_chunk_ids = json.loads(row.get("final_chunk_ids", "[]"))
            except json.JSONDecodeError:
                final_chunk_ids = []

            metrics = compute_metrics_for_row(final_chunk_ids, gold_docs, chunk_to_doc, recall_ks)
            metrics["question_id"] = row.get("question_id")
            metrics["mode"] = row.get("mode")
            per_mode_rows[row.get("mode", "unknown")].append(metrics)

    print(f"Judged questions: {sum(len(v) for v in per_mode_rows.values())} rows "
          f"(skipped {skipped_no_label} rows with no usable article/edcode_section label or errors)")
    print("")

    summary: Dict[str, Any] = {"recall_ks": recall_ks, "per_mode": {}}

    for mode, rows in per_mode_rows.items():
        n = len(rows)
        if n == 0:
            continue
        mrr = sum(r["reciprocal_rank"] for r in rows) / n
        recall_summary = {
            k: sum(r["recall_at_k"][k] for r in rows) / n
            for k in recall_ks
        }
        summary["per_mode"][mode] = {
            "n_questions": n,
            "mrr": round(mrr, 4),
            "recall_at_k": {str(k): round(v, 4) for k, v in recall_summary.items()},
        }

        print(f"=== Mode: {mode} (n={n}) ===")
        print(f"  MRR: {mrr:.4f}")
        for k in recall_ks:
            print(f"  Recall@{k}: {recall_summary[k]:.4f}")
        print("")

    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()
