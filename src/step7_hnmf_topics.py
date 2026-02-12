"""
STEP 7 — HNMF-style topic hierarchy for legal chunks.

This is a practical approximation of HNMFk using:
1) TF-IDF features
2) NMF topic decomposition
3) Automatic k selection via silhouette score
4) Optional second-level split per top-level topic

Input : data/chunks/chunks.jsonl
Output: data/topics/chunk_topics.jsonl
        data/topics/topics_summary.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


@dataclass
class TopicModelResult:
    labels: np.ndarray
    topic_terms: Dict[int, List[str]]
    k: int
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HNMF-style topic hierarchy for legal chunks")
    parser.add_argument("--in-path", default="data/chunks/chunks.jsonl")
    parser.add_argument("--out-dir", default="data/topics")
    parser.add_argument("--min-k", type=int, default=4)
    parser.add_argument("--max-k", type=int, default=16)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--depth", type=int, default=2, choices=[1, 2])
    parser.add_argument("--min-child-size", type=int, default=40)
    parser.add_argument("--top-terms", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_chunks(in_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def fit_nmf_topics(
    texts: List[str],
    min_k: int,
    max_k: int,
    max_features: int,
    top_terms: int,
    seed: int,
) -> TopicModelResult:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    X = vectorizer.fit_transform(texts)
    if X.shape[0] < 3:
        labels = np.zeros(X.shape[0], dtype=int)
        return TopicModelResult(labels=labels, topic_terms={0: []}, k=1, score=-1.0)

    best = None
    best_payload = None

    lo = max(2, min_k)
    hi = min(max_k, max(2, X.shape[0] - 1))

    for k in range(lo, hi + 1):
        try:
            nmf = NMF(
                n_components=k,
                init="nndsvda",
                random_state=seed,
                max_iter=300,
                alpha_W=0.0,
                alpha_H=0.0,
            )
            W = nmf.fit_transform(X)
            labels = W.argmax(axis=1)

            # Need >=2 members per cluster for silhouette in cosine metric.
            km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
            km_labels = km.fit_predict(W)
            if len(np.unique(km_labels)) < 2:
                continue
            score = silhouette_score(W, km_labels, metric="cosine")

            if best is None or score > best:
                terms = vectorizer.get_feature_names_out()
                topic_terms: Dict[int, List[str]] = {}
                for tid in range(k):
                    weights = nmf.components_[tid]
                    idx = np.argsort(weights)[-top_terms:][::-1]
                    topic_terms[tid] = [str(terms[i]) for i in idx]
                best = score
                best_payload = (labels, topic_terms, k, score)
        except Exception:
            continue

    if best_payload is None:
        labels = np.zeros(X.shape[0], dtype=int)
        return TopicModelResult(labels=labels, topic_terms={0: []}, k=1, score=-1.0)

    return TopicModelResult(
        labels=best_payload[0],
        topic_terms=best_payload[1],
        k=best_payload[2],
        score=best_payload[3],
    )


def build_level1(
    chunks: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    texts = [c.get("text", "") for c in chunks]
    model = fit_nmf_topics(
        texts=texts,
        min_k=args.min_k,
        max_k=args.max_k,
        max_features=args.max_features,
        top_terms=args.top_terms,
        seed=args.seed,
    )

    assigned: List[Dict[str, Any]] = []
    for idx, c in enumerate(chunks):
        t1 = int(model.labels[idx])
        topic_path = f"T{t1:02d}"
        assigned.append(
            {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "doc_type": c.get("doc_type"),
                "rel_path": c.get("rel_path"),
                "topic_level1": topic_path,
                "topic_level2": None,
                "topic_path": topic_path,
            }
        )

    summary = {
        "level1_k": model.k,
        "level1_silhouette": model.score,
        "level1_terms": {f"T{tid:02d}": terms for tid, terms in model.topic_terms.items()},
    }
    return assigned, summary


def build_level2(
    chunks: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    by_t1: Dict[str, List[int]] = {}
    for i, a in enumerate(assignments):
        by_t1.setdefault(a["topic_level1"], []).append(i)

    level2_summary: Dict[str, Any] = {}
    for t1, idxs in by_t1.items():
        if len(idxs) < args.min_child_size:
            continue
        subset_texts = [chunks[i].get("text", "") for i in idxs]
        model = fit_nmf_topics(
            texts=subset_texts,
            min_k=2,
            max_k=min(8, args.max_k),
            max_features=args.max_features,
            top_terms=args.top_terms,
            seed=args.seed,
        )
        if model.k <= 1:
            continue
        level2_summary[t1] = {
            "k": model.k,
            "silhouette": model.score,
            "terms": {f"{t1}/S{tid:02d}": terms for tid, terms in model.topic_terms.items()},
        }
        for local_i, global_i in enumerate(idxs):
            s = int(model.labels[local_i])
            t2 = f"S{s:02d}"
            assignments[global_i]["topic_level2"] = t2
            assignments[global_i]["topic_path"] = f"{t1}/{t2}"

    return level2_summary


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    chunks = load_chunks(in_path)
    if not chunks:
        raise RuntimeError("No chunks found in input.")

    assignments, summary = build_level1(chunks, args)
    if args.depth == 2:
        summary["level2"] = build_level2(chunks, assignments, args)
    else:
        summary["level2"] = {}

    out_topics = out_dir / "chunk_topics.jsonl"
    with out_topics.open("w", encoding="utf-8") as f:
        for row in assignments:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary["input_chunks"] = len(chunks)
    summary["depth"] = args.depth
    summary["out_topics"] = str(out_topics)
    out_summary = out_dir / "topics_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("✅ Step 7 complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
