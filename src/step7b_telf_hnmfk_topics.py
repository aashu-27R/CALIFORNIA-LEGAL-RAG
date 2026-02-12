"""
STEP 7B — Optional TELF HNMFk integration (experimental adapter).

Why this file exists:
- Your current Step 7 uses an NMF-based approximation.
- This script adds an optional path to use TELF HNMFk when the package is available.

Important:
- TELF APIs can vary by version.
- This adapter tries common method patterns and writes the same output schema as step7.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chunk topics using TELF HNMFk (experimental)")
    parser.add_argument("--in-path", default="data/chunks/chunks.jsonl")
    parser.add_argument("--out-dir", default="data/topics_telf")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-k", type=int, default=4)
    parser.add_argument("--max-k", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_chunks(in_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def import_hnmfk_class():
    candidates = [
        ("TELF.factorization", "HNMFk"),
        ("telf.factorization", "HNMFk"),
        ("TELF.factorization", "NMFk"),
        ("telf.factorization", "NMFk"),
    ]
    for module_name, class_name in candidates:
        try:
            mod = __import__(module_name, fromlist=[class_name])
            cls = getattr(mod, class_name, None)
            if cls is not None:
                return cls, f"{module_name}.{class_name}"
        except Exception:
            continue
    raise RuntimeError(
        "Could not import TELF HNMFk/NMFk. Install TELF and verify API for your version.\n"
        "Tried: TELF.factorization.HNMFk, telf.factorization.HNMFk, TELF.factorization.NMFk, telf.factorization.NMFk"
    )


def instantiate_model(model_cls, args: argparse.Namespace):
    sig = inspect.signature(model_cls)
    kwargs: Dict[str, Any] = {}
    # Best-effort constructor kwargs; only pass if supported by current TELF version.
    maybe = {
        "k_range": list(range(args.min_k, args.max_k + 1)),
        "K": list(range(args.min_k, args.max_k + 1)),
        "n_components_range": list(range(args.min_k, args.max_k + 1)),
        "max_depth": args.depth,
        "depth": args.depth,
        "random_state": args.seed,
        "seed": args.seed,
    }
    for k, v in maybe.items():
        if k in sig.parameters:
            kwargs[k] = v
    return model_cls(**kwargs)


def run_model(model, X):
    # Common patterns across libraries.
    if hasattr(model, "fit_transform"):
        return model.fit_transform(X)
    if hasattr(model, "fit"):
        return model.fit(X)
    raise RuntimeError("TELF model has neither fit_transform nor fit.")


def extract_labels(output: Any, n_rows: int) -> np.ndarray:
    # Try common output formats.
    if isinstance(output, dict):
        for key in ("labels", "cluster_labels", "topic_labels", "doc_labels"):
            if key in output:
                vals = np.asarray(output[key])
                if vals.shape[0] == n_rows:
                    return vals.astype(int)
    if isinstance(output, (list, tuple)):
        for item in output:
            arr = np.asarray(item)
            if arr.ndim == 1 and arr.shape[0] == n_rows:
                return arr.astype(int)
            if arr.ndim == 2 and arr.shape[0] == n_rows:
                return arr.argmax(axis=1).astype(int)
    arr = np.asarray(output)
    if arr.ndim == 1 and arr.shape[0] == n_rows:
        return arr.astype(int)
    if arr.ndim == 2 and arr.shape[0] == n_rows:
        return arr.argmax(axis=1).astype(int)
    raise RuntimeError(
        "Could not infer document labels from TELF output. "
        "Inspect the output structure for your TELF version and adapt extract_labels()."
    )


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    chunks = load_chunks(in_path)
    texts = [c.get("text", "") for c in chunks]
    if not texts:
        raise RuntimeError("No chunks found.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=args.max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    X = vectorizer.fit_transform(texts)

    model_cls, model_path = import_hnmfk_class()
    model = instantiate_model(model_cls, args)
    output = run_model(model, X)
    labels = extract_labels(output, n_rows=len(chunks))

    rows = []
    for i, c in enumerate(chunks):
        t1 = f"T{int(labels[i]):02d}"
        rows.append(
            {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "doc_type": c.get("doc_type"),
                "rel_path": c.get("rel_path"),
                "topic_level1": t1,
                "topic_level2": None,
                "topic_path": t1,
            }
        )

    out_topics = out_dir / "chunk_topics.jsonl"
    with out_topics.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_chunks": len(chunks),
        "model_class": model_path,
        "note": "Experimental TELF adapter. Validate topic quality and API compatibility for your TELF version.",
        "out_topics": str(out_topics),
    }
    (out_dir / "topics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("✅ Step 7B complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
