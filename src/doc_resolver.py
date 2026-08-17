"""
Shared logic for resolving article/edcode_section labels to reliable
doc_id(s), bypassing the buggy per-chunk "article"/"edcode_section" anchor
metadata (only populated on the chunk containing a document's literal
section-header text -- a sliding-window chunking artifact, confirmed via
direct inspection of chunks.jsonl during this project).

Used by:
  - step5_rag_answer.py / run_validation_batch.py / ui_rag_app.py, to
    automatically upgrade an --article/--edcode-section filter into a
    complete doc_id-based filter before querying Chroma, instead of
    silently returning only the ~1/N anchor-tagged chunks of a document.
  - compute_retrieval_metrics.py, to build the "gold" document labels used
    for Recall@k/MRR.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


ARTICLE_FILENAME_RE = re.compile(r"ARTICLE\s+([IVXLCDM]+)\b", re.IGNORECASE)


def build_resolution_maps(chunks_path: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Returns:
      chunk_to_doc: chunk_id -> doc_id
      article_to_docs: 'IX' -> {doc_id, ...}   (from filename, ca_constitution only)
      edcode_section_to_docs: '48900' -> {doc_id, ...}  (from anchors.edcode_sections)
    """
    chunk_to_doc: Dict[str, str] = {}
    article_to_docs: Dict[str, Set[str]] = defaultdict(set)
    edcode_section_to_docs: Dict[str, Set[str]] = defaultdict(set)

    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunk_id = d.get("chunk_id")
            doc_id = d.get("doc_id")
            if not chunk_id or not doc_id:
                continue
            chunk_to_doc[chunk_id] = doc_id

            doc_type = d.get("doc_type")
            rel_path = d.get("rel_path", "") or ""
            anchors = d.get("anchors") or {}

            if doc_type == "ca_constitution":
                m = ARTICLE_FILENAME_RE.search(rel_path)
                if m:
                    article_to_docs[m.group(1).upper()].add(doc_id)

            if doc_type == "ca_education_code":
                for sec in anchors.get("edcode_sections") or []:
                    edcode_section_to_docs[str(sec)].add(doc_id)

    return chunk_to_doc, article_to_docs, edcode_section_to_docs


def resolve_where_params(
    doc_type: Optional[str],
    article: Optional[str],
    edcode_section: Optional[str],
    article_to_docs: Dict[str, Set[str]],
    edcode_section_to_docs: Dict[str, Set[str]],
) -> Tuple[Optional[str], Optional[str], Optional[List[str]]]:
    """
    Convenience wrapper for callers building a Chroma `where` filter:
    if article/edcode_section resolves to doc_id(s), returns
    (None, None, doc_ids) so the caller uses doc_id filtering (complete
    recall) instead of the raw anchor field (incomplete recall). If nothing
    resolves, returns (article, edcode_section, None) unchanged so the
    caller falls back to the original anchor-based filter.
    """
    resolved = resolve_doc_ids(doc_type, article, edcode_section, article_to_docs, edcode_section_to_docs)
    if resolved:
        return None, None, resolved
    return article, edcode_section, None


def resolve_doc_ids(
    doc_type: Optional[str],
    article: Optional[str],
    edcode_section: Optional[str],
    article_to_docs: Dict[str, Set[str]],
    edcode_section_to_docs: Dict[str, Set[str]],
) -> Optional[List[str]]:
    """
    Resolve an article/edcode_section label to the complete set of doc_ids
    it should match, using filename/anchor-scan-derived maps rather than
    the incomplete per-chunk anchor field. Returns None if nothing could
    be resolved (caller should fall back to the plain anchor-based filter
    in that case, e.g. for edcode_sections not seen anywhere in the corpus).
    """
    article_clean = (article or "").strip().upper()
    section_clean = (edcode_section or "").strip()

    if doc_type == "ca_constitution" and article_clean:
        docs = article_to_docs.get(article_clean)
        return sorted(docs) if docs else None

    if doc_type == "ca_education_code" and section_clean:
        docs = edcode_section_to_docs.get(section_clean)
        return sorted(docs) if docs else None

    return None
