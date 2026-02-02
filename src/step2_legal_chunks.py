"""
STEP 2 — Chunk legal docs for RAG (citation-friendly).

Input : data/corpus/docs.jsonl (from Step 1)
Output: data/chunks/chunks.jsonl
        data/chunks/chunk_stats.json
"""

from __future__ import annotations
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter


IN_PATH  = Path("data/corpus/docs.jsonl")
OUT_DIR  = Path("data/chunks")
OUT_JSONL = OUT_DIR / "chunks.jsonl"
OUT_STATS = OUT_DIR / "chunk_stats.json"

# Chunk sizing (tune later)
TARGET_CHARS = 1800      # ~300-450 tokens depending on text
OVERLAP_CHARS = 250

# Regex patterns to detect legal anchors
RE_CA_CONST_ARTICLE = re.compile(r"\bARTICLE\s+([IVXLCDM]+)\b", re.IGNORECASE)
RE_SECTION_NUM = re.compile(r"\bSEC(?:TION)?\.?\s*(\d+[A-Za-z0-9\.\-]*)\b", re.IGNORECASE)
RE_EDCODE_SECTION = re.compile(r"(?:\bEd(?:ucation)?\.?\s*Code\b)?\s*§+\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
RE_EDCODE_SECTION_GENERIC = re.compile(r"\b(?:Section|Sec\.?)\s+(\d{4,6}(?:\.\d+)*)\b", re.IGNORECASE)

# Sometimes PDFs have "SECTION 1." etc without "Sec"
RE_BARE_SECTION = re.compile(r"\bSECTION\s+(\d+[A-Za-z0-9\.\-]*)\b", re.IGNORECASE)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_anchors(doc_type: str, text: str) -> Dict[str, Any]:
    """
    Best-effort extraction of:
      - constitution article roman numeral
      - section numbers
      - ed code section markers
    """
    anchors: Dict[str, Any] = {}

    if doc_type == "ca_constitution":
        m = RE_CA_CONST_ARTICLE.search(text)
        if m:
            anchors["article"] = m.group(1).upper()

        # constitution PDFs often contain SEC. 1 / SECTION 1
        secs = []
        for rx in (RE_SECTION_NUM, RE_BARE_SECTION):
            secs.extend([x for x in rx.findall(text)])
        if secs:
            anchors["sections"] = list(dict.fromkeys(secs))[:20]  # unique, cap

    elif doc_type == "ca_education_code":
        # Many ed-code docs contain § 48900 etc, but some use "Section 48900"
        ordered = []
        for m in RE_EDCODE_SECTION.finditer(text):
            ordered.append(m.group(1))
        for m in RE_EDCODE_SECTION_GENERIC.finditer(text):
            ordered.append(m.group(1))

        if ordered:
            anchors["edcode_sections"] = list(dict.fromkeys(ordered))[:30]

        # fallback: SEC / SECTION patterns (less specific)
        secs2 = []
        for rx in (RE_SECTION_NUM, RE_BARE_SECTION):
            secs2.extend([x for x in rx.findall(text)])
        if secs2:
            anchors["sections"] = list(dict.fromkeys(secs2))[:20]

    else:
        # generic
        secs = []
        for rx in (RE_SECTION_NUM, RE_BARE_SECTION):
            secs.extend([x for x in rx.findall(text)])
        if secs:
            anchors["sections"] = list(dict.fromkeys(secs))[:20]

    return anchors


def chunk_text(text: str, target: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    Chunk text by paragraph boundaries with a safe fallback for very long paragraphs.
    Returns list of (start_char, end_char, chunk_str)
    """
    text = normalize_ws(text)
    if len(text) <= target:
        return [(0, len(text), text)]

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return [(0, len(text), text)]

    chunks: List[Tuple[int, int, str]] = []
    cursor = 0
    i = 0

    def add_chunk(chunk_str: str, start_cursor: int) -> int:
        pos = text.find(chunk_str, start_cursor)
        if pos == -1:
            pos = start_cursor
        end_pos = min(len(text), pos + len(chunk_str))
        chunks.append((pos, end_pos, chunk_str))
        return end_pos

    while i < len(paras):
        # If a single paragraph is too long, split it safely and move on.
        if len(paras[i]) > target:
            para = paras[i]
            start = 0
            while start < len(para):
                part = para[start:start + target]
                end_pos = add_chunk(part, cursor)
                cursor = max(end_pos - overlap, 0)
                start = max(start + target - overlap, start + 1)
            i += 1
            continue

        chunk_parts: List[str] = []
        chunk_len = 0
        start_cursor = cursor

        while i < len(paras) and chunk_len + len(paras[i]) + 2 <= target:
            chunk_parts.append(paras[i])
            chunk_len += len(paras[i]) + 2
            i += 1

        chunk_str = "\n\n".join(chunk_parts).strip()
        if not chunk_str:
            # Safety fallback: hard slice to avoid infinite loops
            chunk_str = text[cursor:cursor + target]
            if not chunk_str:
                break
            end_cursor = min(len(text), cursor + len(chunk_str))
            chunks.append((cursor, end_cursor, chunk_str))
            cursor = max(end_cursor - overlap, cursor + 1)
            continue

        end_pos = add_chunk(chunk_str, start_cursor)
        cursor = max(end_pos - overlap, 0)

        if end_pos >= len(text):
            break

    deduped = []
    seen = set()
    for s, e, c in chunks:
        h = sha(c[:400])
        if h in seen:
            continue
        seen.add(h)
        deduped.append((s, e, c))

    return deduped


def make_chunk_record(
    doc: Dict[str, Any],
    idx: int,
    start: int,
    end: int,
    chunk: str,
    anchors: Dict[str, Any],
) -> Dict[str, Any]:
    rel = doc.get("rel_path", "")
    doc_id = doc["doc_id"]
    chunk_id = sha(f"{doc_id}::chunk::{idx}::{start}-{end}")[:24]

    # citation label (best-effort)
    if doc.get("doc_type") == "ca_constitution":
        label = f"CA Constitution"
        if "article" in anchors:
            label += f", Art. {anchors['article']}"
    elif doc.get("doc_type") == "ca_education_code":
        label = "CA Education Code"
    else:
        label = doc.get("doc_type", "legal_doc")

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_type": doc.get("doc_type"),
        "rel_path": rel,
        "source_label": label,
        "article": anchors.get("article"),
        "anchors": anchors,          # per-chunk anchors
        "chunk_index": idx,
        "start_char": start,
        "end_char": end,
        "text": chunk,
        "text_len": len(chunk),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    doc_type_counts = Counter()
    chunk_counts = Counter()
    total_docs = 0
    total_chunks = 0

    with IN_PATH.open("r", encoding="utf-8") as f_in, OUT_JSONL.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            total_docs += 1
            doc = json.loads(line)
            doc_type = doc.get("doc_type", "unknown")
            doc_type_counts[doc_type] += 1

            text = doc.get("text", "")
            spans = chunk_text(text, TARGET_CHARS, OVERLAP_CHARS)
            chunk_counts[doc_type] += len(spans)
            total_chunks += len(spans)

            for idx, (s, e, chunk) in enumerate(spans):
                chunk_anchors = extract_anchors(doc_type, chunk)
                rec = make_chunk_record(doc, idx, s, e, chunk, chunk_anchors)
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "input_docs": total_docs,
        "output_chunks": total_chunks,
        "docs_by_type": dict(doc_type_counts),
        "chunks_by_type": dict(chunk_counts),
        "target_chars": TARGET_CHARS,
        "overlap_chars": OVERLAP_CHARS,
        "out_jsonl": str(OUT_JSONL),
    }

    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("✅ Step 2 complete")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
