"""
STEP 1 — Build corpus from a root folder (California Education Code + CA Constitution).

Input:
  - ROOT_FOLDER: path to your main folder (e.g., .../california_dataset)
    which contains subfolders:
      - california_constitution/
      - education_code/

Output:
  - data/corpus/docs.jsonl         (one JSON per document)
  - data/corpus/read_errors.jsonl  (files that failed)
"""

from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Iterable

# ---- PDF support ----
# pip install pypdf
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

# ---- HTML support (optional) ----
# pip install beautifulsoup4 lxml
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except Exception:
    HAS_BS4 = False


SUPPORTED_EXTS = {".txt", ".md", ".html", ".htm", ".pdf"}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def clean_text(text: str) -> str:
    # normalize common non-breaking spaces
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_leginfo_boilerplate(text: str) -> str:
    """
    Remove common Leginfo page chrome and URL noise from scraped PDFs/HTML.
    """
    # Remove leginfo URLs
    text = re.sub(r"https?://leginfo\.legislature\.ca\.gov\S*", " ", text, flags=re.IGNORECASE)
    # Remove "Codes Display Text Page X of Y"
    text = re.sub(r"Codes\s+Display\s+Text\s+Page\s+\d+\s+of\s+\d+", " ", text, flags=re.IGNORECASE)
    # Remove "Code:Select CodeSection: ... Search"
    text = re.sub(r"Code:Select\s+CodeSection:.*?Search", " ", text, flags=re.IGNORECASE | re.DOTALL)
    # Remove navigation chrome lines
    text = re.sub(r"Up\^Add To My Favorites", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"My Subscriptions|My Favorites|Other Resources", " ", text, flags=re.IGNORECASE)
    # Remove timestamps like "11/27/25, 6:03 PM"
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(AM|PM)", " ", text, flags=re.IGNORECASE)
    return text


def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def guess_doc_type_by_parent_folder(file_path: Path) -> str:
    parents = [p.name.lower() for p in file_path.parents]
    if "california_constitution" in parents:
        return "ca_constitution"
    if "education_code" in parents:
        return "ca_education_code"
    return "unknown_legal"


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")

    # Fallback: rough tag stripping if bs4 not installed
    if not HAS_BS4:
        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
        return re.sub(r"<[^>]+>", " ", html)

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def read_pdf(path: Path) -> str:
    if not HAS_PYPDF:
        raise RuntimeError("PDF found but pypdf is not installed. Run: pip install pypdf")

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)


def make_record(path: Path, root: Path, raw_text: str) -> Dict[str, Any]:
    text = clean_text(raw_text)
    text = clean_leginfo_boilerplate(text)
    rel = str(path.relative_to(root))
    doc_type = guess_doc_type_by_parent_folder(path)

    # stable-ish doc id: relative path + text hash prefix
    content_hash = sha256_text(text)
    doc_id = sha256_text(rel + "::" + content_hash[:16])

    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "rel_path": rel,
        "abs_path": str(path),
        "ext": path.suffix.lower(),
        "text": text,
        "text_len": len(text),
        "content_sha256": content_hash,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
    }


def main():
    # ✅ EDIT THIS to your folder from the screenshot
    ROOT_FOLDER = Path("/Users/aashrithasankineni/Downloads/california_dataset").expanduser()

    out_dir = Path("data/corpus")
    out_dir.mkdir(parents=True, exist_ok=True)

    docs_out = out_dir / "docs.jsonl"
    err_out = out_dir / "read_errors.jsonl"

    scanned = 0
    saved = 0

    with docs_out.open("w", encoding="utf-8") as f_docs, err_out.open("w", encoding="utf-8") as f_err:
        for fp in iter_files(ROOT_FOLDER):
            scanned += 1
            try:
                ext = fp.suffix.lower()
                if ext in {".txt", ".md"}:
                    raw = read_txt(fp)
                elif ext in {".html", ".htm"}:
                    raw = read_html(fp)
                elif ext == ".pdf":
                    raw = read_pdf(fp)
                else:
                    continue

                rec = make_record(fp, ROOT_FOLDER, raw)

                # basic quality filter
                if rec["text_len"] < 200:
                    continue

                f_docs.write(json.dumps(rec, ensure_ascii=False) + "\n")
                saved += 1

            except Exception as e:
                f_err.write(json.dumps({"file": str(fp), "error": str(e)}, ensure_ascii=False) + "\n")

    print(f"Scanned files: {scanned}")
    print(f"Saved docs:   {saved}")
    print(f"Wrote: {docs_out}")
    print(f"Errors: {err_out}")


if __name__ == "__main__":
    main()
