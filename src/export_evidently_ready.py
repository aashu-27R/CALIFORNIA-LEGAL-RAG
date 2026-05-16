from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
OUT_DIR = EVAL_DIR / "evidently_ready"


def load_question_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def nonempty_docx_paragraphs(path: Path) -> List[str]:
    doc = Document(path)
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def normalize_question_text(text: str) -> str:
    text = text.strip()
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"^\d+\.\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_docx_by_question_order(
    path: Path,
    question_rows: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    paragraphs = nonempty_docx_paragraphs(path)
    out: List[Dict[str, str]] = []
    para_idx = 0
    for row in question_rows:
        while para_idx < len(paragraphs):
            line = paragraphs[para_idx]
            normalized = normalize_question_text(line)
            if (
                normalized.endswith("?")
                and not normalized.startswith("Q")
                and not normalized.startswith("SECTION")
                and not normalized.startswith("California Legal RAG")
                and not normalized.startswith("Capstone Project")
            ):
                break
            para_idx += 1
        if para_idx >= len(paragraphs):
            raise ValueError(f"Could not find next question in {path.name} for {row['question_id']}")

        question = paragraphs[para_idx]
        para_idx += 1
        answer_parts: List[str] = []
        reference = ""
        while para_idx < len(paragraphs):
            line = paragraphs[para_idx]
            if line.startswith("Reference:"):
                reference = line[len("Reference:"):].strip()
                para_idx += 1
                break
            answer_parts.append(line)
            para_idx += 1

        out.append(
            {
                "question_id": row["question_id"],
                "category": row.get("category", ""),
                "question": row["question"],
                "raw_question": question,
                "response": "\n\n".join(answer_parts).strip(),
                "reference": reference,
            }
        )
    return out


def parse_chatgpt_set1_from_json(path: Path) -> List[Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "question_id": row["question_id"],
            "category": row.get("category", ""),
            "question": row["question"],
            "response": row["chatgpt_answer"],
            "reference": row.get("reference", ""),
        }
        for row in data
    ]


def parse_chatgpt_set2_docx(path: Path, question_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    paragraphs = nonempty_docx_paragraphs(path)
    out: List[Dict[str, str]] = []
    q_idx = 0
    i = 0
    while i < len(paragraphs) and q_idx < len(question_rows):
        line = paragraphs[i]
        if not re.match(r"^\d+\.\s*", line):
            i += 1
            continue
        parts = line.split("\n", 1)
        answer = parts[1].strip() if len(parts) > 1 else ""
        reference = ""
        if i + 1 < len(paragraphs) and paragraphs[i + 1].startswith("Reference:"):
            reference = paragraphs[i + 1][len("Reference:"):].strip()
            i += 2
        else:
            i += 1
        out.append(
            {
                "question_id": question_rows[q_idx]["question_id"],
                "category": question_rows[q_idx].get("category", ""),
                "question": question_rows[q_idx]["question"],
                "raw_question": normalize_question_text(parts[0]),
                "response": answer,
                "reference": reference,
            }
        )
        q_idx += 1
    if q_idx != len(question_rows):
        raise ValueError(f"Parsed only {q_idx} questions from {path.name}, expected {len(question_rows)}")
    return out


def load_results_mode(path: Path, mode: str) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("mode") == mode]
    out: List[Dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "question_id": row["question_id"],
                "category": row.get("category", ""),
                "question": row["question"],
                "response": row["answer"],
                "reference": row.get("sources_json", ""),
                "retrieved_chunks": row.get("retrieved_chunks", ""),
                "context_chars": row.get("context_chars", ""),
                "error": row.get("error", ""),
            }
        )
    return out


def write_csv_json(
    stem: str,
    rows: List[Dict[str, str]],
    set_name: str,
    system_name: str,
    source_file: str,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enriched = []
    for row in rows:
        enriched.append(
            {
                "set_name": set_name,
                "system_name": system_name,
                "question_id": row.get("question_id", ""),
                "category": row.get("category", ""),
                "question": row.get("question", ""),
                "response": row.get("response", ""),
                "reference": row.get("reference", ""),
                "retrieved_chunks": row.get("retrieved_chunks", ""),
                "context_chars": row.get("context_chars", ""),
                "error": row.get("error", ""),
                "source_file": source_file,
            }
        )

    csv_path = OUT_DIR / f"{stem}.csv"
    json_path = OUT_DIR / f"{stem}.json"
    fieldnames = list(enriched[0].keys()) if enriched else [
        "set_name",
        "system_name",
        "question_id",
        "category",
        "question",
        "response",
        "reference",
        "retrieved_chunks",
        "context_chars",
        "error",
        "source_file",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)
    json_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    set1_questions = load_question_rows(EVAL_DIR / "validation_questions.csv")
    set2_questions = load_question_rows(EVAL_DIR / "validation_questions_set2.csv")

    write_csv_json(
        stem="chatgpt_responses_set1",
        rows=parse_chatgpt_set1_from_json(EVAL_DIR / "chatgpt_answers_from_pdf.json"),
        set_name="set1",
        system_name="chatgpt",
        source_file="data/eval/chatgpt_answers_from_pdf.json",
    )
    write_csv_json(
        stem="chatgpt_responses_set2",
        rows=parse_chatgpt_set2_docx(EVAL_DIR / "external_baselines" / "chatgpt_questions_set2.docx", set2_questions),
        set_name="set2",
        system_name="chatgpt",
        source_file="data/eval/external_baselines/chatgpt_questions_set2.docx",
    )
    write_csv_json(
        stem="claude_responses_set1",
        rows=parse_docx_by_question_order(EVAL_DIR / "external_baselines" / "claude_responses_set1.docx", set1_questions),
        set_name="set1",
        system_name="claude",
        source_file="data/eval/external_baselines/claude_responses_set1.docx",
    )
    write_csv_json(
        stem="claude_responses_set2",
        rows=parse_docx_by_question_order(EVAL_DIR / "external_baselines" / "Claude_responses_set2.docx", set2_questions),
        set_name="set2",
        system_name="claude",
        source_file="data/eval/external_baselines/Claude_responses_set2.docx",
    )

    # Use the later set1 crossfix run so both set1 and set2 align on top_k=6 / kg_expand_k=8.
    write_csv_json(
        stem="our_model_vector_only_responses_set1",
        rows=load_results_mode(EVAL_DIR / "results" / "validation_crossfix_20260413_135527_results.csv", "vector"),
        set_name="set1",
        system_name="our_model_vector_only",
        source_file="data/eval/results/validation_crossfix_20260413_135527_results.csv",
    )
    write_csv_json(
        stem="our_model_vector_plus_kg_responses_set1",
        rows=load_results_mode(EVAL_DIR / "results" / "validation_crossfix_20260413_135527_results.csv", "kg"),
        set_name="set1",
        system_name="our_model_vector_plus_kg",
        source_file="data/eval/results/validation_crossfix_20260413_135527_results.csv",
    )
    write_csv_json(
        stem="our_model_vector_only_responses_set2",
        rows=load_results_mode(EVAL_DIR / "results" / "validation_set2_20260511_121544_results.csv", "vector"),
        set_name="set2",
        system_name="our_model_vector_only",
        source_file="data/eval/results/validation_set2_20260511_121544_results.csv",
    )
    write_csv_json(
        stem="our_model_vector_plus_kg_responses_set2",
        rows=load_results_mode(EVAL_DIR / "results" / "validation_set2_20260511_121544_results.csv", "kg"),
        set_name="set2",
        system_name="our_model_vector_plus_kg",
        source_file="data/eval/results/validation_set2_20260511_121544_results.csv",
    )

    manifest = {
        "folder": "data/eval/evidently_ready",
        "notes": [
            "Set 1 uses the later validation_crossfix run so retrieval settings align with set 2 (top_k=6, kg_expand_k=8).",
            "Each CSV has one row per question for a single system and set.",
            "JSON files contain the same rows as their CSV counterpart.",
        ],
        "files": sorted(p.name for p in OUT_DIR.iterdir() if p.is_file()),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
