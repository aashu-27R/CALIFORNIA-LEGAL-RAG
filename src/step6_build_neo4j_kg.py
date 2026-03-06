"""
STEP 6 — Build a Knowledge Graph in Neo4j from legal chunks.

This script:
1) Reads chunked docs from data/chunks/chunks.jsonl
2) Uses OpenAI to extract entities + relations per chunk
3) Upserts nodes/edges into Neo4j

Requirements:
- OPENAI_API_KEY set
- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD set
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI
from neo4j import GraphDatabase


DEFAULT_OPENAI_MODEL = "gpt-5-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Neo4j knowledge graph from legal chunks")
    parser.add_argument("--in-path", default="data/chunks/chunks.jsonl")
    parser.add_argument("--topics-path", default=None, help="Optional chunk topic mapping JSONL from step7")
    parser.add_argument("--max-chunks", type=int, default=200, help="Limit number of chunks for KG extraction")
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--workers", type=int, default=1, help="Parallel OpenAI extraction workers (Neo4j writes stay serialized)")
    parser.add_argument("--retries", type=int, default=3, help="Retries per chunk for API/transient failures")
    parser.add_argument("--retry-backoff", type=float, default=2.0, help="Base seconds for exponential backoff")
    parser.add_argument("--checkpoint-file", default=None, help="Optional JSON checkpoint path for resume support")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint-file if present")
    parser.add_argument("--start-index", type=int, default=0, help="0-based start index in chunks.jsonl (ignored if --resume checkpoint is newer)")
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set.")
    return value


def ensure_constraints(driver) -> None:
    cypher = [
        "CREATE CONSTRAINT kg_entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE",
        "CREATE CONSTRAINT kg_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT kg_doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
        "CREATE CONSTRAINT kg_article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.article IS UNIQUE",
        "CREATE CONSTRAINT kg_section_key IF NOT EXISTS FOR (s:Section) REQUIRE (s.number, s.doc_type) IS UNIQUE",
        "CREATE CONSTRAINT kg_topic_path IF NOT EXISTS FOR (t:Topic) REQUIRE t.path IS UNIQUE",
    ]
    with driver.session() as session:
        for q in cypher:
            session.run(q)


def load_topics_map(path_str: str | None) -> Dict[str, Dict[str, Any]]:
    if not path_str:
        return {}
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Missing topics file: {p}")
    out: Dict[str, Dict[str, Any]] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cid = row.get("chunk_id")
            if cid:
                out[cid] = row
    return out


def build_prompt(chunk: Dict[str, Any]) -> str:
    return (
        "Extract a lightweight knowledge graph from this legal text.\n"
        "Return STRICT JSON with keys: entities, relations.\n"
        "entities: list of {name, type} where type is one of: PERSON, ORG, LAW, POLICY, ROLE, GOV_BODY, CONCEPT.\n"
        "relations: list of {subject, predicate, object} using subject/object names from entities.\n"
        "Keep it small: max 8 entities and max 8 relations.\n\n"
        f"TEXT:\n{chunk.get('text','')}\n"
    )


def extract_kg(client: OpenAI, model: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_prompt(chunk)
    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )
    raw = getattr(resp, "output_text", "") or ""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        # Fallback: attempt to locate JSON substring
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        return {"entities": [], "relations": []}


def extract_kg_with_retry(
    model: str,
    chunk: Dict[str, Any],
    retries: int,
    retry_backoff: float,
) -> Dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            # Per-call client avoids thread-sharing concerns.
            client = OpenAI()
            return extract_kg(client, model, chunk)
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            sleep_s = retry_backoff * (2 ** attempt)
            time.sleep(sleep_s)
    raise RuntimeError(f"KG extraction failed for chunk {chunk.get('chunk_id')}: {last_err}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_checkpoint(path_str: str | None) -> Dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_checkpoint(path_str: str | None, payload: Dict[str, Any]) -> None:
    if not path_str:
        return
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def upsert_chunk_graph(
    driver,
    chunk: Dict[str, Any],
    kg: Dict[str, Any],
    topic_row: Dict[str, Any] | None = None,
) -> None:
    entities = kg.get("entities") or []
    relations = kg.get("relations") or []

    doc = {
        "doc_id": chunk.get("doc_id"),
        "doc_type": chunk.get("doc_type"),
        "rel_path": chunk.get("rel_path"),
        "source_label": chunk.get("source_label"),
    }
    ch = {
        "chunk_id": chunk.get("chunk_id"),
        "start_char": chunk.get("start_char"),
        "end_char": chunk.get("end_char"),
        "text_len": chunk.get("text_len"),
    }
    article = chunk.get("article")
    anchors = chunk.get("anchors") or {}
    ed_sections = anchors.get("edcode_sections") or []
    sections = anchors.get("sections") or []

    with driver.session() as session:
        session.run(
            """
            MERGE (d:Document {doc_id: $doc.doc_id})
            SET d.doc_type = $doc.doc_type,
                d.rel_path = $doc.rel_path,
                d.source_label = $doc.source_label
            MERGE (c:Chunk {chunk_id: $chunk.chunk_id})
            SET c.start_char = $chunk.start_char,
                c.end_char = $chunk.end_char,
                c.text_len = $chunk.text_len
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            doc=doc,
            chunk=ch,
        )

        if article:
            session.run(
                """
                MERGE (a:Article {article: $article})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                MERGE (c)-[:IN_ARTICLE]->(a)
                """,
                article=article,
                chunk_id=ch["chunk_id"],
            )

        if topic_row:
            topic_path = topic_row.get("topic_path")
            t1 = topic_row.get("topic_level1")
            t2 = topic_row.get("topic_level2")
            if topic_path:
                session.run(
                    """
                    MERGE (t:Topic {path: $topic_path})
                    SET t.level1 = $t1,
                        t.level2 = $t2
                    MERGE (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:IN_TOPIC]->(t)
                    """,
                    topic_path=topic_path,
                    t1=t1,
                    t2=t2,
                    chunk_id=ch["chunk_id"],
                )
                if t1 and t2:
                    session.run(
                        """
                        MERGE (p:Topic {path: $parent_path})
                        SET p.level1 = $t1,
                            p.level2 = null
                        MERGE (t:Topic {path: $topic_path})
                        MERGE (t)-[:SUBTOPIC_OF]->(p)
                        """,
                        parent_path=t1,
                        t1=t1,
                        topic_path=topic_path,
                    )

        for sec in ed_sections:
            session.run(
                """
                MERGE (s:Section {number: $number, doc_type: $doc_type})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                MERGE (c)-[:MENTIONS_SECTION]->(s)
                """,
                number=str(sec),
                doc_type=doc.get("doc_type"),
                chunk_id=ch["chunk_id"],
            )

        for sec in sections:
            session.run(
                """
                MERGE (s:Section {number: $number, doc_type: $doc_type})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                MERGE (c)-[:MENTIONS_SECTION]->(s)
                """,
                number=str(sec),
                doc_type=doc.get("doc_type"),
                chunk_id=ch["chunk_id"],
            )

        # Entities
        for ent in entities:
            name = (ent.get("name") or "").strip()
            etype = (ent.get("type") or "CONCEPT").strip().upper()
            if not name:
                continue
            session.run(
                """
                MERGE (e:Entity {name: $name, type: $type})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                name=name,
                type=etype,
                chunk_id=ch["chunk_id"],
            )

        # Relations
        for rel in relations:
            subj = (rel.get("subject") or "").strip()
            obj = (rel.get("object") or "").strip()
            pred = (rel.get("predicate") or "").strip().upper()
            if not subj or not obj or not pred:
                continue
            session.run(
                """
                MERGE (s:Entity {name: $subj, type: 'CONCEPT'})
                MERGE (o:Entity {name: $obj, type: 'CONCEPT'})
                MERGE (s)-[r:RELATION {predicate: $pred}]->(o)
                ON CREATE SET r.count = 1
                ON MATCH SET r.count = r.count + 1
                """,
                subj=subj,
                obj=obj,
                pred=pred,
            )


def main() -> None:
    args = parse_args()

    require_env("OPENAI_API_KEY")
    uri = require_env("NEO4J_URI")
    user = require_env("NEO4J_USER")
    password = require_env("NEO4J_PASSWORD")

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    topics_map = load_topics_map(args.topics_path)
    if topics_map:
        print(f"Loaded topic mappings: {len(topics_map)}")

    ckpt = load_checkpoint(args.checkpoint_file)
    checkpoint_start = int(ckpt.get("next_index", 0)) if (args.resume and ckpt) else 0
    start_index = max(int(args.start_index), checkpoint_start)
    if args.checkpoint_file:
        print(f"Checkpoint file: {args.checkpoint_file}")
    if args.resume:
        print(f"Resume enabled. Starting at chunk index: {start_index}")
    elif start_index > 0:
        print(f"Starting at chunk index: {start_index}")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    ensure_constraints(driver)

    # Load only the requested window of chunks so we can parallelize extraction.
    selected_chunks: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start_index:
                continue
            if len(selected_chunks) >= args.max_chunks:
                break
            selected_chunks.append(json.loads(line))

    processed = 0
    failed = 0
    workers = max(1, int(args.workers))

    def checkpoint_update(done_count: int, fail_count: int, mode: str) -> None:
        write_checkpoint(
            args.checkpoint_file,
            {
                "mode": mode,
                "started_at": ckpt.get("started_at") if ckpt else _utc_now_iso(),
                "updated_at": _utc_now_iso(),
                "start_index": start_index,
                "next_index": start_index + done_count + fail_count,
                "processed_success": done_count,
                "processed_failed": fail_count,
                "max_chunks_requested": int(args.max_chunks),
                "workers": workers,
                "openai_model": args.openai_model,
            },
        )

    checkpoint_update(0, 0, "running")

    if workers == 1:
        for chunk in selected_chunks:
            try:
                kg = extract_kg_with_retry(
                    args.openai_model,
                    chunk,
                    retries=args.retries,
                    retry_backoff=args.retry_backoff,
                )
                upsert_chunk_graph(driver, chunk, kg, topics_map.get(chunk.get("chunk_id")))
                processed += 1
            except Exception as e:
                failed += 1
                print(f"Failed chunk {chunk.get('chunk_id')}: {e}")
            checkpoint_update(processed, failed, "running")
            if (processed + failed) % 20 == 0:
                print(f"Processed {processed} chunks... Failed {failed}...")
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(
                    extract_kg_with_retry,
                    args.openai_model,
                    chunk,
                    args.retries,
                    args.retry_backoff,
                ): chunk
                for chunk in selected_chunks
            }
            for fut in as_completed(future_map):
                chunk = future_map[fut]
                try:
                    kg = fut.result()
                    upsert_chunk_graph(driver, chunk, kg, topics_map.get(chunk.get("chunk_id")))
                    processed += 1
                except Exception as e:
                    failed += 1
                    print(f"Failed chunk {chunk.get('chunk_id')}: {e}")
                checkpoint_update(processed, failed, "running")
                if (processed + failed) % 20 == 0:
                    print(f"Processed {processed} chunks... Failed {failed}...")

    driver.close()
    checkpoint_update(processed, failed, "completed")
    print("✅ Step 6 complete")
    print(f"Chunks processed: {processed}")
    if failed:
        print(f"Chunks failed: {failed}")


if __name__ == "__main__":
    main()
