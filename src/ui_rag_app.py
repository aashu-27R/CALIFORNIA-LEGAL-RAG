"""Streamlit UI for California Legal RAG (vector-only and vector+KG modes)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb
import streamlit as st
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


@st.cache_resource
def get_embed_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


@st.cache_resource
def get_collection(chroma_dir: str, collection_name: str):
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(name=collection_name)


def run_query(
    query: str,
    top_k: int,
    chroma_dir: str,
    collection_name: str,
    embed_model_name: str,
    openai_model: str,
    doc_type: str | None,
    edcode_section: str | None,
    article: str | None,
    use_kg: bool,
    kg_expand_k: int,
) -> Tuple[str, List[Dict[str, Any]], str]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    if not Path(chroma_dir).exists():
        raise FileNotFoundError(f"Missing Chroma dir: {chroma_dir}")

    collection = get_collection(chroma_dir, collection_name)
    embed_model = get_embed_model(embed_model_name)
    query_emb = embed_model.encode([query], normalize_embeddings=True)

    where = build_where(doc_type, edcode_section, article)
    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if use_kg and ids:
        extra_ids = kg_expand_chunk_ids(ids, kg_expand_k)
        if extra_ids:
            extra = collection.get(ids=extra_ids, include=["documents", "metadatas"])
            extra_docs = extra.get("documents", [])
            extra_metas = extra.get("metadatas", [])

            seen = set()
            merged_docs: List[str] = []
            merged_metas: List[Dict[str, Any]] = []
            for d, m in list(zip(docs, metas)) + list(zip(extra_docs, extra_metas)):
                key = (
                    m.get("doc_id"),
                    m.get("chunk_index"),
                    m.get("start_char"),
                    m.get("end_char"),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged_docs.append(d)
                merged_metas.append(m)
            docs, metas = merged_docs, merged_metas

    context = format_context(docs, metas)

    system_prompt = (
        "You are a legal RAG assistant for California Education Code (Title 1) "
        "and the California Constitution. Answer strictly using the provided context. "
        "Cite sources as [#] using the numbered context blocks. If the answer is not "
        "in the context, say you don't have enough information."
    )

    client = OpenAI()
    response = client.responses.create(
        model=openai_model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
        ],
    )
    answer = (getattr(response, "output_text", None) or str(response)).strip()

    cited = set(int(x) for x in re.findall(r"\[(\d+)\]", answer))
    cited_sources: List[Dict[str, Any]] = []
    for i, meta in enumerate(metas, 1):
        if cited and i not in cited:
            continue
        cited_sources.append(
            {
                "idx": i,
                "label": meta.get("source_label", "source"),
                "rel_path": meta.get("rel_path", ""),
                "anchors": meta.get("anchors_json", ""),
            }
        )

    return answer, cited_sources, context


def main() -> None:
    st.set_page_config(page_title="California Legal RAG", layout="wide")
    st.title("California Legal RAG UI")

    with st.sidebar:
        st.subheader("Settings")
        mode = st.radio("Retrieval Mode", ["Vector only", "Vector + KG"], index=1)
        top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
        kg_expand_k = st.slider("KG Expand K", min_value=1, max_value=30, value=12)

        doc_type = st.selectbox(
            "Doc Type",
            options=["(all)", "ca_constitution", "ca_education_code"],
            index=0,
        )
        article = st.text_input("Article (optional, e.g., I, IX)", "")
        edcode_section = st.text_input("EdCode Section (optional)", "")

        chroma_dir = st.text_input("Chroma Dir", "data/chroma")
        collection_name = st.text_input("Collection", DEFAULT_COLLECTION)
        embed_model_name = st.text_input("Embed Model", DEFAULT_EMBED_MODEL)
        openai_model = st.text_input("OpenAI Model", DEFAULT_OPENAI_MODEL)

    query = st.text_area(
        "Question",
        value="What does Article IX say about education?",
        height=90,
    )

    if st.button("Ask", type="primary"):
        try:
            answer, sources, context = run_query(
                query=query.strip(),
                top_k=top_k,
                chroma_dir=chroma_dir.strip(),
                collection_name=collection_name.strip(),
                embed_model_name=embed_model_name.strip(),
                openai_model=openai_model.strip(),
                doc_type=None if doc_type == "(all)" else doc_type,
                edcode_section=edcode_section.strip() or None,
                article=article.strip().upper() or None,
                use_kg=(mode == "Vector + KG"),
                kg_expand_k=kg_expand_k,
            )

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Sources")
            if sources:
                st.dataframe(sources, use_container_width=True)
            else:
                st.info("No citations detected in answer.")

            with st.expander("Retrieved Context"):
                st.text(context)

        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
