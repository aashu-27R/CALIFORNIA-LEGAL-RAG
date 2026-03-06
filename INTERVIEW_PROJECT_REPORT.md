# California Legal RAG + Knowledge Graph Project (Interview Summary)

## 1) Project Goal
Build a legal question-answering system for California law that is more grounded and less hallucinatory than plain LLM chat by combining:
- vector retrieval (semantic search),
- metadata filtering (article/section/doc type),
- knowledge-graph expansion (Neo4j),
- citation-based answer generation.

Corpus scope used so far:
- California Constitution
- California Education Code Title 1

## 2) What We Built So Far (Current State)
Pipeline status:
1. Data ingestion from local legal files -> complete
2. Corpus normalization + structured JSONL output -> complete
3. Legal chunking + anchor metadata -> complete
4. Chroma vector index build -> complete
5. Query retrieval CLI -> complete
6. OpenAI answer generation with citations -> complete
7. Neo4j KG construction pipeline -> implemented and running in batches
8. Topic hierarchy layer (HNMF-style) -> complete and integrated
9. KG-assisted answer mode (`--use-kg`) -> implemented
10. Parallel KG extraction + checkpoint/resume -> implemented
11. Formal benchmark evaluation -> pending (next phase)

## 3) Technologies Used
Core stack:
- Python
- ChromaDB (vector database)
- sentence-transformers (`all-MiniLM-L6-v2`) for embeddings
- OpenAI API (`gpt-5` and earlier tests with `gpt-5-mini`)
- Neo4j AuraDB Free (knowledge graph)
- Cypher (graph queries)

Data & pipeline artifacts:
- JSONL intermediate files for deterministic step-by-step processing
- Chunk metadata/anchors for legal filters and traceability

Infrastructure:
- Local Python virtual environment
- Git/GitHub for versioning
- Google Drive link for dataset sharing (not pushing raw data to GitHub)

## 4) Why Each Step Exists
### Step 1: Build Corpus (`step1_build_corpus.py`)
Why:
- Raw legal files are messy and inconsistent.
- We need a normalized machine-readable corpus.

Output:
- `data/corpus/docs.jsonl`

### Step 2: Legal Chunking (`step2_legal_chunks.py`)
Why:
- Retrieval works on chunk units, not full large documents.
- Legal QA requires section/article-level anchors.

Output:
- `data/chunks/chunks.jsonl`

### Step 3: Build Vector Index (`step3_build_index.py`)
Why:
- Semantic retrieval needs embeddings + vector DB.
- Enables finding meaning-similar passages even with different wording.

Output:
- Chroma collection in `data/chroma`

### Step 4: Retrieve Context (`step4_query_rag.py`)
Why:
- Isolates retrieval quality before generation.
- Lets us debug top-k hits and filters.

### Step 5: Generate Answer with Citations (`step5_rag_answer.py`)
Why:
- Retrieval alone does not produce readable final answers.
- LLM turns retrieved evidence into concise response with citations.

### Step 6: Build Neo4j KG (`step6_build_neo4j_kg.py`)
Why:
- Vector search may miss critical but structurally related chunks.
- KG adds entity/section/article relationships for graph expansion.

What it creates:
- Nodes: `Document`, `Chunk`, `Entity`, `Section`, `Article`, `Topic`
- Relations: `HAS_CHUNK`, `MENTIONS`, `RELATION`, `MENTIONS_SECTION`, `IN_ARTICLE`, `IN_TOPIC`, `SUBTOPIC_OF`

### Step 7: Topic Hierarchy (`step7_hnmf_topics.py`)
Why:
- Improves interpretability and optional retrieval focus.
- Adds topical structure over chunks.

### Step 7B: TELF/HNMFk adapter (`step7b_telf_hnmfk_topics.py`)
Why:
- Closer alignment with research-style method.
- Kept optional due to library/API variability.

## 5) Architecture (Current)
Two-lane hybrid system:
1. Vector lane:
- chunk -> embedding -> Chroma retrieval -> answer generation

2. Graph lane:
- chunk -> entity/relation extraction -> Neo4j graph
- optional graph expansion (`--use-kg`) before final answer generation

Result:
- Hybrid RAG: vector recall + graph relational expansion.

## 6) What We Validated So Far
### Functional validation done
- End-to-end query-answer flow works.
- Citation output works.
- Neo4j ingestion works and is scaling in batches.
- Checkpoint + resume for long runs works.

### Qualitative comparison observed
Example query:
- "What does Article I of the California Constitution say about due process and equal protection?"

Observation:
- Vector-only run returned insufficient evidence for the specific Section 7 language.
- Vector+KG run pulled additional Article I chunks containing Section 7 text and produced a grounded cited answer.

Interpretation:
- KG expansion improved evidence coverage on this query.

## 7) Current Data/Scale Snapshot
- Source docs parsed: 512
- By type:
  - `ca_education_code`: 480
  - `ca_constitution`: 32
- Chunks indexed in Chroma: ~3948
- Neo4j currently has >11k nodes and >23k relationships (running totals during ingestion)

## 8) Challenges Faced and How We Handled Them
1. Neo4j URI/DNS issues
- Cause: stale/invalid Aura endpoint
- Fix: recopy URI from Aura `Connect -> Python` and verify connectivity

2. Long runtime for KG extraction
- Cause: per-chunk OpenAI calls + growing Neo4j merge costs
- Fixes:
  - parallel extraction workers
  - retries/backoff
  - checkpoint/resume

3. gpt-5 temporary 403 safety limitation on some chunks
- Cause: OpenAI safety policy trigger (`cyber_policy_violation`)
- Mitigation:
  - resume later
  - reduce concurrency
  - use `gpt-5-mini` for bulk recovery if needed

4. KG expansion adding off-target context
- Observation: improved recall but added noise from other articles
- Next fix: stricter filter/rerank on expanded candidates

## 9) Why We Chose This Design (Interview Justification)
Design rationale:
1. JSONL between steps
- deterministic pipeline
- debuggable artifacts
- reproducibility

2. Chroma vector retrieval
- strong semantic recall baseline
- easy top-k + metadata filtering

3. Neo4j KG
- recovers relational/legal links vector search can miss
- improves explainability of retrieved context

4. Citation-grounded generation
- reduces unsupported claims
- easier legal traceability

5. Incremental batch processing
- practical for free-tier/limited resources
- robust to interruptions via checkpoint/resume

## 10) What Is Still Pending
1. Full-corpus KG completion (currently in progress batch-wise)
2. Formal benchmark suite (fixed validation query set)
3. Quantitative metrics report across configurations
4. Better KG expansion precision (filter + rerank)

## 11) Next Steps (Near-Term Plan)
1. Build fixed validation set (50-100 legal queries)
2. Include both single-source and cross-source queries (Constitution + Education Code)
3. Compare these setups:
- vector-only
- vector+KG
- optional vector+KG+topic filtering
- external LLM baselines (ChatGPT, Claude) for reference
4. Report metrics:
- retrieval quality (Recall@k / MRR / nDCG)
- citation correctness
- hallucination/groundedness rate

## 12) Interview One-Liner
"We implemented a custom hybrid legal RAG system (without LangChain/LlamaIndex) that combines Chroma vector retrieval, Neo4j knowledge-graph expansion, and OpenAI citation-grounded generation; the system is functionally complete and now in benchmark/evaluation phase for retrieval quality and hallucination reduction."

## 13) Commands You Can Mention
Vector-only answer:
```bash
python src/step5_rag_answer.py --query "..." --top-k 5 --doc-type ca_constitution --article I
```

Vector+KG answer:
```bash
python src/step5_rag_answer.py --query "..." --top-k 5 --doc-type ca_constitution --article I --use-kg --kg-expand-k 12
```

KG build with resume:
```bash
python src/step6_build_neo4j_kg.py --max-chunks 400 --openai-model gpt-5 --workers 4 --retries 3 --retry-backoff 2 --checkpoint-file data/kg_step6_checkpoint.json --resume --topics-path data/topics/chunk_topics.jsonl
```
