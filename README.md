# California Legal RAG (Education Code Title 1 + CA Constitution)

This project builds a lightweight Retrieval‑Augmented Generation (RAG) pipeline over:
- California Education Code **Title 1**
- California Constitution

It includes:
- Corpus building from local files
- Legal‑aware chunking
- Chroma vector indexing
- Retrieval query CLI
- OpenAI RAG answer generation with citations

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Location

Place your dataset under a single root folder with subfolders:

```
california_dataset/
  california_constitution/
  education_code/
```

Update the root path in `src/step1_build_corpus.py`:

```
ROOT_FOLDER = Path("/path/to/california_dataset")
```

## Dataset

The dataset is hosted on Google Drive:
```
https://drive.google.com/drive/folders/1Lv-H1qTTgoDB7zDV4Go49OdfEPdytfTm?usp=sharing
```

Download the folder and place it locally as:
```
california_dataset/
  california_constitution/
  education_code/
```

## Pipeline Steps

### Step 1 — Build corpus
```bash
python src/step1_build_corpus.py
```

### Step 2 — Chunk legal docs
```bash
python src/step2_legal_chunks.py
```

### Step 3 — Build Chroma index
```bash
python src/step3_build_index.py --reset
```

### Step 4 — Query retrieval
```bash
python src/step4_query_rag.py --query "What does Article IX say about education?" --doc-type ca_constitution --article IX --top-k 5
```

### Step 5 — OpenAI RAG answer

Set your API key **locally** (do not commit this):

```bash
export OPENAI_API_KEY="YOUR_KEY_HERE"
```

Then run:

```bash
python src/step5_rag_answer.py --query "What does Article IX say about education?" --doc-type ca_constitution --article IX --top-k 5
```

## Notes

- The pipeline currently supports **Title 1** of the Education Code only.
- Chroma index is stored in `data/chroma` and is excluded from Git.
- For cleaner retrieval, the corpus build step removes Leginfo page boilerplate.
