# Resume Job RAG Matcher

This project implements the second feature described in `rag_feature_plan.md`: a resume-to-job matcher using OpenAI embeddings, Pinecone retrieval, deterministic hybrid scoring, and grounded skill-gap generation.

## Pipeline

1. `data/load_dataset.py` loads `batuhanmtl/job-skill-set` from Hugging Face.
2. `indexing/embed.py` embeds job text with `text-embedding-3-small`.
3. `indexing/upsert_pinecone.py` creates/upserts a Pinecone serverless index.
4. `rag/retriever.py` embeds a resume and retrieves top-k matching jobs.
5. `rag/scorer.py` computes the report-friendly match percentage.
6. `rag/generator.py` asks `gpt-4o-mini` for grounded JSON explanations.
7. `api/main.py` exposes `POST /match` and serves `frontend/index.html`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `OPENAI_API_KEY` and `PINECONE_API_KEY` in `.env`.

If `/match` returns `PINECONE_API_KEY is required for retrieval`, the project is still at the setup stage. Create `.env` first:

```bash
cp .env.example .env
```

Then edit `.env` and replace the placeholder keys with real keys. The app cannot retrieve jobs until Pinecone has both a valid API key and an indexed dataset.

## Index the dataset

For a tiny test index:

```bash
python -m indexing.upsert_pinecone --limit 20
```

For the full dataset:

```bash
python -m indexing.upsert_pinecone
```

The index stores one vector per job, with metadata for category filtering and RAG prompts.

## Run the app

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Test deterministic logic

```bash
pytest
```

These tests do not call OpenAI or Pinecone. They verify the scoring formula and the deterministic generator fallback.

## Classifier bridge

If feature 1 has a trained sklearn pipeline, export it with `joblib` and set:

```bash
RESUME_CLASSIFIER_PATH=/absolute/path/to/category_classifier.joblib
```

The object must support:

```python
model.predict([resume_text])
```

and return one of `HR`, `IT`, `Business-Dev`, `Finance`, or `Sales`. If no classifier path is provided, the API still works, but category bonus is only applied when the request supplies `predicted_category`.
