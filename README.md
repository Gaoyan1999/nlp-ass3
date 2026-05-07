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

## Category classifier

The API automatically loads the saved feature 1 classifier from `saved_models/`
through `load_model.py`. By default it uses the best model recorded in
`saved_models/metadata.json`, currently `BiLSTM`, and normalises predictions to:

```text
HR, INFORMATION-TECHNOLOGY, BUSINESS-DEVELOPMENT, FINANCE, SALES
```

To force a different saved model:

```bash
RESUME_CLASSIFIER_MODEL=TextCNN
```

If the full local `saved_models/distilbert/` directory is available:

```bash
RESUME_CLASSIFIER_MODEL=DistilBERT
```

If feature 1 has a separate trained sklearn pipeline, export it with `joblib`
and set this fallback path:

```bash
RESUME_CLASSIFIER_PATH=/absolute/path/to/category_classifier.joblib
```

The object must support:

```python
model.predict([resume_text])
```

and return one of the supported category labels. If no saved model or fallback
classifier is available, the API still works, but category bonus is only applied
when the request supplies `predicted_category`.
