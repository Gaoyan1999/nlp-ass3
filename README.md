# Resume Job RAG Matcher

This project combines resume category prediction with a RAG-based job matcher. A resume is classified into one of five job categories, embedded with OpenAI, matched against indexed job descriptions in Pinecone, scored with deterministic signals, and optionally explained with an LLM.

## Quick Start

From the project root:

```bash
cd "/Users/daniel/UTS S2/NLP/ass3"
pip3 install -r requirements.txt
set -a; source local.env; set +a
python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Keep the terminal running while using the app. If the terminal process stops, the browser will no longer be able to connect.

If you prefer a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
set -a; source local.env; set +a
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Environment

The app needs keys for retrieval and generation. This repo uses `local.env` locally:

```bash
set -a; source local.env; set +a
```

Required values:

```text
OPENAI_API_KEY
PINECONE_API_KEY
PINECONE_INDEX_NAME
PINECONE_NAMESPACE
```

Optional values:

```text
OPENAI_EMBEDDING_MODEL
OPENAI_CHAT_MODEL
OPENAI_EVAL_MODEL
PINECONE_CLOUD
PINECONE_REGION
RESUME_CLASSIFIER_MODEL
RESUME_CLASSIFIER_MODEL_DIR
RESUME_CLASSIFIER_PATH
```

If `/match` returns `PINECONE_API_KEY is required for retrieval`, the server was started without loading `local.env`.

## Project Flow

1. Offline dataset preparation:
   `data/load_dataset.py` loads the Hugging Face `batuhanmtl/job-skill-set` dataset. Each job record includes `job_id`, `category`, `job_title`, `job_description`, and `job_skill_set`.

2. Offline indexing:
   `indexing/upsert_pinecone.py` embeds each job using `indexing/embed.py` and stores the vectors in Pinecone. Pinecone metadata keeps the job category, title, description, and skills so the app can retrieve and display grounded job details.

3. Resume category prediction:
   `rag/category.py` calls `load_model.py`, which loads the saved feature 1 classifier from `saved_models/`. By default it uses the best model recorded in `saved_models/metadata.json`, currently `BiLSTM`. The prediction is normalised to one of:

```text
HR
INFORMATION-TECHNOLOGY
BUSINESS-DEVELOPMENT
FINANCE
SALES
```

4. Retrieval:
   `rag/retriever.py` embeds the resume and queries Pinecone for the top matching jobs. The user can optionally apply a category filter from the frontend.

5. Deterministic scoring:
   `rag/scorer.py` computes the backend match score:

```text
match_score =
  0.5 * semantic_similarity
  + 0.3 * skill_overlap
  + 0.2 * category_bonus
```

The category bonus is `100` when the predicted resume category matches the job category, otherwise `0`. This is the bridge between the classification feature and the RAG matcher.

6. Explanation and skill gaps:
   `rag/generator.py` asks the configured OpenAI chat model to produce grounded explanations and skill-gap suggestions. If LLM generation is disabled or fails, the app falls back to deterministic explanations from the backend scores.

7. API and frontend:
   `api/main.py` exposes `POST /match`, `POST /match-pdf`, `/categories`, and `/health`. It also serves `frontend/index.html`, where users paste a resume, upload a PDF, choose filters, run matching, see the predicted resume category, and inspect job matches.

8. LLM evaluation:
   When `/match` runs with LLM explanations enabled, the system saves a local eval record to `data/llm_eval_records.jsonl`. The `/eval` page lists these records and can ask a separate evaluator LLM to judge the generated explanation quality.

## API Endpoints

Health check:

```bash
curl http://127.0.0.1:8000/health
```

List supported categories:

```bash
curl http://127.0.0.1:8000/categories
```

Open the LLM eval page:

```text
http://127.0.0.1:8000/eval
```

Run a text resume match:

```bash
curl -X POST http://127.0.0.1:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Software engineer with Python, SQL, React, cloud APIs, and database design experience.",
    "top_k": 5,
    "category_filter": null,
    "predicted_category": null,
    "use_llm": false
  }'
```

## Index the Dataset

Only needed if Pinecone has not already been populated.

For a tiny test index:

```bash
set -a; source local.env; set +a
python3 -m indexing.upsert_pinecone --limit 20
```

For the full dataset:

```bash
set -a; source local.env; set +a
python3 -m indexing.upsert_pinecone
```

## Category Classifier

The saved classifier artifacts live in `saved_models/`. The default path uses:

```text
saved_models/bilstm.pt
saved_models/vocab.pkl
saved_models/label_encoder.pkl
saved_models/metadata.json
```

To force TextCNN:

```bash
RESUME_CLASSIFIER_MODEL=TextCNN
```

If the full local `saved_models/distilbert/` directory is available, DistilBERT can be used locally:

```bash
RESUME_CLASSIFIER_MODEL=DistilBERT
```

DistilBERT files are ignored by Git because they are large optional artifacts. The default BiLSTM/TextCNN models are enough for this app.

If a separate sklearn classifier is available, it can be used as a fallback:

```bash
RESUME_CLASSIFIER_PATH=/absolute/path/to/category_classifier.joblib
```

The object must support:

```python
model.predict([resume_text])
```

## Tests

Run:

```bash
python3 -m pytest
```

The tests do not call OpenAI or Pinecone. They cover category normalisation, scoring, and deterministic generator fallback.

## LLM Evaluation

The eval page is designed for the report's generation-quality evaluation. It does not evaluate the category classifier.

Workflow:

1. Start the app with `local.env`.
2. Open `http://127.0.0.1:8000`.
3. Keep `Generate LLM explanation` checked and run the matcher.
4. Choose the OpenAI explanation model, for example `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.2`, `gpt-4o-mini`, `gpt-4o`, or `gpt-4.1-mini`.
5. Open `http://127.0.0.1:8000/eval`.
6. Select a saved generation record.
7. Click `Run LLM eval`.

Each eval record stores which model generated the explanation. The evaluator uses `OPENAI_EVAL_MODEL` when set, otherwise it defaults to `gpt-4o-mini`.

The evaluator LLM scores:

```text
Groundedness: whether claims are supported by the resume, retrieved jobs, skills, or backend scores.
Helpfulness: whether the explanation helps the user understand the match.
No hallucination: whether unsupported skills, credentials, job facts, or numeric claims are avoided.
Actionability: whether skill-gap suggestions are specific and tied to missing skills.
Evidence use: whether the explanation clearly uses backend evidence.
```

The evaluator returns a 1-5 score for each criterion, a pass/review/fail recommendation, and a short summary. The app computes the total score by summing the five criteria, so the maximum score is 25. Records are stored locally in `data/llm_eval_records.jsonl`, which is ignored by Git because it may contain resume text.

## Report Notes

For the report, describe the system as a two-feature integration:

- Feature 1 predicts the resume's job category using a saved classifier.
- Feature 2 retrieves semantically similar jobs from Pinecone using OpenAI embeddings.
- The final score combines semantic similarity, explicit skill overlap, and category agreement.
- The frontend shows both the predicted resume category and each job's score breakdown, making the result explainable instead of only showing a final percentage.
- The LLM is used only for narrative explanations and skill-gap wording; the numeric score is computed deterministically in Python.
- LLM output is evaluated separately by an evaluator LLM for groundedness, helpfulness, hallucination control, actionability, and evidence use.
