# Resume -> Job RAG Matcher TODO

This file tracks the feature in small learning steps. Each step maps to one part of the RAG pipeline in `rag_feature_plan.md`.

## Step 1: Dataset loader

- [x] Create `data/load_dataset.py`.
- [x] Load `batuhanmtl/job-skill-set` from Hugging Face.
- [x] Convert each row into a clean `JobPosting` object.
- [x] Parse `job_skill_set` safely because the dataset stores it as text that looks like a Python list.

Why this matters: RAG quality starts with clean source documents. If job IDs, titles, categories, descriptions, or skills are messy, retrieval and scoring become hard to explain.

## Step 2: Embedding wrapper

- [x] Create `indexing/embed.py`.
- [x] Use OpenAI `text-embedding-3-small`.
- [x] Batch requests so indexing 1.17k jobs is practical.

Why this matters: embeddings turn resumes and job descriptions into vectors. Pinecone can then retrieve semantically similar jobs even when the words are not exact keyword matches.

## Step 3: Pinecone indexing

- [x] Create `indexing/upsert_pinecone.py`.
- [x] Create a Pinecone serverless index if it does not exist.
- [x] Store one vector per job with metadata needed for filtering and RAG prompts.

Why this matters: Pinecone is the vector database. It stores vectors and supports fast nearest-neighbour search with metadata filters such as category.

## Step 4: Retrieval

- [x] Create `rag/retriever.py`.
- [x] Embed the resume query.
- [x] Query Pinecone with optional category filtering.
- [x] Return structured retrieved jobs instead of raw Pinecone responses.

Why this matters: retrieval is the "R" in RAG. It chooses the evidence that the LLM is allowed to use.

## Step 5: Deterministic match scoring

- [x] Create `rag/scorer.py`.
- [x] Implement the weighted formula from the plan.
- [x] Extract resume skill mentions and compare them with JD skills.
- [x] Return score breakdowns for the UI and report.
- [ ] Improve skill matching with LLM skill extraction so semantically equivalent phrases can match, not only exact normalised strings.

Why this matters: the LLM should explain matches, not invent scores. Scores are computed in Python so they are reproducible.

## Step 6: Skill-gap generation

- [x] Create `rag/generator.py`.
- [x] Build a grounded prompt from resume + retrieved jobs + backend scores.
- [x] Ask OpenAI `gpt-4o-mini` for JSON.
- [x] Validate and normalise the JSON before returning it to the API.

Why this matters: generation is where RAG becomes useful to a job seeker. The model summarises why jobs match and which skills are repeatedly missing.

## Step 7: Classifier bridge

- [x] Create `rag/category.py`.
- [x] Support an optional existing sklearn/joblib classifier via `RESUME_CLASSIFIER_PATH`.
- [x] Allow manual `predicted_category` in the API request for demos when the classifier file is not present.
- [ ] Add a real category predictor in this repo

Why this matters: the original category classifier becomes a reranking signal. If the classifier is unavailable in this repo, the RAG feature still runs and clearly reports that no category bonus was applied.

## Step 8: FastAPI endpoint

- [x] Create `api/main.py`.
- [x] Add `POST /match`.
- [x] Serve the local frontend at `/`.

Why this matters: the API coordinates retrieval, scoring, and generation in one workflow.

## Step 9: Local frontend

- [x] Create `frontend/index.html`.
- [x] Add resume input, category filter, top-k control, and ranked results.
- [x] Render score breakdown and skill gaps.

Why this matters: the demo needs to show the real workflow, not just backend code.

## Step 10: Verification

- [x] Add unit tests for scoring and generator fallback logic.
- [ ] Run indexing with real `OPENAI_API_KEY` and `PINECONE_API_KEY`.
- [ ] Smoke-test `/match` with a sample IT resume after indexing.
- [ ] Add evaluation notebook or script for the report.

Why this matters: unit tests prove the deterministic parts. The live smoke test proves the external services and index are configured correctly.
