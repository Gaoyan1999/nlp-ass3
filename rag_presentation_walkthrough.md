# Resume-to-Job RAG Matcher: Presentation Walkthrough

## 1. Problem

Job seekers often struggle to identify which job descriptions best match their resume. Traditional keyword search can miss good matches because the same skill can be written in different ways.

Example:

- Resume says: `dashboard reporting`
- Job description says: `data visualisation`

A keyword system may treat these as different, but a semantic system can recognise that they are related.

Our feature solves this by combining:

- semantic retrieval with embeddings,
- Pinecone vector search,
- deterministic match scoring,
- LLM-generated skill-gap explanations.

## 2. Dataset

We use the Hugging Face dataset:

```text
batuhanmtl/job-skill-set
```

The dataset contains 1,167 job postings.

Each row has:

```text
job_id
category
job_title
job_description
job_skill_set
```

The categories are:

```text
HR
INFORMATION-TECHNOLOGY
BUSINESS-DEVELOPMENT
FINANCE
SALES
```

The dataset is loaded in:

```text
data/load_dataset.py
```

We clean each row into a `JobPosting` object, then save all rows into:

```text
data/job_postings.json
```

This JSON file is generated locally and ignored by Git.

## 3. Embedding Job Descriptions

For each job, we create one embedding text:

```python
job_title + job_description + job_skill_set
```

In code:

```python
def embedding_text(self) -> str:
    skills = ", ".join(self.job_skill_set)
    return f"{self.job_title}\n{self.job_description}\nSkills: {skills}"
```

We embed this text using:

```text
OpenAI text-embedding-3-small
```

This produces a dense vector with:

```text
1536 dimensions
```

Why these fields?

- `job_title` gives the role signal.
- `job_description` gives semantic meaning.
- `job_skill_set` gives explicit required skills.

We do not embed `job_id` or `category`. Those are stored as metadata.

## 4. Pinecone Vector Database

We store job vectors in Pinecone.

Our setup:

```text
Index: job-skill-rag
Namespace: job-skill-set
Type: dense
Dimension: 1536
Metric: cosine
Records: 1167
```

Each Pinecone record contains:

```python
{
  "id": job_id,
  "values": embedding_vector,
  "metadata": {
    "job_id": "...",
    "category": "...",
    "job_title": "...",
    "job_description": "...",
    "skills": [...]
  }
}
```

The indexing script is:

```text
indexing/upsert_pinecone.py
```

It loads the dataset, embeds jobs, creates the Pinecone index if needed, and upserts vectors.

## 5. Resume Input

The user can provide a resume in two ways:

1. Paste resume text directly.
2. Upload a PDF CV.

For PDF upload, the backend:

1. extracts text using `pdftotext`,
2. converts the extracted text into simple Markdown,
3. sends the Markdown resume into the same matching pipeline.

This is implemented in:

```text
rag/pdf_resume.py
api/main.py
frontend/index.html
```

The Markdown conversion makes the resume easier to inspect and easier to pass into the embedding and LLM steps.

## 6. Retrieval

Retrieval is the "R" in RAG.

The resume is embedded using the same OpenAI embedding model:

```text
resume text or resume Markdown
→ text-embedding-3-small
→ 1536-dimensional vector
```

Then we query Pinecone:

```python
index.query(
    vector=resume_vector,
    top_k=10,
    include_metadata=True,
    namespace="job-skill-set",
    filter=category_filter,
)
```

Pinecone returns the most semantically similar job postings.

The raw Pinecone response includes:

```text
id
metadata
score
namespace
usage
```

Because the index metric is `cosine`, Pinecone's `score` is the cosine similarity between the resume vector and the job vector.

We convert raw Pinecone matches into clean `RetrievedJob` objects:

```python
class RetrievedJob:
    job_id
    category
    job_title
    job_description
    skills
    cosine_sim
```

This keeps the rest of the app independent from Pinecone's raw response format.

## 7. Match Scoring

Retrieval gives semantically similar jobs, but we still need a user-friendly match score.

The final score is deterministic and computed in Python:

```python
match_score = round(
    0.5 * semantic_pct
  + 0.3 * skill_overlap_pct
  + 0.2 * category_bonus,
  1,
)
```

### Semantic Percentage

Pinecone cosine similarity is rescaled:

```python
SEMANTIC_FLOOR = 0.25
SEMANTIC_CEILING = 0.85
```

So:

```text
0.25 cosine → 0%
0.85 cosine → 100%
```

This is necessary because embedding cosine scores rarely reach 1.0.

### Skill Overlap

We compare resume skills against job-required skills:

```python
matched_skills = jd_skills & resume_skills
missing_skills = jd_skills - resume_skills
```

Current implementation uses normalised string matching.

Known limitation:

```text
"dashboard reporting" may not match "data visualisation"
```

Future TODO:

```text
Use LLM skill extraction to match semantically equivalent skills.
```

### Category Bonus

If the predicted resume category matches the job category:

```text
category_bonus = 100
```

Otherwise:

```text
category_bonus = 0
```

This connects the first feature, category classification, with the RAG matcher.

Current TODO:

```text
Add a real category predictor in this repo.
```

## 8. LLM Skill-Gap Generation

After retrieval and scoring, we use `gpt-4o-mini` to generate readable explanations.

The LLM receives:

- resume text,
- retrieved jobs,
- backend match scores,
- matched skills,
- missing skills,
- job description excerpts.

The system prompt says:

```text
Use only the provided resume and retrieved job evidence.
Return JSON.
Do not fabricate numeric scores.
The backend has already computed the scores.
```

The LLM returns JSON with:

```text
ranked_matches
skill_gaps
notes
```

Example output:

```json
{
  "ranked_matches": [
    {
      "job_id": "3902861423",
      "job_title": "Information Technology Programmer",
      "why_match": "The candidate has SQL and communication experience...",
      "missing_skills": ["Power BI", "DAX Queries", "Excel"]
    }
  ],
  "skill_gaps": [
    {
      "skill": "Power BI",
      "evidence_job_ids": ["3902861423"],
      "suggestion": "Add project evidence or training in Power BI."
    }
  ]
}
```

If the LLM output is malformed, the backend uses a deterministic fallback so the API still returns useful results.

## 9. Frontend Demo

The frontend is a local HTML interface served by FastAPI.

Main features:

- paste resume text,
- upload PDF CV,
- select optional category filter,
- set top-k results,
- enable or disable LLM explanation,
- view ranked job cards,
- inspect score breakdown,
- preview full job description in a modal,
- save the latest result in browser localStorage.

Each result card shows:

```text
job title
category
job ID
match score
semantic score
skill overlap
category bonus
matched skills
missing skills
LLM explanation
JD preview
```

## 10. End-to-End Workflow

The complete system works like this:

```text
PDF CV or pasted resume
        ↓
resume Markdown / text
        ↓
OpenAI embedding
        ↓
Pinecone vector search
        ↓
top-k retrieved jobs
        ↓
Python match scoring
        ↓
LLM grounded explanation
        ↓
frontend ranked results + skill gaps
```

## 11. Why This Is RAG

RAG means Retrieval-Augmented Generation.

In this project:

- Retrieval: Pinecone retrieves relevant job descriptions.
- Augmentation: retrieved jobs and backend scores are inserted into the LLM prompt.
- Generation: the LLM generates skill-gap explanations grounded in retrieved evidence.

The LLM does not answer from memory. It explains based on retrieved job evidence.

## 12. Technical Strengths

This design is strong because:

- semantic embeddings handle paraphrased job requirements,
- Pinecone provides scalable vector search,
- scoring is deterministic and explainable,
- the LLM is grounded with retrieved evidence,
- PDF upload makes the demo realistic,
- score breakdown makes results auditable.

## 13. Current Limitations

Current limitations:

- skill matching is still mostly exact string matching,
- category predictor is not fully implemented yet,
- PDF formatting is simple and may need improvement for complex layouts,
- evaluation methodology still needs to be finalised.

Planned improvements:

- add TF-IDF + Logistic Regression category predictor,
- use LLM skill extraction for semantic skill matching,
- add evaluation script comparing retrieval quality against a TF-IDF baseline.

## 14. Demo Script

Suggested live demo:

1. Show Pinecone index:
   - index `job-skill-rag`,
   - namespace `job-skill-set`,
   - 1,167 vectors.

2. Open local app:

   ```text
   http://127.0.0.1:8000
   ```

3. Upload a PDF CV.

4. Click `Run matcher`.

5. Explain the result card:
   - match score,
   - semantic similarity,
   - skill overlap,
   - missing skills,
   - JD preview modal.

6. Explain that the LLM generates advice only after retrieval, using retrieved job evidence.

7. Mention future improvements:
   - category predictor,
   - LLM-based skill extraction.
