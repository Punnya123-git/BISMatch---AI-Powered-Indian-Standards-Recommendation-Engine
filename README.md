# SIH 2026 — Indian Standards Recommendation Engine

Backend and frontend **foundation** for an AI-assisted engine that maps a procurement
requirement (tender text or an uploaded specification document) to the Indian Standards
that apply to it.

> **Honesty first.** This repository contains the *architecture* plus a small **verified**
> standards dataset - it is not a working AI. No standard number is ever invented in the UI
> or the API: with no configured LLM, no embedding provider and no verified standards
> dataset, the API returns an explicit `status`
> (`not_configured` / `dataset_unavailable`) plus the reason, and the UI displays exactly
> that. Wiring a real provider is a deliberate, verifiable step (see [Roadmap](#8-roadmap)).
>
> The bundled dataset is a **limited prototype subset** for rotating electrical machines. It
> is *not* the complete BIS catalogue, and fields the source material did not provide are
> `null` rather than guessed (see [section 4](#4-the-verified-standards-dataset)).

---

## 1. What is implemented

| Layer | Status |
| --- | --- |
| FastAPI app, routers, error handling, CORS, logging | ✅ working |
| Health endpoint reporting per-dependency configuration | ✅ working (also reports catalogue + index size) |
| Document upload → store → extract → clean → persist | ✅ working (PDF via `pypdf`, plain text) |
| Document ingestion/registry (in-memory repository) | ✅ working (swappable) |
| **Verified Indian Standards dataset** (20 records, rotating electrical machines) | ✅ bundled at `data/standards/standards.json` |
| **Dataset schema + validator** (types, uniqueness, year/designation, provenance) | ✅ fails loudly with a full problem list |
| **Dataset loader → repository → `GET /api/standards`** | ✅ working |
| **Standard record → searchable document → chunk conversion** | ✅ working (verified fields only) |
| Chunking, embedding/LLM/vector-store abstractions | ✅ interfaces + honest "unconfigured" stubs |
| ChromaDB adapter | ✅ implemented, resolved lazily at first use |
| **Standards indexing command** | ✅ `python -m app.rag.index_standards` (needs an embedding provider) |
| Embedding provider | ⛔ none configured — indexing reports `embedding_provider_not_configured` |
| Recommendation endpoint contract | ✅ working, always returns an explicit status |
| Recommendation reasoning (LLM + catalogue grounding) | ⛔ not implemented — returns `placeholder` |
| React UI (upload, requirement input, results, evidence, engine + catalogue status) | ✅ working |

## 2. Repository layout

```
sih/
├── backend/                     FastAPI service
│   ├── app/
│   │   ├── api/                 routers + DI (`dependencies.py`, `router.py`)
│   │   ├── ai/                  LLM abstraction + prompt templates
│   │   ├── core/                settings, logging, exceptions, error handlers
│   │   ├── database/            connection + repositories (in-memory now, SQL later)
│   │   ├── document_processing/ extractors (pdf/text) + cleaning + entities
│   │   ├── models/              persistence-ready domain entities
│   │   ├── rag/                 chunking, embeddings, vector store, retrieval, pipeline
│   │   │                        + `index_standards.py` (explicit indexing CLI)
│   │   ├── schemas/             Pydantic request/response contracts
│   │   ├── services/            business logic used by the routes
│   │   ├── standards/           dataset schema, validator, loader, documents, indexing
│   │   └── main.py              app factory (`app.main:app`)
│   ├── tests/                   pytest suite (API + units)
│   ├── .env.example
│   └── requirements*.txt
├── frontend/                    Vite + React (JavaScript, no build step needed to read)
│   └── src/
│       ├── components/          presentational building blocks
│       ├── hooks/               useBackendHealth, useAnalyzeRequirement
│       ├── pages/HomePage.jsx   page composition
│       ├── services/            apiClient + one module per API area
│       └── utils/               constants + formatters
├── data/
│   ├── standards/standards.json verified prototype catalogue (committed)
│   ├── raw/                     uploaded files (git-ignored)
│   ├── processed/              extracted text (git-ignored)
│   └── vector_store/           ChromaDB index (git-ignored)
└── docs/architecture.md         design decisions and extension points
```

## 3. Quick start

### 3.1 Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env        # sets STANDARDS_DATASET_PATH to the bundled dataset
uvicorn app.main:app --reload --port 8000
```

* API root: <http://127.0.0.1:8000/>
* Interactive docs: <http://127.0.0.1:8000/docs>
* Health: <http://127.0.0.1:8000/api/health>
* Catalogue: <http://127.0.0.1:8000/api/standards>

Without `backend/.env` (or with an empty `STANDARDS_DATASET_PATH`) the API still starts and
reports `dataset_unavailable` instead of pretending to have a catalogue.

Run the tests:

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest
```

Optional RAG extras (ChromaDB): `pip install -r requirements-rag.txt`.

### 3.2 Frontend

```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173
```

Vite proxies `/api/*` to `http://127.0.0.1:8000` (see `vite.config.js`), so no CORS
configuration is required in development. For a build:

```powershell
npm run build      # outputs frontend/dist
```

---

## 4. The verified standards dataset

### 4.1 What is in the repository

`data/standards/standards.json` holds **20 Indian Standards related to rotating electrical
machines**, collected from official BIS material.

> ⚠️ **This is a limited prototype subset, not the complete BIS catalogue.** It exists so the
> ingestion → validation → document → index path can be demonstrated and tested end to end.

Only verified information is stored. The source material supplied the designations and the
product area, so:

* `standard_number` and `product_category` are populated;
* `year` is the year suffix of that same designation (for example `IS 12615:2018` → `2018`),
  and the validator rejects any year that contradicts it;
* `title`, `scope`, `status`, `revision`, `amendments`, `related_standards`, `references` and
  `certification` were **not** provided, so they are `null` / `[]`;
* `source.organization` is recorded; `source.url` / `source.document` stay `null` until the
  exact source document is recorded.

Nothing is inferred, and `backend/tests/test_shipped_standards_dataset.py` fails if a record
ever gains a title, scope or amendment that the source material did not provide.

### 4.2 Schema

```json
{
  "schema_version": "1.0",
  "dataset": { "name": "...", "version": "1.0.0", "record_count": 20, "notes": ["..."] },
  "standards": [
    {
      "standard_number": "IS 12615:2018",
      "title": null,
      "scope": null,
      "product_category": "rotating electrical machines",
      "status": null,
      "revision": null,
      "year": 2018,
      "amendments": [],
      "related_standards": [],
      "references": [],
      "certification": null,
      "source": {
        "organization": "Bureau of Indian Standards",
        "url": null,
        "document": null
      }
    }
  ]
}
```

`references` entries are `{ "code", "title", "relation" }`; `certification`, when present, is
`{ "scheme", "marking", "notes" }`. Unknown fields are rejected, never ignored.

The validator (`app/standards/dataset_schema.py`) checks that `standard_number` is present,
unique, single-line and numeric; that every field has the right type; that `amendments`,
`related_standards` and `references` are arrays; that `source` is a structured object with an
organization; that nothing references itself; and that `dataset.record_count` matches the
array. Any problem raises `DatasetValidationError` (`code: standards_dataset_invalid`) with
the **full list** of problems - invalid records are never skipped silently.

The validator runs when the API starts, whenever the catalogue is requested and in the
indexing command, so a broken dataset can never be used half-way.

### 4.3 Configure the path

```dotenv
# backend/.env
STANDARDS_DATASET_PATH=../data/standards/standards.json
```

Relative paths are resolved against `backend/` first, then the project root; absolute paths
are used as-is. Paths are configuration only - they are never exposed through the API.

### 4.4 Index (and validate) from the command line

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.rag.index_standards [--dataset PATH] [--reset]
```

The command performs, and prints, every step: load → validate → number of standards →
prepare searchable documents → check the embedding configuration → index into ChromaDB →
final chunk count.

Indexing is **never** triggered by the API at start-up: a demo cannot silently depend on
vectors that were never created. Build the index explicitly, then check
`GET /api/health` → `indexed_chunks`.

### 4.5 When embeddings are not configured

With `EMBEDDING_PROVIDER=unconfigured` (the default) the command exits with code `2` and
states exactly what happened:

```
3. Standards loaded        : 20
4. Documents prepared      : 20
   Chunks prepared         : 20
5. Embedding provider      : unconfigured (not configured)
6. Result                  : BLOCKED: embedding provider not configured
7. Indexed chunks in store : 0

Nothing was indexed: no embedding provider is configured, so the standards could not be
embedded. The dataset itself is valid and the searchable documents were prepared.

  - Set EMBEDDING_PROVIDER, EMBEDDING_MODEL and EMBEDDING_API_KEY in backend/.env, then run this command again.
  - No placeholder or fake embeddings are ever generated.
```

No dummy vectors are written, so `/api/health` keeps reporting `indexed_chunks: 0` and the UI
keeps saying that semantic search is unavailable.

### 4.6 Searchable text and metadata

A record becomes text using **only** the fields the dataset holds - one line per populated
field, in a fixed order (`Standard Number`, `Title`, `Scope`, `Product Category`, `Status`,
`Revision`, `Year`, `Amendments`, `Related Standards`, `References`, `Certification`,
`Source Organization`). A missing field produces no line at all, so the indexed text can
never contain information the dataset does not hold.

Every chunk carries `standard_number`, `standard_code`, `product_category`, `source`,
`dataset_version`, `year` and `record_type: "standard"` as vector metadata, and its id is a
path/URL-safe slug of the designation (`standard::IS-12615-2018`).

## 5. API contract

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service banner (name, version, docs link) |
| `GET` | `/api/health` | Service + per-dependency configuration status, catalogue and index size |
| `GET` | `/api/standards` | Loaded standards catalogue (safe metadata + records) |
| `POST` | `/api/documents/upload` | Multipart upload; returns metadata + cleaned text |
| `POST` | `/api/recommendations/analyze` | Requirement (+ optional `document_id`) → result |

### 5.1 `GET /api/health`

```json
{
  "status": "ok",
  "service": "SIH PS-108 Standards Recommendation API",
  "version": "0.1.0",
  "environment": "development",
  "dependencies": [
    { "name": "llm_provider", "configured": false, "detail": "unconfigured (not configured)" },
    { "name": "embedding_provider", "configured": false, "detail": "unconfigured (not configured)" },
    { "name": "vector_store", "configured": true, "detail": "chroma (0 records)" },
    { "name": "standards_dataset", "configured": true, "detail": "20 standards loaded from dataset version 1.0.0 (20 record(s) in the dataset file)." },
    { "name": "sql_database", "configured": false, "detail": "not configured" }
  ],
  "standards_available": 20,
  "indexed_chunks": 0,
  "dataset": {
    "available": true,
    "count": 20,
    "dataset_version": "1.0.0",
    "organization": "Bureau of Indian Standards",
    "product_category": "rotating electrical machines",
    "message": "20 standards loaded from dataset version 1.0.0 (20 record(s) in the dataset file).",
    "issues": []
  }
}
```

`dataset.available: false` with a populated `issues` list is the honest "no usable catalogue"
answer; the endpoint itself still returns `200`.

### 5.2 `GET /api/standards`

Query parameters: `limit` (1-200, default 50), `offset` (default 0) and `q` (optional text
query over designation and whatever wording the dataset actually holds).

```json
{
  "available": true,
  "count": 20,
  "dataset": { "available": true, "count": 20, "dataset_version": "1.0.0", "...": "..." },
  "standards": [
    {
      "code": "IS 12615:2018",
      "title": null,
      "summary": null,
      "category": "rotating electrical machines",
      "keywords": [],
      "scope": null,
      "version": { "year": 2018, "revision": null, "amendments": [], "status": null },
      "certification": [],
      "references": [],
      "related_standards": [],
      "source": {
        "organization": "Bureau of Indian Standards",
        "url": null,
        "document": null
      }
    }
  ]
}
```

`count` is the size of the loaded catalogue, while `standards` is the requested page (or the
matches for `q`). No filesystem path is ever returned. Failures are explicit: `503`
`standards_dataset_unavailable` when nothing is configured, `500` `standards_dataset_invalid`
(with the validator's `details` list) when the file is broken.

### 5.3 `POST /api/documents/upload`

* `multipart/form-data` with a single `file` field.
* Accepted: `.pdf`, `.txt`, `.md` (max size from `MAX_UPLOAD_SIZE_MB`).
* Response `201`:

```json
{
  "metadata": {
    "document_id": "9f2c…",
    "filename": "tender.pdf",
    "content_type": "application/pdf",
    "size_bytes": 84213,
    "uploaded_at": "2026-01-01T10:00:00Z",
    "stored_path": "…/data/raw/9f2c….pdf"
  },
  "document": {
    "document_id": "9f2c…",
    "filename": "tender.pdf",
    "page_count": 12,
    "character_count": 48120,
    "word_count": 7954,
    "text": "…cleaned full text…",
    "pages": [{ "page_number": 1, "text": "…", "character_count": 3021 }],
    "warnings": []
  }
}
```

### 4.3 `POST /api/recommendations/analyze`

Request:

```json
{ "requirement": "Supply of 43 grade OPC in 50 kg bags", "document_id": null, "top_k": 10 }
```

Response (`status` is the source of truth for the UI):

```json
{
  "status": "dataset_unavailable",
  "message": "No Indian Standards dataset is loaded, so no recommendation can be made.",
  "requirement": "Supply of 43 grade OPC in 50 kg bags",
  "document_id": null,
  "recommendations": [],
  "retrieved_evidence": [],
  "pipeline": {
    "ready": false,
    "llm_provider": "unconfigured",
    "embedding_provider": "unconfigured",
    "vector_store": "chroma",
    "indexed_chunks": 0,
    "standards_available": 0,
    "reasons": ["No embedding provider is configured (EMBEDDING_PROVIDER / …)."]
  },
  "warnings": []
}
```

`status` values: `ok` (real, grounded recommendations), `placeholder` (pipeline ready but
reasoning not implemented), `not_configured` (a required provider is missing),
`dataset_unavailable` (no catalogue/index loaded).

### 4.4 Errors

Every failure returns a uniform payload and never a stack trace:

```json
{ "code": "unsupported_file_type", "message": "…", "details": null }
```

Codes in use: `validation_error`, `unsupported_file_type`, `file_too_large`,
`document_processing_error`, `not_found`, `configuration_error`, `provider_not_configured`,
`service_unavailable`, `internal_error`.

---

## 5. Configuration

All settings come from environment variables / `backend/.env` (template:
`backend/.env.example`). Nothing is hard-coded, and every value has a safe default.

| Variable | Default | Meaning |
| --- | --- | --- |
| `APP_NAME`, `APP_VERSION` | see config | Service metadata |
| `ENVIRONMENT`, `DEBUG`, `LOG_LEVEL` | `development`, `true`, `INFO` | Runtime mode |
| `CORS_ORIGINS` | Vite dev origins | Allowed browser origins |
| `DATA_DIR` | `../data` | Root for raw/processed uploads and the vector index |
| `MAX_UPLOAD_SIZE_MB` | `25` | Upload limit enforced before reading the file |
| `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` | `unconfigured` | Reasoning model |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_API_KEY` | `unconfigured` | Embeddings |
| `VECTOR_STORE_PROVIDER`, `VECTOR_COLLECTION_NAME`, `VECTOR_DB_PATH` | `chroma` | Vector index |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K` | `1200`, `200`, `10` | RAG tuning |
| `DATABASE_URL` | empty | Reserved for PostgreSQL (repositories are pluggable) |
| `STANDARDS_DATASET_PATH` | empty | Path to a verified standards dataset |

Frontend: `VITE_API_BASE_URL` (empty = same origin / dev proxy).

## 6. Rules this codebase follows

1. **Never fabricate standards.** No code path may emit a standard number that did not come
   from the loaded catalogue or retrieved document text.
2. **Unknown state is an answer.** Missing provider/settings produce an explicit status and
   reason, not an empty-but-plausible recommendation list.
3. **Providers fail loudly and early.** Unconfigured LLM/embedding providers raise
   `ProviderNotConfiguredError` instead of returning dummy vectors or canned text.
4. **Routers are thin.** HTTP handling lives in `app/api`, business logic in `app/services`.
5. **Everything is replaceable.** LLM, embeddings, vector store and repositories are all
   chosen through factories, so no module imports a concrete vendor directly.
6. **Evidence is traceable.** Retrieved chunks are surfaced verbatim with source, page and
   score, so every future recommendation can be audited.

## 7. Roadmap

| Step | Work | Where |
| --- | --- | --- |
| 1 | Build a *verified* standards dataset (BIS catalogue metadata, scope, amendments, references) and load it into a repository | `app/database/repositories/`, `data/` |
| 2 | Implement a real embedding provider and index the catalogue + uploaded documents | `app/rag/embeddings/providers/`, `app/rag/pipeline.py` |
| 3 | Implement the LLM provider, prompt with retrieved evidence only, and require citations per recommendation | `app/ai/llm/providers/`, `app/ai/prompts/` |
| 4 | Implement the ranking/reasoning stage and flip `status` to `ok` only when grounded | `app/services/recommendation_service.py` |
| 5 | Add OCR (scanned tenders), job queue for large files, and Postgres/pgvector persistence | new modules + `app/database/` |
| 6 | Add evaluation harness (precision/recall against expert-labelled pairs) before any demo claim | `backend/tests/` |

## 8. Common commands

```powershell
# Backend: run tests, start dev server
cd backend
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# Frontend: dev server, production build, preview
cd frontend
npm run dev
npm run build
npm run preview
```
