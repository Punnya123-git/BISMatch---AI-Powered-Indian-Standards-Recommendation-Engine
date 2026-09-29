# Architecture

This document explains how the SIH 2026 Indian Standards recommendation engine is put
together, what is real today, and where new code belongs.

## 1. Request flow

```
Browser (React)
   │  POST /api/documents/upload         POST /api/recommendations/analyze
   ▼
FastAPI app (app/main.py → app/api/router.py)
   │  routes are thin: parse HTTP, call a service, return a schema
   ▼
Services (app/services)
   ├── DocumentService    validate → store raw → extract → clean → persist processed
   └── RecommendationService
         ├── DocumentService.get_processed_text(document_id)   (optional context)
         ├── RagPipeline.readiness() / retrieve()             (evidence, if indexed)
         └── LLM provider                                     (reasoning, later)
   ▼
Adapters (factories hide concrete vendors)
   ├── document_processing/factory   → pypdf extractor | plain-text extractor
   ├── rag/embeddings/factory        → (unconfigured) | real provider later
   ├── rag/vector_store/factory      → ChromaVectorStore (lazy import) | in-memory
   └── ai/llm/factory                → (unconfigured) | real provider later
```

## 2. Layer responsibilities

| Package | Owns | Must not |
| --- | --- | --- |
| `app/core` | Settings, logging, exceptions, global error handlers | Import services |
| `app/schemas` | Pydantic request/response contracts | Contain business logic |
| `app/models` | Rich domain entities used by repositories | Depend on FastAPI |
| `app/document_processing` | Extractors + text cleaning + entities | Talk HTTP or DB |
| `app/rag` | Chunking, embeddings, vector store, retrieval, pipeline orchestration | Fabricate content |
| `app/ai` | LLM interface, provider registry, prompt templates | Embed RAG logic |
| `app/database` | Connection management + repositories (in-memory today) | Hold API schemas |
| `app/services` | Business rules; the only place that combines the above | Do HTTP parsing |
| `app/api` | Routes, dependency wiring, status codes | Contain business rules |

## 3. The honesty contract (most important design decision)

An AI feature is only trustworthy if the UI can tell the difference between *a real answer*
and *no answer*. Therefore:

* `RecommendationResponse.status` ∈ {`ok`, `placeholder`, `not_configured`,
  `dataset_unavailable`} and the UI renders a banner from it.
* `PipelineInfo` reports the LLM provider, embedding provider, vector store, indexed chunk
  count, catalogue size and a list of human-readable `reasons`.
* `RecommendationService._determine_status()` can only return `ok` when the pipeline is
  ready **and** the reasoning stage produced grounded recommendations. Until then it returns
  `placeholder`/`not_configured`/`dataset_unavailable` with empty `recommendations`.
* Retrieved evidence is returned even when no recommendation is produced, because retrieval
  output is real data; only the reasoning/ranking step is missing.
* Unconfigured providers raise `ProviderNotConfiguredError` rather than returning neutral
  vectors or canned text.

## 4. Document pipeline

1. `_validate` — extension allow-list and size limit (checked before parsing).
2. `_store_raw` — `data/raw/<document_id><ext>` (uuid4 hex id).
3. `extract` — `PdfExtractor` (pypdf) or `PlainTextExtractor`, per-page text.
4. `clean_pages` — whitespace collapsing, de-hyphenation, empty-page removal.
5. `_store_processed` — `data/processed/<document_id>.txt`.
6. Warnings are attached instead of failing: a scanned PDF with no text layer returns a
   warning explaining that OCR is required — never an exception, never invented text.

## 5. RAG components

* `chunking.py` — paragraph-aware chunking with overlap; emits `vector_metadata()` that is
  flat and vector-store friendly (no `None` values, `page_number` preserved).
* `embeddings/` — `EmbeddingProvider` ABC + factory; the only implementation today is the
  honouring "unconfigured" stub.
* `vector_store/` — `VectorStore` ABC, `InMemoryVectorStore` (tests) and
  `ChromaVectorStore`. ChromaDB is imported lazily and given a no-op embedding function so
  it never downloads a default model behind your back; embeddings are always supplied
  explicitly by the pipeline.
* `retrieval.py` / `pipeline.py` — `RagPipeline.readiness()` (can we search at all?) and
  `retrieve()` (top-k chunks with scores).

## 6. Extending it

**Add an LLM provider:** implement `LLMProvider` in `app/ai/llm/providers/`, register it in
the factory's provider map, set `LLM_PROVIDER` + `LLM_MODEL` (+ `LLM_API_KEY`). Health and
`/api/recommendations/analyze` immediately start reporting `ready`-ness.

**Add an embedding provider:** same pattern under `app/rag/embeddings/providers/`; then
index the catalogue and uploaded documents through `RagPipeline`.

**Swap repositories:** implement the repository interface in
`app/database/repositories/` (SQL/pgvector) and return it from the service factory; services
and routes stay untouched.

**Implement the reasoning stage:** replace the body of `RecommendationService.analyze()`'s
decision point so that `RecommendedStandard` objects are built strictly from retrieved
evidence, each carrying `evidence` entries; keep `_determine_status()` as the gate.

## 7. Testing strategy

* `tests/test_*_api.py` — contract tests through `TestClient` (status codes, error codes,
  the honesty contract, upload of a real generated PDF).
* `tests/test_chunking.py`, `tests/test_text_cleaner.py` — pure-unit behaviour.
* `tests/test_standards_service.py` — repository/service semantics with *test-only*
  placeholder records (never shipped catalogue data).
* Fixtures write to a temporary `DATA_DIR`, so tests never touch real uploads.
