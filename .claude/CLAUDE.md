# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This repo contains a single active project under `.claude/hackathon/`. The top-level directory also contains:

- `.claude/legacy_stack/` — empty placeholder directories (`apr-agreement-clustering/`, `apr-fields-processor/`) kept as references to the legacy V1 stack that V2 replaces. Do not expect code here.
- Design docs at the `.claude/` root: `README.md`, `cluster_v2_engineering.md`, `cluster_v2_product.md`, `clustering_v2_roadmap.md`, `hackathon_plan.md`.

All code, commands, and development work happen inside `.claude/hackathon/`. Treat that as the project root — `config.py` computes `PROJECT_ROOT` as its own parent directory.

## Common Commands

All commands run from `.claude/hackathon/`.

Setup:

```bash
./setup.sh                          # Installs Poetry + deps + creates data dirs
poetry install                      # Re-install deps only
conda env create -f environment-gpu.yml && conda activate clustering-v2-gpu  # GPU path (RAPIDS cuML)
```

Run the Streamlit UI:

```bash
poetry run streamlit run app.py --server.port 8501
```

Run the FastAPI server (cluster job orchestration):

```bash
poetry run uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Run pipelines directly:

```bash
# Full production featurizer + clustering (persists BERTopic artifacts under data/artifacts/<account_id>/)
poetry run python -m core.cluster_pipeline /path/to/pdfs --account-id acct_001
poetry run python scripts/run_pipeline.py 1000       # wrapper around cluster_pipeline

# PDF-only ingest → cluster → persist (no LLM)
poetry run python -m core.pdf_pipeline /path/to/pdfs

# EVoC-based pipeline (replaces UMAP+HDBSCAN with EVoC multi-layer clustering)
poetry run python -m core.evoc_pipeline /path/to/pdfs --account-id acct_001 --mode clause      # flat
poetry run python -m core.evoc_pipeline /path/to/pdfs --account-id acct_001 --mode macro-micro # hierarchical
poetry run python -m core.evoc_pipeline /path/to/pdfs --account-id acct_001 --mode hybrid

# Original 7-stage CPU pipeline (MiniLM + sklearn + FAISS + LLM)
poetry run python -c "from core.pipeline import run_full_pipeline; run_full_pipeline()"

# GPU pipeline (cuML + Ray + LanceDB)
poetry run python -c "from core.gpu_pipeline import run_gpu_pipeline; run_gpu_pipeline()"
```

Tests / validation (no pytest suite — these scripts are the test harness):

```bash
poetry run python scripts/validate_pipeline.py              # 8 infrastructure tests
poetry run python scripts/validate_pipeline.py --skip-llm   # infra only, no LLM API calls
poetry run python scripts/test_gpu_stack.py                  # 7 GPU component tests
```

Evaluation harness (runs against an existing DuckDB produced by a pipeline run):

```bash
poetry run python -m eval.runner data/evoc_200_refined.duckdb
poetry run python -m eval.runner data/evoc_200_refined.duckdb --gold eval/gold_standard.json --judge --judge-sample 30
poetry run python -m eval.runner <db> --baseline eval/baseline_results.json --save-baseline eval/baseline_results.json
```

## Architecture

### Two storage stores, split by purpose

- **DuckDB** (`core/store.py`, `ClusteringStore`) — system of record. Holds metadata, relationships, and pipeline state across ~12 tables: `agreements`, `domains`, `clusters`, `chunks`, `clauses`, `cluster_assignments`, `field_definitions`, `extractions`, `corrections`, `pipeline_runs`, `cluster_centroids`, `novel_documents`, `composite_definitions`. Embeddings live here as BLOBs alongside metadata. `_init_schema()` runs on every construction and is idempotent — safe to extend.
- **LanceDB** (`core/lancedb_store.py`, `LanceVectorStore`) — persistent ANN index with metadata filtering. Two tables: `chunks` (chunk vectors + domain/cluster IDs) and `documents` (summary vectors). Used by the GPU and EVoC paths.

Default DuckDB path: `data/evoc_200_refined.duckdb` (override via `DB_PATH` env var). Default LanceDB path: `data/vectors.lance` (override via `LANCEDB_PATH`).

### Multiple pipeline modules — pick the right one

`core/` contains several pipeline orchestrators that have accumulated over the life of the project. They share components but differ in algorithms, storage targets, and whether LLMs are called:

| Module                | What it does                                                                                                                                                                     | Storage                           | LLM      |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | -------- |
| `pipeline.py`         | Original 7-stage CPU MVP: parse → chunk → embed (MiniLM) → macro BERTopic → micro BERTopic → topic merge → field discovery → extraction                                          | DuckDB                            | yes      |
| `gpu_pipeline.py`     | GPU variant: Nomic 768-dim + cuML UMAP/HDBSCAN + Ray-parallel Bayesian optimization                                                                                              | DuckDB + LanceDB                  | yes      |
| `pdf_pipeline.py`     | Ingest-only: PDF → chunk → embed → macro/micro cluster → persist. Auto-labels clusters from keywords                                                                             | DuckDB + LanceDB                  | no       |
| `cluster_pipeline.py` | Production featurizer + clustering with **model persistence** (BERTopic safetensors per-domain, embeddings.npz, run_manifest.json) under `data/artifacts/<account_id>/<run_id>/` | DuckDB + LanceDB + disk artifacts | optional |
| `evoc_pipeline.py`    | Uses EVoC (purpose-built embedding clustering) instead of UMAP+HDBSCAN. Three modes: `clause`, `macro-micro`, `hybrid`. Multi-layer clusters.                                    | DuckDB + LanceDB                  | optional |
| `clause_pipeline.py`  | Flat clause-type clustering variant                                                                                                                                              | DuckDB                            | —        |

When adding features, prefer `cluster_pipeline.py` / `evoc_pipeline.py` for anything "production-like" (these persist artifacts and are driven by the FastAPI server and `scripts/run_pipeline.py`). `pipeline.py` and `gpu_pipeline.py` are the older end-to-end demos.

### Core flow (two-stage hierarchical clustering)

This is the central V2 idea and differs from legacy V1 (which was flat BERTopic):

1. **Parse** — `pdf_parser.py` uses PyMuPDF to emit `.txt` + manifest under `data/parsed_texts/`.
2. **Chunk** — `chunker.py` provides `sliding_window_chunk` (default 256 tokens, 64 overlap) and `semantic_chunk` (legal-structure aware). Also produces a short document summary (`SUMMARY_MAX_CHARS`, default 2000).
3. **Embed** — `embedder.py` (MiniLM-L6-v2, 384-dim, CPU) or `nomic_embedder.py` (Nomic-embed-text-v1.5, 768-dim, GPU). `clause_embedder.py` is a clause-context-aware variant (FIELD-110).
4. **Macro cluster** — `macro_clusterer.py` clusters **document summary embeddings** into domains (e.g., NDA, SaaS). BERTopic + UMAP + HDBSCAN.
5. **Per-domain chunk routing** — each chunk is assigned to its document's domain.
6. **Micro cluster** — `micro_clusterer.py` runs BERTopic per-domain on clause embeddings → clause-type clusters.
7. **Topic merge** — `topic_merger.py` (FIELD-310) collapses near-duplicate topics with hierarchical similarity dedup (thresholds: `MERGE_SIMILARITY_THRESHOLD=0.82`, `SYNONYM_SIMILARITY_THRESHOLD=0.78`).
8. **Assignment** — `cluster_assigner.py` (FIELD-801) does incremental assignment with confidence tiers (`HIGH_CONFIDENCE_THRESHOLD=0.85`, `TENTATIVE_THRESHOLD=0.60`). Below tentative → buffered in `novel_documents`.
9. **(Optional) LLM labeling, field discovery, field extraction** — `llm_client.py` routes to OpenAI / Azure OpenAI / Ollama / LiteLLM based on `LLM_BACKEND`. Prompts live in `prompts/`.

### GPU path specifics

- `cuml_wrappers.py` provides `make_umap()` / `make_hdbscan()` that use RAPIDS cuML when available and **gracefully fall back to sklearn/hdbscan on CPU**. Call `cuml_available()` to branch.
- `ray_optimizer.py` uses `@ray.remote` tasks + Optuna TPE for parallel Bayesian hyperparameter optimization (default 50 trials, 4 parallel). Gated by `USE_RAY` env.
- `scoring.py` implements the z-score + 4-penalty composite objective ported from legacy `apr-agreement-clustering`.
- `environment-gpu.yml` is a conda env for RAPIDS. `pyproject.toml` comments note that `cuml/cudf/cupy` must come from conda (not PyPI), and the code falls back to CPU if they're missing.

### Retrieval & extraction

- `retrieval.py` — hybrid BM25 + dense (FAISS/LanceDB) + CrossEncoder reranker + RRF fusion. New in V2 (V1 was dense-only).
- `field_discovery.py` — LLM discovers field schema per cluster.
- `extractor.py` — cluster-conditioned LLM extraction (reduces tokens by reusing the cluster's field schema).
- `composite.py` — derived/computed fields with cascade propagation.

### UI

`app.py` is a Streamlit app with sidebar navigation grouped into 5 sections (Discover / Explore / Author / Analyze / Configure). Each page is a module under `ui/` exposing `render(store)`. The single shared `ClusteringStore` is cached via `@st.cache_resource`. Adding a page = creating `ui/<name>_page.py` with `render(store)` and registering it in the `SECTIONS` dict in `app.py`.

`api_server.py` is a parallel FastAPI surface that triggers `cluster_pipeline.run_cluster_pipeline` as a background task and tracks jobs in an in-memory dict (`_jobs` + `_jobs_lock`). Pipeline run artifacts on disk are the durable record; the `_jobs` dict is only for in-flight status.

## Configuration

All tunables go through `config.py`, which reads env vars (with `.env` loaded via `python-dotenv`). Key categories:

- **LLM backend** — `LLM_BACKEND` (openai/ollama/litellm), `OPENAI_MODEL`, plus `AZURE_OPENAI_*` for Azure and `OLLAMA_*` for local.
- **Embeddings** — `EMBEDDING_MODEL` (CPU, MiniLM), `NOMIC_MODEL`/`NOMIC_DEVICE`/`NOMIC_DIM` (GPU). `CROSS_ENCODER_MODEL` for reranking.
- **Clustering** — `MACRO_MIN_CLUSTER_SIZE`, `MICRO_MIN_CLUSTER_SIZE`, `UMAP_N_NEIGHBORS`, `UMAP_N_COMPONENTS`, `UMAP_MIN_DIST`.
- **Optimization** — `USE_RAY`, `OPTIMIZE_TRIALS`, `OPTIMIZE_PARALLEL`.
- **Chunking** — `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `SUMMARY_MAX_CHARS`.
- **Thresholds** — `MERGE_SIMILARITY_THRESHOLD`, `HIGH_CONFIDENCE_THRESHOLD`, `TENTATIVE_THRESHOLD`, `NOVEL_BUFFER_SIZE`.
- **RLM (dspy Recursive Language Model)** — `RLM_SUB_MODEL`, `RLM_MAX_ITERATIONS`, `RLM_MAX_LLM_CALLS` — used by optional field discovery path that iterates over snippets.
- **Storage** — `DB_PATH`, `LANCEDB_PATH`, `SNAPSHOT_CONTAINER`/`SNAPSHOT_BLOB_PREFIX` (for Azure Blob snapshots via `blob_snapshot.py` when `AZURE_STORAGE_CONNECTION_STRING` is set).

`LEGAL_STOPWORDS` in `config.py` is the curated legal-domain stopword list applied alongside sklearn's English stopwords in BERTopic's CountVectorizer.

## Conventions to preserve

- **Legal stopwords + English stopwords** are combined for BERTopic analyzers. The custom `StopwordFilteringAnalyzer` in `cluster_pipeline.py` filters n-grams where **all** tokens are stopwords (so "termination clause" survives but "pursuant to" does not). It implements `__getstate__`/`__setstate__` to stay picklable for BERTopic safetensors persistence — preserve that when editing.
- **Idempotent schema init** — `ClusteringStore._init_schema()` runs on every construction. New tables/columns should use `CREATE TABLE IF NOT EXISTS` / `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.
- **Graceful GPU fallback** — any new GPU code path should check `cuml_available()` and degrade to CPU rather than hard-failing.
- **Artifact layout** — `cluster_pipeline.py` writes `data/artifacts/<account_id>/<run_id>/{macro_model/, micro_models/<domain_id>/, embeddings.npz, run_manifest.json}`. Downstream consumers read from this layout; keep it stable.
- **No pytest suite** — `scripts/validate_pipeline.py` and `scripts/test_gpu_stack.py` are the test entry points; add new checks there rather than introducing a separate test framework.
- **`data/` is gitignored** (see `.gitignore`). Don't assume any DuckDB or artifact files exist in a fresh clone.
