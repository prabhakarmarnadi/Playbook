# Clustering V2 — Local-First Agreement Intelligence Pipeline

Replaces the legacy multi-service clustering stack (AIDB gRPC + Azure Blob + Redis + ServiceBus) with a single local-first pipeline. Zero cloud dependencies for compute. All state in DuckDB + LanceDB.

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  PDF / TXT   │────▶│  Chunker     │────▶│  Embedder        │────▶│  Macro Clusterer  │
│  Documents   │     │  (sliding    │     │  (Nomic GPU or   │     │  (doc summaries → │
│              │     │   window)    │     │   MiniLM CPU)    │     │   BERTopic)       │
└─────────────┘     └──────────────┘     └──────────────────┘     └────────┬─────────┘
                                                                           │
                                                                           ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  DuckDB      │◀────│  Field       │◀────│  Micro Clusterer │◀────│  Per-Domain       │
│  (metadata)  │     │  Discovery   │     │  (clause-level   │     │  Chunk Routing    │
│              │     │  + Extraction│     │   BERTopic)      │     │                  │
│  LanceDB     │     │  (LLM)      │     │  + Topic Merger  │     │                  │
│  (vectors)   │     └──────────────┘     └──────────────────┘     └──────────────────┘
└──────────────┘
```

### Data Flow

```
PDFs ──parse──▶ raw text ──chunk──▶ ~256-token clauses
                                        │
                                        ├──embed──▶ 768-dim vectors (Nomic GPU)
                                        │                │
                                        │                ├──▶ LanceDB (persistent ANN index)
                                        │                └──▶ DuckDB chunks table (BLOB)
                                        │
                    doc summaries ──embed──▶ macro BERTopic ──▶ domains (NDA, SaaS, ...)
                                                                    │
                                        clauses per domain ────▶ micro BERTopic ──▶ clause types
                                                                    │
                                                                    ├──▶ DuckDB clusters table
                                                                    ├──▶ DuckDB cluster_assignments
                                                                    └──▶ LanceDB (domain/cluster annotations)
                                                                    │
                                                        (optional)  ├──▶ LLM field discovery
                                                                    └──▶ LLM field extraction
```

### Storage Split

| Store | Role | Tables / Data |
|-------|------|---------------|
| **DuckDB** | System of record — metadata, relationships, pipeline state | `agreements`, `domains`, `clusters`, `chunks`, `cluster_assignments`, `field_definitions`, `extractions`, `corrections`, `pipeline_runs`, `cluster_centroids`, `novel_documents`, `composite_definitions` |
| **LanceDB** | Vector index — persistent ANN with metadata filtering | `chunks` (chunk vectors + domain/cluster ID), `documents` (summary vectors) |

## Three Pipelines

### 1. CPU Pipeline (`core/pipeline.py`)
The original hackathon MVP. MiniLM-L6-v2 embeddings (384-dim, CPU), sklearn UMAP/HDBSCAN, FAISS retrieval, LLM labeling. Full 7-stage flow including field discovery and extraction.

```bash
poetry run python -c "from core.pipeline import run_full_pipeline; run_full_pipeline()"
```

### 2. GPU Pipeline (`core/gpu_pipeline.py`)
GPU-accelerated upgrade. Nomic embeddings (768-dim, CUDA), cuML UMAP/HDBSCAN, LanceDB vectors, optional Ray-distributed Bayesian hyperparameter optimization (50 parallel trials).

```bash
# Requires: CUDA GPU + optionally conda install cuml
poetry run python -c "from core.gpu_pipeline import run_gpu_pipeline; run_gpu_pipeline()"
```

### 3. PDF Pipeline (`core/pdf_pipeline.py`)
Focused ingest-to-cluster pipeline. PDF parsing (PyMuPDF) → chunk → embed (Nomic GPU) → macro/micro clustering → persist to DuckDB + LanceDB. No LLM calls. Auto-labels clusters from keywords.

```bash
poetry run python -m core.pdf_pipeline /path/to/pdfs
poetry run python -m core.pdf_pipeline /path/to/pdfs --db data/my.duckdb --lance data/my.lance
```

## Directory Structure

```
clustering-v2/
├── README.md                          # This file
├── hackathon_plan.md                  # Original hackathon design doc
├── cluster_v2_engineering.md          # Engineering spec
├── cluster_v2_product.md              # Product spec
│
├── legacy_stack/                      # Legacy V1 code (reference only)
│   ├── apr-agreement-clustering/      # BERTopic + Optuna optimizer
│   └── apr-fields-processor/          # Chunking + embedding via AIDB gRPC
│
└── hackathon/                         # V2 implementation
    ├── pyproject.toml                 # Poetry deps (ML, GPU, LLM, storage, UI)
    ├── poetry.lock
    ├── environment-gpu.yml            # Conda env for RAPIDS cuML (GPU UMAP/HDBSCAN)
    ├── setup.sh                       # One-command setup script
    ├── config.py                      # All configuration (paths, models, thresholds)
    ├── app.py                         # Streamlit app entrypoint
    │
    ├── core/                          # Pipeline modules
    │   ├── pdf_parser.py              # PyMuPDF PDF → text + metadata + manifest
    │   ├── pdf_pipeline.py            # PDF-only ingest → cluster → persist pipeline
    │   ├── pipeline.py                # Full CPU pipeline (7 stages + LLM)
    │   ├── gpu_pipeline.py            # Full GPU pipeline (cuML + Ray + LanceDB)
    │   │
    │   ├── chunker.py                 # Sliding-window token chunker
    │   ├── embedder.py                # MiniLM-L6-v2 CPU embedder (384-dim)
    │   ├── nomic_embedder.py          # Nomic-embed-text-v1.5 GPU embedder (768-dim)
    │   ├── clause_embedder.py         # Chunk-level embedding with context (FIELD-110)
    │   │
    │   ├── macro_clusterer.py         # Doc-summary BERTopic → macro domains
    │   ├── micro_clusterer.py         # Per-domain clause BERTopic → clause types
    │   ├── topic_merger.py            # Hierarchical topic dedup (FIELD-310)
    │   ├── cluster_assigner.py        # Incremental cluster assignment (FIELD-801)
    │   │
    │   ├── cuml_wrappers.py           # GPU UMAP/HDBSCAN via RAPIDS (CPU fallback)
    │   ├── scoring.py                 # Z-score + 4 penalty terms (ported from legacy)
    │   ├── ray_optimizer.py           # Ray @ray.remote Bayesian optimization (50 trials)
    │   │
    │   ├── store.py                   # DuckDB storage layer (12 tables)
    │   ├── lancedb_store.py           # LanceDB persistent vector store
    │   │
    │   ├── retrieval.py               # Hybrid BM25 + Dense + CrossEncoder + RRF
    │   ├── field_discovery.py         # LLM-based field schema discovery
    │   ├── extractor.py               # LLM-based field value extraction
    │   ├── composite.py               # Composite/computed field engine
    │   └── llm_client.py              # OpenAI / Ollama / LiteLLM client
    │
    ├── ui/                            # Streamlit pages
    │   ├── dashboard_page.py          # Pipeline overview + stats
    │   ├── upload_page.py             # Document upload (PDF/TXT)
    │   ├── domain_page.py             # Macro domain explorer
    │   ├── cluster_page.py            # Clause cluster deep-dive
    │   ├── ontology_page.py           # Domain → cluster tree view
    │   ├── search_page.py             # Hybrid retrieval demo
    │   ├── field_crud_page.py         # Field definition management
    │   ├── extraction_page.py         # Extraction results browser
    │   ├── example_set_page.py        # Example set curation
    │   └── composite_page.py          # Composite field builder
    │
    ├── scripts/
    │   ├── validate_pipeline.py       # 8 infrastructure tests (all passing)
    │   ├── test_gpu_stack.py          # 7 GPU component tests (CUDA, cuML, Nomic, LanceDB, Ray)
    │   └── prep_demo_data.py          # Generate 30 synthetic agreements
    │
    ├── prompts/                       # LLM prompt templates
    │   ├── macro_cluster_label.txt
    │   ├── cluster_label.txt
    │   ├── field_discovery.txt
    │   ├── field_extraction_conditioned.txt
    │   └── field_extraction_generic.txt
    │
    └── data/
        ├── sample_agreements/         # 30 synthetic agreements (5 types × 5-10 each)
        ├── parsed_texts/              # Output from PDF parser (.txt + manifest.json)
        └── cluster_sampler/           # Real PDF samples for testing
```

## Technology Stack

| Component | CPU Pipeline | GPU Pipeline |
|-----------|-------------|-------------|
| **Embeddings** | all-MiniLM-L6-v2 (384-dim) | nomic-embed-text-v1.5 (768-dim, CUDA) |
| **UMAP** | umap-learn (sklearn) | cuML UMAP (RAPIDS GPU) |
| **HDBSCAN** | hdbscan (CPU) | cuML HDBSCAN (RAPIDS GPU) |
| **Topic Modeling** | BERTopic | BERTopic (with cuML backends) |
| **Vector Store** | FAISS IndexFlatIP (in-memory) | LanceDB (persistent, filterable) |
| **Metadata Store** | DuckDB | DuckDB |
| **Optimization** | — | Ray @ray.remote + Optuna TPE (50 trials) |
| **Retrieval** | BM25 + Dense + CrossEncoder + RRF | Same + LanceDB ANN |
| **PDF Parsing** | PyMuPDF | PyMuPDF |
| **LLM** | OpenAI / Ollama / LiteLLM | Same |
| **UI** | Streamlit (10 pages) | Same |

## Quick Start

### CPU-only (no GPU required)

```bash
cd hackathon
./setup.sh                          # Install Poetry + deps + generate demo data
poetry run streamlit run app.py     # Launch UI at localhost:8501
```

### GPU-accelerated

```bash
# 1. Create conda env with RAPIDS
conda env create -f environment-gpu.yml
conda activate clustering-v2-gpu

# 2. Install remaining deps via Poetry
cd hackathon
poetry install

# 3. Run GPU diagnostic
poetry run python scripts/test_gpu_stack.py

# 4. Run PDF pipeline on your documents
poetry run python -m core.pdf_pipeline /path/to/pdfs
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `openai` | LLM provider: `openai`, `ollama`, `litellm` |
| `OPENAI_API_KEY` | — | Required for OpenAI backend |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model for field discovery/extraction |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | CPU embedder model |
| `NOMIC_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | GPU embedder model |
| `NOMIC_DEVICE` | `cuda` | Torch device for Nomic embedder |
| `NOMIC_DIM` | `768` | Embedding dimension (256/512 for Matryoshka) |
| `USE_RAY` | `true` | Enable Ray parallel optimization |
| `OPTIMIZE_TRIALS` | `50` | Bayesian optimization trial count |
| `OPTIMIZE_PARALLEL` | `4` | Max concurrent Ray tasks |

## Validation

```bash
# Infrastructure tests (8/8)
poetry run python scripts/validate_pipeline.py

# GPU stack tests (7/7)
poetry run python scripts/test_gpu_stack.py
```

Test coverage:
- Chunker, Embedder, Clause Embedder (FIELD-110)
- Composite Engine, Topic Merger (FIELD-310), Incremental Assigner (FIELD-801)
- DuckDB Store (full CRUD), Hybrid Retrieval (BM25 + Dense + RRF)
- CUDA detection, cuML wrappers, Nomic GPU embedder
- LanceDB vector store, Ray cluster, Scoring engine, Mini E2E pipeline

## Legacy vs V2 Comparison

| Aspect | Legacy V1 | V2 |
|--------|-----------|-----|
| **Services** | 4+ (AIDB, AIFlow, fields_processor, clustering_executor) | 1 script |
| **Network hops** | 8+ per pipeline run | 0 (compute is local) |
| **Embedding storage** | AIDB gRPC (remote) | DuckDB (local) + LanceDB (local) |
| **Job tracking** | Redis + Azure Blob markers | DuckDB `pipeline_runs` table |
| **Clustering** | Single-level BERTopic (global) | Two-stage: macro domains → micro clause types |
| **Optimization** | Optuna TPE, 30 trials, sequential | Ray tasks + Optuna TPE, 50 trials, parallel |
| **Retrieval** | HyDE → single ANN via AIDB | BM25 + Dense + CrossEncoder + RRF fusion |
| **GPU support** | None | cuML UMAP/HDBSCAN, Nomic GPU embeddings |
