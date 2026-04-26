# Contract Ontology Discovery Pipeline

Automated system that ingests legal PDF agreements, discovers clause-type taxonomy via unsupervised clustering, extracts structured fields per clause type, and produces a navigable contract knowledge graph — all running locally on GPU.

```
PDFs → Parse → Clause-Aware Chunk → Embed (Nomic 768d, CUDA)
     → EVoC Cluster (Optuna-tuned) → Domain Discovery → Field Discovery
     → Confidence-Gated RAG Extraction → Knowledge Graph → Dashboard
```

---

## Table of Contents

- [Onboarding: Run on Your Own Data](#onboarding-run-on-your-own-data)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Running the Pipeline](#running-the-pipeline)
  - [Monitoring a Run](#monitoring-a-run)
  - [Viewing Results](#viewing-results)
  - [Troubleshooting](#troubleshooting)
- [Architecture Overview](#architecture-overview)
- [Pipeline Stages](#pipeline-stages)
- [Pipeline Modes](#pipeline-modes)
- [Post-Pipeline Refinement](#post-pipeline-refinement)
- [Knowledge Graph](#knowledge-graph)
- [Agreement Network](#agreement-network)
- [Interactive Visualizations](#interactive-visualizations)
- [Streamlit UI](#streamlit-ui)
- [API Server](#api-server)
- [Storage Architecture](#storage-architecture)
- [Configuration](#configuration)
- [CLI Reference](#cli-reference)
- [Evaluation Framework](#evaluation-framework)
- [Project Structure](#project-structure)
  - [Dependency Graph](#dependency-graph-simplified)
- [Production Results](#production-results)
- [Requirements](#requirements)

---

## Onboarding: Run on Your Own Data

This section walks you through running the full pipeline end-to-end on any set of PDF contracts.

### Prerequisites

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| **Python** | 3.10+ | 3.10.x |
| **GPU** | NVIDIA with 16 GB VRAM | A100 80 GB |
| **CUDA** | 11.8+ | 12.x |
| **RAM** | 32 GB | 64 GB+ |
| **Disk** | 20 GB free | 50 GB+ (for lance vector store) |
| **Azure OpenAI** | Access to a GPT-4o-class deployment | GPT-4o + GPT-4o-mini (two deployments) |

> **No GPU?** The pipeline falls back to CPU embeddings (`all-MiniLM-L6-v2`, 384-dim) automatically but will be significantly slower for >100 docs.

### Installation

```bash
cd hackathon

# Option A: Poetry (recommended)
./setup.sh                        # installs Poetry deps + creates data dirs

# Option B: Manual
pip install -e .                  # or: poetry install --no-interaction
mkdir -p data/parsed_texts data/artifacts
```

### Environment Setup

Create a `.env` file (or export the variables directly):

```bash
# ── Required: Azure OpenAI ─────────────────────────────────────
export AZURE_OPENAI_ENDPOINT="https://YOUR-RESOURCE.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"          # primary model (field discovery, extraction)

# ── Optional: Second model for cost-efficient routing ───────────
export AZURE_OPENAI_DEPLOYMENT_MINI="gpt-4o-mini" # used for simple fields, labeling
# If not set, all calls use the primary deployment.

# ── Optional: Embedding overrides ───────────────────────────────
# export NOMIC_MODEL="nomic-ai/nomic-embed-text-v1.5"  # default GPU embedder
# export EMBEDDING_MODEL="all-MiniLM-L6-v2"            # default CPU fallback

# ── Optional: Tuning ───────────────────────────────────────────
# export CLUSTER_CONCURRENCY=15    # parallel clusters in Stage 4 (default 3)
# export RLM_MAX_ITERATIONS=25     # max agentic REPL iterations
# export RLM_MAX_LLM_CALLS=50     # max LLM calls per cluster discovery
```

Load the env before running:

```bash
source .venv/bin/activate        # if using Poetry venv
export $(grep -v '^#' .env | xargs)
```

### Running the Pipeline

#### Option 1: CLI (simplest)

```bash
# Full pipeline on a folder of PDFs
python -m core.minimal_pipeline run /path/to/your/pdfs \
    --account-id my_project \
    --device cuda \
    --optuna-trials 30 \
    --discovery rlm_v3_unified

# Limit to first 50 docs for a quick test
python -m core.minimal_pipeline run /path/to/your/pdfs \
    --account-id my_test \
    --max-docs 50
```

This creates:
- `data/my_project.duckdb` — all structured results
- `data/my_project.lance/` — vector embeddings
- `data/artifacts/my_project/` — JSON exports

#### Option 2: Python script (more control)

Create a script like `scripts/run_my_data.py`:

```python
import asyncio, os, sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.minimal_pipeline import MinimalPipelineConfig, run_minimal_pipeline

async def main():
    config = MinimalPipelineConfig(
        device="cuda",
        max_docs=None,                      # None = all docs in the folder
        use_optuna=True,
        optuna_trials=30,
        discovery_mode="rlm_v3_unified",    # agentic discovery + inline extraction
        max_fields_per_cluster=10,          # fields per clause type (10 is sweet spot)
        max_rag_retry_rounds=2,             # agentic re-query rounds
        extract_values=True,
        enable_confidence_gate=True,        # skip low-quality clusters
        enable_soft_membership=True,        # probabilistic cluster assignment
        enable_feedback=True,
        cluster_merge_threshold=0.88,       # merge similar super-clusters
        # resume_from_stage=4,              # uncomment to reuse cached clustering
    )

    os.environ["CLUSTER_CONCURRENCY"] = "15"

    result = await run_minimal_pipeline(
        pdf_dir="/path/to/your/pdfs",
        account_id="my_project",
        config=config,
    )
    print(f"Done: {result.get('n_clusters', 0)} clusters, "
          f"{result.get('n_fields', 0)} fields, "
          f"{result.get('n_extractions', 0)} extractions")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(name)-25s | %(message)s")
    asyncio.run(main())
```

Run it:

```bash
python scripts/run_my_data.py 2>&1 | tee data/my_run.log
# Or background:
nohup python scripts/run_my_data.py > data/my_run.log 2>&1 &
```

#### Option 3: Incremental — add new docs to existing clusters

```bash
python -m core.minimal_pipeline assign new_contract_1.pdf new_contract_2.pdf \
    --account-id my_project \
    --device cuda
```

### Monitoring a Run

```bash
# Tail the log
tail -f data/my_run.log

# Count clusters processed so far (Stage 4)
grep -c "fields (" data/my_run.log

# Count domain discovery progress (Stage 3.5)
grep -c "domain_discovery.*Domain '" data/my_run.log

# Check process is alive
ps -p $(pgrep -f run_my_data) -o pid,etime,rss,%cpu

# Quick DB stats (only while pipeline is NOT running — DuckDB is single-writer)
python3 -c "
import duckdb
db = duckdb.connect('data/my_project.duckdb', read_only=True)
for t in ['agreements','chunks','clusters','field_definitions','extractions']:
    n = db.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
    print(f'  {t}: {n:,}')
db.close()
"
```

#### Runtime Estimates

| Corpus Size | Stage 1-3 (Parse+Embed+Cluster) | Stage 3.5 (Domain Discovery) | Stage 4 (Field+Extract) | Total |
|:-----------:|:------:|:------:|:------:|:------:|
| 50 docs | ~5 min | ~5 min | ~15 min | ~25 min |
| 200 docs | ~15 min | ~15 min | ~60 min | ~90 min |
| 500 docs | ~30 min | ~30 min | ~3 hrs | ~4 hrs |
| 2,800 docs | ~45 min | ~30 min | ~8 hrs | ~9 hrs |

Stage 4 dominates. Set `CLUSTER_CONCURRENCY=15` to parallelise. LLM API rate limits are the main bottleneck.

### Viewing Results

After the pipeline completes:

```bash
# Generate an interactive HTML dashboard
python scripts/generate_demo_dashboard.py

# Serve it
python3 -m http.server 8899 --directory data/
# Open: http://localhost:8899/demo_dashboard.html

# Or query the DB directly
python3 -c "
import duckdb
db = duckdb.connect('data/my_project.duckdb', read_only=True)

# Top fields by extraction count
print(db.execute('''
    SELECT fd.name, fd.field_type, COUNT(e.extraction_id) AS n
    FROM field_definitions fd
    JOIN extractions e ON e.field_id = fd.field_id
    GROUP BY fd.name, fd.field_type
    ORDER BY n DESC LIMIT 20
''').fetchdf().to_string())

db.close()
"
```

#### Build the Knowledge Graph (post-pipeline)

```bash
python -c "
from core.context_graph import ContractContextGraph
from core.store import ContractStore

store = ContractStore('data/my_project.duckdb', 'data/my_project.lance')
kg = ContractContextGraph(store)
kg.build()
print(kg.summary())
"
```

#### Run the Evaluation Framework

```bash
# Quick unsupervised eval
python -m eval.runner data/my_project.duckdb

# Full eval with LLM judge
python -m eval.runner data/my_project.duckdb --judge --judge-sample 50
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `CLUSTER_CONCURRENCY` to 5, or set `--device cpu` |
| `content_filter` warnings in log | Azure content filter triggered on some contract text (e.g. indemnity clauses). The pipeline retries and skips — no action needed. |
| DuckDB `Conflicting lock` | Another process has the DB open for writing. Wait for the pipeline to finish, or use `read_only=True` for queries. |
| `ModuleNotFoundError: core.rlm_v3` | Run from the `hackathon/` directory, or add it to `PYTHONPATH`. |
| Slow on CPU | Expected. GPU embedding is ~50× faster. Use `--max-docs 50` for a test run. |
| 0 fields discovered | Check Azure OpenAI credentials. The discovery stage requires LLM access. |
| Low extraction count | Check `CLUSTER_CONCURRENCY` and that `--no-extraction` was not passed. |

### Key Configuration Knobs

| Parameter | Default | Effect |
|-----------|:-------:|--------|
| `max_fields_per_cluster` | 10 | Fields discovered per clause type. Higher = more coverage but more phantom fields. |
| `cluster_merge_threshold` | 0.88 | Cosine sim to merge similar clusters. Lower = fewer, broader clusters. |
| `max_rag_retry_rounds` | 2 | Agentic re-query attempts per field×doc. Higher = better recall, more LLM cost. |
| `confidence_gate_threshold` | 0.65 | Min avg confidence to proceed with full extraction. Lower = more clusters extracted. |
| `CLUSTER_CONCURRENCY` | 3 | Parallel cluster processing. Set to 10-15 for faster runs (needs more VRAM). |
| `min_cluster_chunks` | 20 | Skip clusters with fewer chunks (likely noise). |
| `optuna_trials` | 30 | Clustering optimization trials. More = better clusters, slower Stage 3. |

---

## Quick Start (Legacy Pipelines)

```bash
# EVoC hybrid mode (pre-minimal-pipeline; field_discovery.py + extractor.py)
python -m core.evoc_pipeline /path/to/pdfs --mode hybrid --optuna-trials 30

# Launch the dashboard
python3 -m http.server 8765    # serves dashboard.html + ontology_explorer.html

# Streamlit UI (10 pages)
streamlit run app.py

# REST API
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

---

## Architecture Overview

The pipeline is a 7-stage process that transforms raw PDF contracts into a structured ontology with extracted field values. Two additional post-pipeline modules refine quality and enable graph-based analysis.

```
┌──────────────────────────────────────────────────────────────────┐
│                        PIPELINE CORE                             │
│                                                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐    │
│  │  Parse   │→ │  Chunk  │→ │  Embed  │→ │  EVoC Cluster   │    │
│  │ PyMuPDF  │  │ Semantic│  │  Nomic  │  │  + Optuna Tune  │    │
│  │          │  │ + Legal │  │ 768-dim │  │  + Layer Select  │    │
│  └─────────┘  └─────────┘  └─────────┘  └────────┬────────┘    │
│                                                    │             │
│              ┌─────────────────────────────────────┘             │
│              ▼                                                   │
│  ┌─────────────┐  ┌────────────────┐  ┌───────────────────┐    │
│  │  LLM Label  │→ │ Field Discover │→ │ Field Extraction  │    │
│  │ GPT-4o-mini │  │  Per Cluster   │  │ Cluster-Condtnd   │    │
│  └─────────────┘  └────────────────┘  └───────────────────┘    │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                     POST-PIPELINE                                │
│                                                                  │
│  ┌──────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │ Ontology Refiner │  │ Knowledge Graph │  │ Intent Extract │ │
│  │  Outlier Assign  │  │  4 Node Types   │  │ Dynamic Disco  │ │
│  │  Field Dedup     │  │  7 Edge Types   │  │ Intent Catalog │ │
│  │  Label Synthesis │  │  Recommendations│  │                │ │
│  └──────────────────┘  └─────────────────┘  └────────────────┘ │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                     VISUALIZATION                                │
│                                                                  │
│  ┌──────────────────┐  ┌─────────────────────────────────────┐ │
│  │ Ontology Explorer│  │ Contract Ontology Workspace         │ │
│  │ D3.js Force Graph│  │ 3-Level Dashboard (Domain→Type→Doc) │ │
│  │ 4 Views (Cluster │  │ Health cards, field schema,         │ │
│  │  Agreement,Intent│  │ extraction grid, document detail    │ │
│  │  Matrix)         │  │                                     │ │
│  └──────────────────┘  └─────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

### Stage 1: Parse (`core/pdf_parser.py`)

PyMuPDF extracts text from each PDF. Results are cached to `data/parsed_texts/` with a manifest — subsequent runs skip parsing if the source directory is unchanged.

### Stage 2: Clause-Aware Chunking (`core/chunker.py`)

Documents are split into **clause segments** using structural signals (numbered headings, legal section patterns), then oversized clauses are sub-split via semantic breakpoint detection (Greg Kamradt method with legal header boosting). Each chunk is linked to its parent clause.

**Output:** `chunks` table (chunk_id, text, clause_id) + `clauses` table (clause_id, clause_title, clause_number, heading_level)

- Default chunk size: 256 tokens with 64-token overlap
- `--fast-chunk` flag switches to simple sliding window

### Stage 3: Embed (`core/nomic_embedder.py`)

All chunks are embedded with **nomic-embed-text-v1.5** (768-dim, CUDA). Clause-level embeddings are computed as L2-normalized means of constituent chunk embeddings. Document-level summaries are generated via field-aware scoring (`core/field_registry.py`).

Embeddings are persisted to both DuckDB (as BLOBs) and LanceDB (for vector search).

- GPU: `nomic-embed-text-v1.5` — 768-dim (Matryoshka, supports 256/512/768)
- CPU fallback: `all-MiniLM-L6-v2` — 384-dim

### Stage 4: Cluster (`core/evoc_pipeline.py`)

**EVoC** (Embedding Vector Oriented Clustering) clusters clause embeddings across all documents in a single pass — no domain boundary constraints. An "Indemnification" clause in a lease and one in a service agreement are grouped together.

Key components:

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| EVoC multi-layer | `core/evoc_cluster.py` | Produces clustering at multiple granularities |
| Optuna optimization | 30 trials (configurable) | Tunes `base_min_cluster_size`, `noise_level`, `n_neighbors`, `min_samples` |
| Smart layer selection | Silhouette + cosine coherence + outlier ratio | Picks best EVoC layer (replaces persistence-score) |
| Merge threshold | 0.96 cosine similarity | Collapses near-duplicate clause types |
| Outlier assignment | KNN to nearest centroid | Recovers unclustered clauses |
| Scoring engine | `core/scoring.py` | Silhouette, cosine coherence, DBCV per cluster |

In **hybrid mode**, a second EVoC pass clusters document-level summary embeddings to produce **document domains** (e.g., "Lease Agreements", "Service Contracts") as a navigation/metadata layer — not a clustering constraint.

### Stage 5: LLM Label

Azure OpenAI labels each clause type with a descriptive name and description using representative excerpts from the cluster. Prompt template: `prompts/cluster_label.txt`.

### Stage 6: Field Discovery (`core/field_discovery.py`)

For each clause type, the LLM analyzes representative chunks to discover recurring fields (e.g., `effective_date`, `termination_notice_period`, `governing_law`). Each field gets a name, type, description, and examples. Typically produces 5–15 fields per clause type.

### Stage 7: Field Extraction (`core/extractor.py`)

**Cluster-conditioned extraction** — the LLM knows exactly which clause type it's processing and which fields to extract. This yields:
- 40% fewer tokens than generic extraction
- 0 hallucinated fields (vs. ~8% with generic prompts)
- Values stored with confidence scores and source chunk references

---

## Pipeline Modes

| Mode | Flag | Algorithm | Output | Best For |
|------|------|-----------|--------|----------|
| **Hybrid** | `--mode hybrid` | Clause EVoC + Domain EVoC | Clause types + domains as nav layer | Production (recommended) |
| Clause | `--mode clause` | Flat EVoC on all clause embeddings | Clause types only | Simple corpora |
| Macro-Micro | `--mode macro-micro` | Domain EVoC → per-domain clause EVoC | Domains → clause types | Exploratory analysis |

**Legacy pipelines** (HDBSCAN-based, pre-EVoC):
- `python -m core.cluster_pipeline` — HDBSCAN macro-micro
- `python -m core.clause_pipeline` — HDBSCAN clause-type

### Algorithm Comparison (5 runs, 96–98 docs)

| Metric | HDBSCAN Macro-Micro | HDBSCAN Clause | EVoC Clause | EVoC Hybrid |
|--------|:---:|:---:|:---:|:---:|
| Quality Score | 0.724 | 0.886 | 0.915 | **0.920** |
| Clause Types | 63 | 72 | 85 | 79 |
| Fields | 742 | 844 | 1,012 | 944 |
| Outlier Rate | 44.1% | 26.1% | 24.8% | 26.2% |
| Time | 24.3 min | 16.2 min | 22.1 min | 20.7 min |

EVoC Hybrid wins on quality (0.920) while maintaining reasonable runtime.

---

## Post-Pipeline Refinement

### Ontology Refiner (`core/ontology_refiner.py`)

Three-phase structural improvement applied after clustering:

| Phase | Method | Effect |
|-------|--------|--------|
| **Outlier Assignment** | KNN to nearest cluster centroid (cosine ≥ 0.35) | Coverage: 61.6% → 99.9% |
| **Field Deduplication** | Cross-cluster merge of duplicate field definitions | 1,642 → 1,523 fields (–7.2%) |
| **Label Synthesis** | LLM regenerates cluster names from refined membership | ~120/136 clusters relabeled |

### Intent Extractor (`core/intent_extractor.py`)

Dynamic intent discovery system that analyzes clauses within each cluster to identify legal intents (obligations, rights, conditions, restrictions). Builds a growing intent catalog that evolves as more documents are processed.

- 200-doc run: 17,024 intents across 136 clause types
- Intent types discovered dynamically (no fixed taxonomy)

---

## Knowledge Graph

### `core/knowledge_graph.py` — ContractKnowledgeGraph

Multi-relational graph built from the refined ontology:

**Node Types (4):**

| Type | Count (200-doc) | Attributes |
|------|:---:|---|
| Agreement | 198 | filename, domain, n_clauses |
| Clause Type | 136 | label, domain, field_count |
| Intent Type | ~17K | label, cluster, frequency |
| Party | variable | name, role |

**Edge Types (7):**

| Edge | Meaning |
|------|---------|
| `CONTAINS` | Agreement → Clause Type |
| `CT_CO_OCCURS` | Two clause types co-occur in same agreements |
| `HOSTS` | Domain → Clause Type |
| `CO_OCCURS` | Intent → Intent co-occurrence |
| `BEARS` | Agreement → Risk obligation |
| `BENEFITS_FROM` | Agreement → Right/benefit |
| `SIMILAR_INTENT` | Intent similarity linkage |

**Analytics (200-doc results):**
- 18,770 nodes, 76,596 edges
- 98 implication rules (if clause A → likely clause B)
- 18 strategies (recurring clause bundles)
- 17 archetypes (agreement structural families via Louvain community detection)
- Anomaly detection (unusual clause combinations)
- Recommendation engine (missing clauses)

---

## Agreement Network

Human-readable agreement-to-agreement similarity scoring built on top of the knowledge graph:

**Scoring methodology:**
1. **IDF-based importance** — clause types weighted by inverse document frequency
2. **Rarity categories** — Boilerplate (≥15%), Common (5–15%), Specialized (2–5%), Unique (<2%)
3. **Weighted Jaccard** — shared clause types weighted by rarity: Boilerplate ×0.2, Common ×0.5, Specialized ×1.0, Unique ×1.5
4. **Strength labels** — Strong / Moderate / Weak based on weighted connection score
5. **Positive/negative reasons** — human-readable explanation for every agreement pair

**200-doc results:**
- 1,913 total connections (Jaccard ≥ 0.2)
- 58 Strong, 694 Moderate, 1,161 Weak
- 2 isolated agreements
- Each connection includes: breakdown bars, shared types, +/− reason cards

Output: `data/agreement_network_viz.json`

---

## Interactive Visualizations

### Ontology Explorer (`ontology_explorer.html`)

D3.js force-directed graph with 4 views:

| View | Content |
|------|---------|
| **Clusters** | 136 clause-type nodes, 464 edges (centroid similarity, shared fields, shared intents). Domain-colored, filterable. |
| **Agreements** | 198 agreement nodes, edges colored by connection strength (green=Strong, orange=Moderate, blue=Weak). Isolation rings for disconnected docs. |
| **Intents** | Top 50 intent types ranked by frequency. |
| **Matrix** | Domain × cluster heatmap with clause counts. |

Features: zoom/pan, drag, search, domain filters, edge-type toggles, force/size sliders, detail panel with fields + intents + neighbors.

### Contract Ontology Workspace (`dashboard.html`)

3-level nested navigation for non-technical users:

| Level | View | Content |
|-------|------|---------|
| **L1: Dashboard** | Health cards + domain grid | 5 KPI cards (agreements, coverage, types, confidence, review) + domain cards with mini bar charts |
| **L2: Cluster View** | Split list + detail | Filterable clause-type list with rarity tags → field schema table, agreements table, extraction grid |
| **L3: Document Detail** | Split composition + fields | Clause composition bar + type cards → extracted field editor with confidence badges |

Navigation: breadcrumb (Dashboard › Domain › Document), clickable rows, back buttons.

### Serving the Visualizations

```bash
cd hackathon
python3 -m http.server 8765

# Open in browser:
#   http://localhost:8765/ontology_explorer.html
#   http://localhost:8765/dashboard.html
```

Both load JSON data from `data/`:
- `data/ontology_viz_data.json` — clusters, edges, intents, domains
- `data/agreement_network_viz.json` — agreement nodes, scored edges
- `data/dashboard_data.json` — domains, clusters, agreements, fields, extractions

---

## Streamlit UI

10-page interactive application:

```bash
streamlit run app.py
```

| Page | Purpose |
|------|---------|
| Upload & Process | Ingest PDFs, run pipeline, monitor progress |
| Domain Explorer | Browse document domains, view agreement breakdown |
| Cluster Explorer | Browse clause types, view representative chunks |
| Ontology Tree | Hierarchical domain → cluster → field tree |
| Example Set Review | Review and curate representative chunk sets |
| Extraction Comparison | Side-by-side: generic vs. cluster-conditioned extraction |
| Composite Fields | Define derived fields (e.g., `TCV = price × qty × term`) |
| Hybrid Search | BM25 + dense + cross-encoder search |
| Field CRUD | Create/edit/delete field definitions |
| Dashboard | Aggregate stats, cluster quality, extraction accuracy |

---

## API Server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/cluster` | POST | Start pipeline job `{pdf_dir, account_id, max_docs, device}` |
| `/jobs/{job_id}` | GET | Poll job status + progress |
| `/runs/{account_id}` | GET | List all runs for an account |
| `/runs/{account_id}/{run_id}` | GET | Full run manifest + stats |
| `/health` | GET | Health check |

---

## Storage Architecture

### DuckDB (per account)

13 tables tracking the full pipeline state:

| Table | Purpose |
|-------|---------|
| `agreements` | Ingested documents with domain assignment |
| `clauses` | Parsed clause segments (title, number, heading level, clause type) |
| `chunks` | Text chunks with embeddings (linked to clause + agreement) |
| `domains` | Document-level domain clusters |
| `clusters` | Clause-type clusters with centroids |
| `cluster_assignments` | Chunk → cluster mapping with confidence |
| `field_definitions` | Discovered fields per cluster |
| `extractions` | Extracted values with confidence + source |
| `composite_definitions` | Derived field formulas |
| `corrections` | User correction audit trail |
| `cluster_centroids` | Running centroids for incremental assignment |
| `novel_documents` | Buffered docs for re-clustering |
| `pipeline_runs` | Run status / stage tracking |

### LanceDB (per account)

Two vector tables for hybrid search:
- `chunks` — chunk embeddings (768-dim) with metadata (agreement_id, domain_id, cluster_id)
- `documents` — document summary embeddings

### Export Artifacts

JSON artifacts written to `data/artifacts/<account_id>/<run_id>/`:

| File | Content |
|------|---------|
| `clause_types.json` | type → clauses → fields hierarchy |
| `fields.json` | all field definitions with cluster context |
| `chunks.json` | all chunks with cluster assignments |
| `embeddings.npz` | chunk + summary embeddings (numpy) |
| `cluster_quality.json` | silhouette, cosine, DBCV scores per cluster |
| `run_manifest.json` | run metadata, timing, stats |

---

## Configuration

All settings have defaults in `config.py` and can be overridden via environment variables.

### Required (for LLM stages)

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name (default: `gpt-5.4-mini`) |

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `NOMIC_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | GPU embedding model |
| `NOMIC_DEVICE` | `cuda` | Embedding device |
| `NOMIC_DIM` | `768` | Embedding dimensionality (256/512/768) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | CPU fallback embedder |

### Clustering

| Variable | Default | Description |
|----------|---------|-------------|
| `OPTIMIZE_TRIALS` | `30` | Optuna trial count |
| `OPTIMIZE_PARALLEL` | `4` | Ray parallel workers |
| `MACRO_MIN_CLUSTER_SIZE` | `5` | Document-level cluster minimum |
| `MICRO_MIN_CLUSTER_SIZE` | `3` | Clause-level cluster minimum |
| `MERGE_SIMILARITY_THRESHOLD` | `0.82` | Merge clusters above this cosine sim |

### LLM Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `openai` | Provider: `openai` / `ollama` / `litellm` |
| `OPENAI_API_KEY` | — | For OpenAI backend |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | For Ollama backend |
| `OLLAMA_MODEL` | `llama3.1` | Ollama model |

### Chunking

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE_TOKENS` | `256` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `64` | Overlap between chunks |

---

## CLI Reference

```bash
# ── Main Pipeline (EVoC) ──────────────────────────────────────
python -m core.evoc_pipeline <pdf_dir> \
    --account-id <id> \
    --mode hybrid \                # clause | macro-micro | hybrid
    --optuna-trials 30 \           # Optuna trials (default 30)
    --no-optuna \                  # Skip optimization, use defaults
    --noise-level 0.3 \            # EVoC noise level (0 = cluster more)
    --max-docs 200 \               # Limit document count
    --device cuda \                # cuda | cpu
    --fast-chunk \                 # Sliding window (skip semantic chunking)
    --llm-summaries                # LLM scenario summaries for macro

# ── Legacy Pipelines (HDBSCAN) ────────────────────────────────
python -m core.cluster_pipeline <pdf_dir> --account-id <id>
python -m core.clause_pipeline <pdf_dir> --account-id <id>

# ── Utilities ─────────────────────────────────────────────────
python scripts/validate_pipeline.py --skip-llm
python scripts/inspect_db.py data/<account>.duckdb
python scripts/compare_runs.py --account-id <id>
python scripts/test_gpu_stack.py
```

---

## Evaluation Framework

Standalone evaluation pipeline that runs alongside clustering for any account. Measures component-wise quality (unit testing for ML), system-wide coverage (integration testing), and produces a single **Pipeline Health Score** as the gold metric.

### Pipeline Health Score

The single number that answers "is the output good enough to ship?":

```
★ Health Score = Quality × 0.4 + Coverage × 0.3 + Structure × 0.2 + Consistency × 0.1
```

| Component | Weight | What It Measures | 200-doc Score |
|-----------|--------|-----------------|:-----------:|
| **Quality** | 40% | Avg intra-cluster cosine coherence | 0.8939 |
| **Coverage** | 30% | Clauses typed (40%) + with intents (30%) + with extractions (30%) | 0.8693 |
| **Structure** | 20% | Outlier health (50%) + dominance health (50%) | 1.0000 |
| **Consistency** | 10% | KG neighbor cluster overlap | 0.7707 |
| **★ Health** | | | **0.8954** |

**Grades:** ≥0.85 Excellent · ≥0.70 Good · ≥0.50 Degraded · <0.50 Broken

> **Why not the legacy Z-score?** The legacy composite uses DBCV (density-based validity, -0.18) and silhouette (0.085), which penalize non-spherical clusters. EVoC produces *semantically coherent* clusters, not *density-connected* ones — so the legacy Z-score reports 0.22 for a pipeline that's actually excellent. The health score uses cosine coherence instead, which is what EVoC actually optimizes for.

### Metrics Tracked (48 total)

| Category | Count | Key Metrics |
|----------|:-----:|-------------|
| **Clustering** | 22 | `silhouette_score`, `davies_bouldin_index`, `dbcv_score`, `chunk_outlier_pct`, `clause_outlier_pct`, `avg_coherence`, `z_score`, `adjusted_score`, penalty breakdown |
| **KG Embeddings** | 2 | `neighborhood_domain_consistency`, `neighborhood_cluster_overlap` |
| **Fields** (vs gold) | 6 | `precision`, `recall`, `f1` |
| **Naming** (vs gold) | 5 | `avg_semantic_similarity`, `exact_match_pct` |
| **Intent Gold** | 4 | `avg_best_match_sim`, `recall_at_0.8` |
| **LLM Judge** | 7 | `avg_faithfulness`, `avg_actionability`, `healthy_insight_score`, `hallucination_rate` |
| **System** | 12 | Coverage cascade, ontology completeness, cross-component consistency |
| **Health** | 5 | `score`, `grade`, component breakdown |

### Usage

```bash
# Quick unsupervised eval on any account DB
python -m eval.runner data/my_account.duckdb

# Compare against saved baseline (detects regressions)
python -m eval.runner data/my_account.duckdb \
    --baseline eval/baseline_results.json

# Full eval: gold set + LLM intent judge + regression + save baseline
python -m eval.runner data/my_account.duckdb \
    --gold eval/gold_standard.json \
    --baseline eval/baseline_results.json \
    --judge --judge-sample 50 \
    --save-baseline eval/baseline_results.json \
    --output eval/report.md

# Bootstrap a gold set draft from pipeline output (human reviews & corrects)
python -m eval.bootstrap data/evoc_200_refined.duckdb --n-docs 50
```

### Regression Detection

The eval runner compares every metric against a saved baseline and flags:
- **✅** — within 5% relative tolerance
- **❌** — degraded beyond threshold or crossed a hard limit
- **🆕** — new metric, no baseline to compare

Hard limits that always trigger ❌ regardless of baseline:

| Metric | Threshold | Meaning |
|--------|-----------|----------|
| `chunk_outlier_pct` | > 20% | Too many unassigned chunks |
| `clause_outlier_pct` | > 50% | Too many untyped clauses |
| `silhouette_score` | < 0.05 | No cluster structure |
| `davies_bouldin_index` | > 5.0 | Clusters overlapping badly |
| `health.score` | < 0.50 | Pipeline broken |

### LLM-as-a-Judge (Intent Quality)

Opt-in via `--judge`. Samples N clauses and evaluates each discovered intent on:
- **Faithfulness** (1–5): Is the intent actually present in the source clause?
- **Actionability** (1–5): Would a legal ops team act on this insight?
- **Hallucination detection**: Flags intents with no textual support

```
Healthy Insight Score = (Avg Faithfulness + Avg Actionability) / 10
```

Ranges from 0.2 (both=1, garbage) to 1.0 (both=5, perfect).

### Gold Set Strategy

50 documents selected via stratified sampling (high/low quality clusters, multi-domain, edge cases). See [`eval/GOLD_SET_STRATEGY.md`](eval/GOLD_SET_STRATEGY.md) for detailed annotation instructions.

Bootstrap workflow:
1. `python -m eval.bootstrap data/evoc_200_refined.duckdb` → generates draft with pipeline predictions pre-filled
2. Human annotator reviews & corrects the draft
3. `python -m eval.runner data/... --gold eval/gold_standard.json --save-baseline eval/baseline_results.json`

---

## Project Structure

> **Reading guide:** Files are classified by tracing actual imports from the four entry points (`minimal_pipeline.py`, `run_full_458.py`, `ontology_evolution.py`, `generate_demo_dashboard.py`). Start with **CRITICAL**, understand **SUPPORTING**, ignore **LEGACY** unless you're doing archaeology.

```
hackathon/
├── config.py                       # Centralized configuration (env vars + defaults)
├── app.py                          # Streamlit UI entrypoint (10 pages)
├── api_server.py                   # FastAPI REST server
├── setup.sh                        # One-command install script
├── .env                            # Azure OpenAI + LLM credentials
│
├── core/
│   │
│   │  ── CRITICAL: Active pipeline (you must understand these) ──────
│   │
│   ├── minimal_pipeline.py         # ★ Main orchestrator. 7-stage pipeline + incremental assign.
│   │                               #   This is THE entry point. Everything else is called from here.
│   ├── rlm_v3.py                   # ★ Agentic field discovery (v3). REPL with 12 tools.
│   │                               #   Discovers fields per cluster via iterative LLM sessions.
│   ├── rag_extractor.py            # ★ RAG-based field extraction. 7-stage per (doc, field):
│   │                               #   Sample → Retrieve → Extract → Retry → Index → Review → Normalize.
│   │                               #   Also contains all 8 deterministic type normalizers.
│   ├── store.py                    # ★ DuckDB storage layer. 19 tables. All pipeline state lives here.
│   ├── lancedb_store.py            # ★ LanceDB vector store for hybrid search (chunks + docs).
│   ├── evoc_pipeline.py            # ★ EVoC clustering engine + Optuna tuning + layer selection.
│   │                               #   Called by minimal_pipeline for Stage 3 (clustering).
│   ├── cluster_assigner.py         # ★ Three-tier soft assignment (A/B/C) for new documents.
│   │                               #   Tier A: hard-assign + centroid update.
│   │                               #   Tier B: soft membership, no centroid update.
│   │                               #   Tier C: buffer for recluster.
│   ├── context_graph.py            # ★ Post-pipeline knowledge graph. Temporal facts, episodes,
│   │                               #   anomaly detection, recommendations, archetypes.
│   │
│   │  ── SUPPORTING: Imported transitively by critical modules ──────
│   │
│   ├── pdf_parser.py               # PyMuPDF: PDF → text + metadata. Stage 1.
│   ├── chunker.py                  # Semantic + sliding-window chunking. Stage 2.
│   ├── nomic_embedder.py           # GPU embedder: nomic-embed-text-v1.5 (768-dim). Stage 2.
│   ├── embedder.py                 # CPU fallback embedder: MiniLM-L6-v2 (384-dim).
│   ├── retrieval.py                # Hybrid BM25 + vector retrieval. Used by rag_extractor.
│   ├── latent_briefing.py          # Text compaction for RLM context windows. Used by rlm_v3.
│   ├── llm_client.py               # Azure OpenAI / LLM API wrapper.
│   ├── llm_summarizer.py           # LLM-powered cluster label/summary generation.
│   ├── scoring.py                  # Cluster quality metrics (silhouette, cosine, DBCV).
│   ├── topic_merger.py             # Merge near-duplicate clusters by cosine similarity.
│   ├── macro_clusterer.py          # Document-level domain clustering (Stage 3 domains).
│   ├── micro_clusterer.py          # Per-domain clause clustering. Used by evoc_pipeline.
│   ├── cuml_wrappers.py            # GPU UMAP/HDBSCAN wrappers (cuML, CPU fallback).
│   ├── field_features.py           # Predefined field definitions + feature extraction.
│   ├── field_registry.py           # Dynamic field catalog. Used by macro_clusterer.
│   ├── field_discovery.py          # Original LLM field discovery. Still reachable via evoc_pipeline.
│   ├── intent_extractor.py         # Clause intent classification. Dynamic intent catalog.
│   ├── feedback_store.py           # User feedback JSON persistence.
│   ├── personalization.py          # User preference engine for personalized extraction.
│   ├── ontology_refiner.py         # Post-pipeline outlier assign + dedup + relabel.
│   │                               #   ⚠ 5 schema mismatches with current store — use with caution.
│   ├── rlm_with_memory.py          # RLM with latent briefing memory. Transitive via field_discovery.
│   │
│   │  ── POST-PIPELINE: Analytics modules (not in main pipeline path) ──
│   │
│   ├── knowledge_graph.py          # Backward-compat shim → re-exports from context_graph.py.
│   ├── risk_scorer.py              # Contract risk scoring. Only via evoc_pipeline run_analytics().
│   ├── clause_deviation.py         # Clause deviation detection vs norms.
│   ├── field_normalizer.py         # Field value normalization (different scope from rag_extractor).
│   ├── playbook.py                 # Playbook template library + matching.
│   │
│   │  ── LEGACY: Not imported by any active entry point ─────────────
│   │
│   ├── pipeline.py                 # Original v1 pipeline. Replaced by minimal_pipeline.
│   ├── production_pipeline.py      # Superseded "production" variant. Nobody imports it.
│   ├── gpu_pipeline.py             # GPU pipeline variant. Replaced by evoc_pipeline's cuml path.
│   ├── scalable_pipeline.py        # Distributed pipeline with PgStore. Dead code.
│   ├── cluster_pipeline.py         # Intermediate pipeline version. Only used by legacy scripts.
│   ├── cluster_pipeline copy.py    # Literal file copy. Dead code.
│   ├── clause_pipeline.py          # Clause-specific pipeline variant. Unused.
│   ├── pdf_pipeline.py             # Minimal PDF-only pipeline. Unused.
│   ├── rlm_v2.py                   # RLM v2 (non-DSPy). Superseded by rlm_v3.
│   ├── rlm_v2_dspy.py              # RLM v2 with DSPy signatures. Superseded by rlm_v3.
│   ├── extractor.py                # Original extraction. Superseded by rag_extractor.
│   ├── grpc_lm.py                  # gRPC client for Triton/vLLM. Unused.
│   ├── pg_store.py                 # PostgreSQL backend. Only used by scalable_pipeline.
│   ├── clause_embedder.py          # Clause-aware embedding. Only used by legacy pipeline.py.
│   ├── composite.py                # Composite scoring/features. Only used by validate_pipeline.
│   ├── blob_snapshot.py            # Azure Blob snapshot. Unused.
│   ├── personalization_injector.py # Personalization middleware. Unused (personalization.py used).
│   ├── ray_optimizer.py            # Ray-based hyperparameter opt. Only used by legacy pipelines.
│   └── __init__.py                 # Empty package marker.
│
├── scripts/
│   │
│   │  ── CRITICAL: Active entry points ──────────────────────────────
│   │
│   ├── run_full_458.py             # ★ Production run: all 458 clusters through minimal_pipeline.
│   ├── run_mini_top30.py           # ★ Quick test: top-30 clusters through minimal_pipeline.
│   │
│   │  ── POST-PIPELINE: Analysis / visualization ────────────────────
│   │
│   ├── ontology_evolution.py       # Ontology health monitoring + drift detection + recluster.
│   │                               #   CLI: python scripts/ontology_evolution.py check|evolve <db>
│   ├── generate_demo_dashboard.py  # Self-contained HTML dashboard from DuckDB data.
│   ├── run_analytics.py            # Risk scoring, deviations, playbooks runner.
│   ├── generate_hierarchy_data.py  # Domain→cluster hierarchy JSON export.
│   ├── generate_viz_data.py        # Visualization data export from ClusteringStore.
│   ├── generate_comparison_data.py # Comparison dataset generator.
│   ├── recluster_with_layers.py    # Re-cluster with different layer configs.
│   │
│   │  ── UTILITY: Dev tools, inspectors, eval ───────────────────────
│   │
│   ├── run_full_pipeline.py        # Alternate full run wrapper (similar to run_full_458).
│   ├── run_public_contracts.py     # Public contracts runner (uses minimal_pipeline).
│   ├── run_public_contracts_perf.py # Performance-focused run w/ profiling.
│   ├── run_v3_vs_rag_eval.py       # Eval: rlm_v3 vs rag_extractor comparison.
│   ├── eval_evoc_clustering.py     # Clustering quality metrics evaluator.
│   ├── compare_runs.py             # Diff comparison between pipeline runs.
│   ├── llm_judge_eval.py           # LLM-as-judge evaluation.
│   ├── inspect_db.py               # DuckDB schema/content inspection.
│   ├── prep_demo_data.py           # Demo dataset preparation.
│   ├── build_presentation.py       # Presentation slide generation.
│   ├── rag_extraction.py           # Standalone RAG extraction runner.
│   ├── test_gpu_stack.py           # GPU stack smoke test (cuml, nomic, lance).
│   ├── validate_pipeline.py        # Pipeline component validation.
│   ├── resume_cuad.py              # Resume CUAD evaluation from checkpoint.
│   ├── extraction_value_comparison_v2_standalone.py  # Standalone extraction comparison.
│   ├── run_cuad_eval.py            # CUAD eval (uses evoc_pipeline directly).
│   │
│   │  ── LEGACY: Only use deprecated modules (rlm_v2, dspy, old pipelines) ──
│   │
│   ├── run_pipeline.py             # Old runner using legacy cluster_pipeline.
│   ├── run_combined_pipeline.py    # Combined pipeline using rlm_v2_dspy.
│   ├── rlm_full_pipeline.py        # Full pipeline using rlm_v2_dspy.
│   ├── agentic_extractions.py      # Agentic extraction experiments (rlm_v2_dspy).
│   ├── run_dspy_comparison.py      # DSPy vs non-DSPy comparison.
│   ├── run_unified_eval.py         # Unified eval harness (rlm_v2_dspy).
│   ├── compare_field_discovery.py  # Field discovery comparison (old eval).
│   ├── compare_rlm_vs_rlm_lb.py   # RLM vs RLM+latent_briefing comparison.
│   ├── extraction_value_comparison.py      # Extraction comparison v1.
│   ├── extraction_value_comparison_clause.py  # Clause-level extraction comparison.
│   └── extraction_value_comparison_v2.py   # Extraction comparison v2 (rlm_v2).
│
├── ui/                             # ── Streamlit Pages ──
│   ├── upload_page.py              # Upload & process agreements
│   ├── domain_page.py              # Domain explorer
│   ├── cluster_page.py             # Cluster explorer + UMAP viz
│   ├── ontology_page.py            # Domain → Type → Field tree
│   ├── example_set_page.py         # Representative doc review
│   ├── extraction_page.py          # V1 vs V2 extraction comparison
│   ├── composite_page.py           # Composite field formulas
│   ├── search_page.py              # Hybrid search interface
│   ├── field_crud_page.py          # Field management (CRUD)
│   └── dashboard_page.py           # Quality + coverage dashboard
│
├── prompts/                        # ── LLM Prompt Templates ──
│   ├── cluster_label.txt           # Clause type naming from keywords
│   ├── macro_cluster_label.txt     # Document domain naming
│   ├── field_discovery.txt         # Field discovery from representative chunks
│   ├── field_extraction_conditioned.txt   # Cluster-conditioned extraction
│   └── field_extraction_generic.txt       # Generic extraction (baseline)
│
├── data/
│   ├── parsed_texts/               # Cached PDF extractions
│   ├── artifacts/                  # Run outputs (per account_id/run_id)
│   ├── ontology_viz_data.json      # Cluster graph (136 nodes, 464 edges)
│   ├── agreement_network_viz.json  # Agreement network (198 nodes, 1,913 edges)
│   ├── dashboard_data.json         # Dashboard data (domains, clusters, extractions)
│   ├── *.duckdb                    # DuckDB databases (per account)
│   └── *.lance/                    # LanceDB vector stores (per account)
│
├── eval/                           # ── Evaluation Framework ──
│   ├── runner.py                   # Main CLI: python -m eval.runner <db>
│   ├── component_metrics.py        # Clustering, KG, field, naming evaluators
│   ├── system_metrics.py           # Coverage cascade, health score
│   ├── intent_judge.py             # LLM-as-a-Judge for intent quality
│   ├── regression.py               # Baseline comparison + regression detection
│   ├── bootstrap.py                # Auto-generate gold set draft from DB
│   ├── gold_standard.json          # Gold standard schema template
│   ├── baseline_results.json       # Saved baseline (48 metrics)
│   └── GOLD_SET_STRATEGY.md        # Annotation instructions + strategy
│
├── pyproject.toml                  # Poetry dependencies
├── ALGORITHM_COMPARISON.md         # HDBSCAN vs EVoC benchmark
├── PIPELINE_COMPARISON.md          # Macro-Micro vs Clause vs Hybrid benchmark
└── PERFORMANCE_DELTA_REPORT.md     # 96-doc → 200-doc scaling report
```

### Dependency Graph (Simplified)

```
               ┌───────────────────────────────────────────────────────┐
               │           scripts/run_full_458.py                     │
               │           scripts/run_mini_top30.py                   │
               └───────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │    core/minimal_pipeline.py   │  ← THE entry point
                    │    (Stages 1-7 + assign)      │
                    └──┬──────┬──────┬──────┬──────┘
                       │      │      │      │
          ┌────────────┘      │      │      └────────────┐
          ▼                   ▼      ▼                   ▼
  ┌───────────────┐  ┌────────────┐ ┌────────────┐  ┌────────────────┐
  │ evoc_pipeline │  │  rlm_v3    │ │rag_extractor│  │cluster_assigner│
  │ (Stage 3)     │  │(Stage 3.5) │ │ (Stage 4)  │  │ (Stage 5)      │
  └──┬──┬──┬──┬──┘  └─────┬──────┘ └──────┬─────┘  └────────────────┘
     │  │  │  │            │               │
     ▼  ▼  ▼  ▼            ▼               ▼
  nomic_embedder      latent_briefing   retrieval
  chunker             llm_client        lancedb_store
  pdf_parser                            store (DuckDB)
  scoring
  topic_merger
  macro_clusterer
  micro_clusterer
  cuml_wrappers
```

---

## Production Results

### 200-Document Run (EVoC Hybrid, 30 Optuna Trials)

| Metric | Raw | After Refinement |
|--------|:---:|:---:|
| Agreements | 198 | 198 |
| Total Clauses | 7,360 | 7,360 |
| Clustered Clauses | 4,534 | 7,351 |
| **Coverage** | **61.6%** | **99.9%** |
| Clause Types | 136 | 136 |
| Document Domains | 7 | 7 |
| Field Definitions | 1,642 | 1,523 |
| Total Extractions | 23,225 | 23,225 |
| Avg Confidence | 0.720 | 0.720 |
| High Confidence (≥0.8) | 47.3% | 47.3% |
| Intent Types | 17,024 | 17,024 |
| KG Nodes | 18,770 | 18,770 |
| KG Edges | 76,596 | 76,596 |
| Agreement Connections | — | 1,913 |
| **★ Health Score** | **0.8495 (Good)** | **0.8954 (Excellent)** |
| Pipeline Time | 77m 17s | +~5m refinement |

### Cost Analysis (200-doc run)

| Stage | LLM Calls | Est. Cost |
|-------|:---------:|:---------:|
| Intent Extraction | ~1,820 | $3.61 |
| Cluster Labeling | ~136 | $0.27 |
| Field Discovery | ~136 | $0.41 |
| Field Extraction | ~500 | $0.86 |
| **Total** | **~2,600** | **~$5.15** |

### Infrastructure

- 2× NVIDIA A100 80GB PCIe (Azure Standard_NC24ads_A100_v4)
- Python 3.10.2
- EVoC 0.3.1, Nomic nomic-embed-text-v1.5, DuckDB 1.5.1

---

## Requirements

- Python 3.10+
- CUDA GPU (for Nomic embeddings; CPU fallback available)
- Azure OpenAI access (for field discovery, labeling, extraction)

**Core dependencies:** `evoc`, `sentence-transformers`, `torch`, `bertopic`, `umap-learn`, `hdbscan`, `scikit-learn`, `optuna`, `duckdb`, `lancedb`, `pymupdf`, `openai`, `networkx`

**Optional:** `cuml` (RAPIDS via conda — 50–100× faster UMAP/HDBSCAN), `ray` (distributed Optuna), `streamlit`, `fastapi`, `plotly`
