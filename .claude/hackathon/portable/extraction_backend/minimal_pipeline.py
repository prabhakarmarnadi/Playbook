"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ MINIMAL PIPELINE — ARCHITECTURAL BREAKDOWN FOR MICROSERVICES DECOMPOSITION ║
╚══════════════════════════════════════════════════════════════════════════════╝

This file (2438 lines) is the monolithic extraction backend.  It implements a
7-stage pipeline that turns a directory of PDF agreements into structured field
extractions, cluster profiles, and incremental-assignment capabilities.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ENTRY POINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1a. CLI (main)
      invocation:  python -m core.minimal_pipeline <subcommand> [args]
      subcommands:
        - "run"    → full pipeline on a PDF directory
        - "assign" → incremental single-document assignment to existing clusters

  1b. run_minimal_pipeline(pdf_dir, account_id, config, callback)
      async function — the full 7-stage pipeline.
      Returns: dict with run_id, timing, cluster counts, extraction counts,
               profiles directory path, and LLM stats.

  1c. assign_new_documents(pdf_paths, account_id, config)
      async function — incremental ingestion.  For each PDF: parse → chunk →
      embed → cosine-match to nearest centroid → return assignment with
      confidence tier (high / tentative / novel).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. CLI ARGUMENTS (argparse)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ "run" subcommand ──────────────────────────────────────────────────────┐
  │ POSITIONAL                                                              │
  │   pdf_dir              Path to directory of PDF files                   │
  │                                                                         │
  │ OPTIONAL                                                                │
  │   --account-id         Tenant/account ID              default: "default"│
  │   --device             Compute device                 default: "cuda"   │
  │   --max-docs           Cap on # PDFs to process       default: None     │
  │   --max-concurrency    LLM semaphore limit            default: 8        │
  │   --optuna-trials      Optuna hyperparameter trials   default: 30       │
  │   --no-optuna          Disable Optuna (use EVoC)      flag              │
  │   --discovery          Discovery backend mode         default:          │
  │                        choices: rlm_v3, rlm_v3_unified   rlm_v3_unified│
  │   --no-extraction      Skip RAG extraction phase      flag              │
  │   --run-intents        Enable Stage 5 intent disc.    flag              │
  │   --no-field-features  Disable field feature extrac.  flag              │
  │   --legacy-fields-dir  Path to apr-fields-processor   default: None     │
  │   --guided-keywords    JSON file or inline JSON dict  default: None     │
  │                        {"domain": ["kw1", "kw2"]}                       │
  │   --guided-alpha       Guided clustering strength     default: 0.3      │
  │   --no-feedback        Disable feedback/personalize   flag              │
  │   --tenant-id          Tenant ID for sidecar DB       default: None     │
  │   --no-confidence-gate Disable confidence gating      flag              │
  │   --confidence-threshold Gate threshold               default: 0.65     │
  │   --no-soft-membership Disable probabilistic routing  flag              │
  │   --enable-multidim    Enable field+obligation sigs   flag              │
  │   --db                 Custom DuckDB path             default: None     │
  │   --lance              Custom LanceDB path            default: None     │
  │   --log-level          Logging level                  default: "INFO"   │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ "assign" subcommand ───────────────────────────────────────────────────┐
  │ POSITIONAL                                                              │
  │   pdf_paths            One or more paths to PDF files (nargs="+")       │
  │                                                                         │
  │ OPTIONAL                                                                │
  │   --account-id         default: "default"                               │
  │   --device             default: "cuda"                                  │
  │   --db                 Custom DuckDB path             default: None     │
  │   --log-level          default: "INFO"                                  │
  └─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. CONFIGURATION (MinimalPipelineConfig dataclass)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  All 35 tunables with defaults:

  CLUSTERING
    device                    str    "cuda"       GPU/CPU device for embeddings
    use_optuna                bool   True         Use Optuna for hyperparameter search
    optuna_trials             int    30           Number of Optuna trials
    evoc_noise_level          float  0.3          EVoC Gaussian noise std dev

  CHUNKING
    min_chunk_chars           int    100          Minimum embedding chunk size
    max_chunk_chars           int    2000         Maximum embedding chunk size (~2K tok)
    extraction_chunk_chars    int    16000        Extraction chunk size (~4K tok)
    max_chunks_per_doc        int    80           Per-document chunk cap

  LLM
    max_llm_concurrency       int    8            Semaphore + ThreadPool workers

  DISCOVERY
    discovery_mode            str    "rlm_v3_unified"   rlm_v3 | rlm_v3_unified
    max_sample_chunks         int    30           Max chunks sampled for RLM
    max_fields_per_cluster    int    15           Field count cap per cluster
    min_cluster_chunks        int    5            Skip clusters smaller than this
    discovery_temperature     float  0.2          LLM temperature for discovery

  EXTRACTION
    min_extraction_confidence float  0.5          Min confidence to persist extraction
    extract_values            bool   True         Enable RAG value extraction

  INTENT DISCOVERY (optional Stage 5)
    run_intents               bool   False        Enable intent discovery
    min_intent_clause_chars   int    80           Min clause length for intent analysis
    intent_match_threshold    float  0.85         Cosine threshold for intent matching

  PATHS
    db_path                   str?   None         Override DuckDB path
    lance_path                str?   None         Override LanceDB path
    artifacts_dir             str?   None         Override artifacts root dir
    max_docs                  int?   None         Cap on documents to process

  FIELD FEATURES
    legacy_fields_dir         str?   None         Path to apr-fields-processor
    enable_field_features     bool   True         Extract field patterns from chunks

  GUIDED CLUSTERING
    guided_keywords           dict?  None         {domain_name: [keyword_list]}
    guided_alpha              float  0.3          Guidance strength (0-1)

  FEEDBACK & PERSONALIZATION
    enable_feedback           bool   True         Wire FeedbackStore + PersonalizationEngine
    tenant_id                 str?   None         Sidecar DB tenant ID
    feedback_data_dir         str?   None         Dir for sidecar DuckDB files

  CONFIDENCE-GATED SCALING
    enable_confidence_gate    bool   True         Gate extraction on example set quality
    confidence_gate_threshold float  0.65         Min avg confidence to proceed
    cochran_sample_fraction   float  0.3          Fraction of cluster for example set
    cochran_min_sample        int    5            Floor on example set size
    cochran_max_sample        int    30           Ceiling on example set size

  PROBABILISTIC ROUTING
    enable_soft_membership    bool   True         Compute soft cluster memberships
    soft_membership_k         int    3            Top-K clusters for soft membership
    low_confidence_threshold  float  0.40         Below = route to fallback/review

  MULTI-DIMENSIONAL SIGNALS
    enable_multidim_signals   bool   False        Use field-overlap + obligation dims
    field_overlap_weight      float  0.15         Weight for field-overlap signal
    obligation_weight         float  0.10         Weight for obligation-pattern signal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. ENVIRONMENT VARIABLES (read at runtime)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  AZURE_OPENAI_ENDPOINT          Azure OpenAI base URL
  AZURE_OPENAI_API_KEY           Azure OpenAI API key
  AZURE_OPENAI_API_VERSION       API version (default: "2024-12-01-preview")
  AZURE_OPENAI_DEPLOYMENT        Model deployment name (default: "gpt-5.4")
  RLM_MAX_ITERATIONS             RLM V3 REPL max iterations (default: 25)
  RLM_MAX_LLM_CALLS              RLM V3 max LLM calls (default: 50)
  RAG_MAX_SAMPLE_DOCS            RAG extractor doc sample (default: 10)
  RAG_MAX_RETRY_ROUNDS           RAG retry rounds (default: 3)
  RAG_ENABLE_REVIEWER            RAG reviewer agent (default: "1")
  CLUSTER_CONCURRENCY            Cluster processing concurrency (default: 3)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. END-TO-END DATA FLOW (with data shapes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  INPUT: /path/to/pdfs/ (directory of .pdf files)

  ┌────────────────────────────────────────────────────────────────────────┐
  │ STAGE 1: PDF PARSING                                                   │
  │                                                                        │
  │   Input:  pdf_dir (filesystem path)                                    │
  │   Output: list[dict] — each {agreement_id, filename, raw_text}         │
  │   Cache:  data/parsed_text/ with manifest.json keyed on source_dir     │
  │                                                                        │
  │   Imports: core.pdf_parser (parse_pdf_directory, load_parsed_agreements)│
  │   Data shape: N documents → N agreement dicts with raw_text strings    │
  └────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ STAGE 2: CHUNK + EMBED + FIELD FEATURES                                │
  │                                                                        │
  │   Sub-steps:                                                           │
  │     2a. Generate 3 chunk types per document:                           │
  │         - embedding chunks  (≤2000 chars) → used for clustering + RAG  │
  │         - extraction chunks (≤16000 chars) → stored for future use     │
  │         - clause chunks     (full clause)  → clause records with       │
  │                                              headers, numbering        │
  │                                                                        │
  │     2b. Field Feature Extraction (optional):                           │
  │         Uses apr-fields-processor patterns to annotate each embedding   │
  │         chunk with {matched_fields, matched_categories, top_confidence,│
  │         n_matches, summary}.                                           │
  │         Also builds per-doc feature summaries for macro clustering.     │
  │                                                                        │
  │     2c. Nomic Embedding:                                               │
  │         - chunk_embeddings:   (N_chunks, 768) float32 L2-normalised    │
  │         - summary_embeddings: (N_docs, 768) float32 L2-normalised      │
  │         Embedder: NomicEmbedder (nomic-embed-text-v1.5, CUDA)          │
  │                                                                        │
  │     2d. Persist to DuckDB + LanceDB:                                   │
  │         - agreements, chunks, clauses → DuckDB                         │
  │         - chunk vectors, doc vectors → LanceDB                         │
  │                                                                        │
  │   Imports: core.chunker, core.nomic_embedder, core.field_features,     │
  │            core.macro_clusterer, core.store, core.lancedb_store         │
  │   Data shapes:                                                         │
  │     all_embedding_chunks:  list[dict] (chunk_id, text, agreement_id,   │
  │                            chunk_index, clause_id, field_features)      │
  │     all_extraction_chunks: list[dict] (larger chunks for future use)   │
  │     all_clauses:           list[dict] (clause metadata + linked IDs)   │
  │     chunk_embeddings:      np.ndarray (N_chunks, 768)                  │
  │     summary_embeddings:    np.ndarray (N_docs, 768)                    │
  └────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ STAGE 3: PARALLEL EVoC CLUSTERING (macro ∥ micro)                      │
  │                                                                        │
  │   3a. Macro EVoC — clusters document summaries:                        │
  │       Input:  summary_embeddings (N_docs, 768)                         │
  │       Method: Optuna-tuned HDBSCAN or EVoC + layer selection           │
  │       Output: domain_map {topic_id: {domain_id, agreement_indices,     │
  │               label}}, domains persisted to DuckDB                     │
  │                                                                        │
  │   3b. Micro EVoC — clusters clause/embedding chunks:                   │
  │       Input:  chunk_embeddings (N_chunks, 768), optionally modified by:│
  │         - Guided clustering: compute_guided_embeddings()               │
  │           (applies attractor centroids from user keyword groups)        │
  │         - Multi-dim signals: compute_multidimensional_embeddings()      │
  │           (blends field-overlap + obligation-pattern dimensions)        │
  │       Method: Optuna-tuned HDBSCAN or EVoC + layer selection           │
  │       Output: cluster records with labels, keywords, centroids         │
  │       For each cluster: BM25 keyword extraction, LLM labelling,        │
  │         centroid computation, domain linkage via majority vote          │
  │                                                                        │
  │   3c. Soft (probabilistic) memberships:                                │
  │       Input:  chunk_embeddings, cluster centroids                      │
  │       Method: softmax over top-K cosine similarities (temperature=5.0) │
  │       Output: soft_memberships {chunk_idx: {cluster_id: prob}}         │
  │               low_confidence_chunks: list[int] (max_prob < 0.40)       │
  │                                                                        │
  │   Imports: core.evoc_pipeline, core.topic_merger                       │
  │   Data shapes:                                                         │
  │     clustering_embeddings: np.ndarray (N_chunks, 768) — possibly       │
  │       modified by guided/multidim transforms.  Original                │
  │       chunk_embeddings preserved for centroid + assignment ops.         │
  │     cluster centroids: dict[str, np.ndarray(768)]                      │
  │     soft_memberships: dict[str, dict[str, float]]                      │
  └────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ STAGE 4: RLM V3 DISCOVERY + CONFIDENCE-GATED RAG EXTRACTION            │
  │                                                                        │
  │   Per-cluster processing (_process_cluster), run concurrently under    │
  │   asyncio.Semaphore(CLUSTER_CONCURRENCY, default 3).                   │
  │                                                                        │
  │   Phase 1 — Discovery:                                                 │
  │     - Diversity-sample chunks across agreements                        │
  │     - Run RLM V3 (agentic REPL with 12 tools) for field + clause      │
  │       sub-type + obligation discovery                                  │
  │     - If rlm_v3_unified: also produces "free" inline extractions       │
  │     - Fallback: single-shot LLM standard discovery prompt              │
  │     - Output: list[dict] field definitions, rlm_meta dict,             │
  │       inline_extractions list                                          │
  │                                                                        │
  │   Phase 2 — Confidence-gated example set:                              │
  │     - Cochran sample: extract on small subset first                    │
  │     - Measure aggregate confidence (mean, median, p25)                 │
  │     - If mean_confidence < threshold (0.65): SKIP full extraction      │
  │       (saves LLM cost on low-quality clusters)                         │
  │     - If passed: proceed to Phase 3                                    │
  │                                                                        │
  │   Phase 3 — Production RAG extraction:                                 │
  │     - RAGExtractor.extract_cluster() with retries + reviewer agent     │
  │     - Personalization: tenant correction examples injected into context│
  │     - Personalization: confidence thresholds overridden from feedback   │
  │     - Outputs persisted to DuckDB (fields + extractions tables)        │
  │                                                                        │
  │   Post: Build ClusterProfile ("agreement genome") per cluster.         │
  │   Persist profiles as JSON to artifacts/<account>/<run>/profiles/      │
  │   Attach soft membership data to profiles.                             │
  │                                                                        │
  │   Imports: core.rlm_v3, core.rag_extractor, core.embedder,            │
  │            core.feedback_store, core.personalization                    │
  │   Data shapes:                                                         │
  │     fields per cluster: list[dict] {name, type, description, examples} │
  │     extractions: persisted rows {extraction_id, agreement_id, field_id,│
  │                  value, confidence, source_chunk_id}                    │
  │     ClusterProfile: dataclass with 14 fields (see class definition)    │
  └────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ STAGE 5: [OPTIONAL] INTENT DISCOVERY                                   │
  │                                                                        │
  │   Only runs if config.run_intents=True (CLI: --run-intents).           │
  │   For each clause not yet processed:                                   │
  │     - LLM prompt identifying obligations, rights, prohibitions, etc.   │
  │     - Intents matched against IntentRegistry for canonical types       │
  │     - Persisted to clause_intents table in DuckDB                      │
  │                                                                        │
  │   Imports: core.intent_extractor                                       │
  │   Data shapes: {label, summary, party_from, party_to, attributes}      │
  └────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ STAGE 6: INCREMENTAL CLUSTER ASSIGNER INITIALIZATION                   │
  │                                                                        │
  │   Loads all cluster centroids into IncrementalClusterAssigner.          │
  │   Uses personalization thresholds if available.                         │
  │   3-tier confidence model:                                             │
  │     HIGH     (>0.85): assign + update centroid running mean            │
  │     TENTATIVE (0.60-0.85): assign + flag for human review             │
  │     NOVEL    (<0.60): buffer for periodic re-clustering               │
  │                                                                        │
  │   Imports: core.cluster_assigner                                       │
  │   Data shapes: IncrementalClusterAssigner with centroid dict            │
  └────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ POST: MANIFEST + CLEANUP                                               │
  │                                                                        │
  │   - Close feedback store connection                                    │
  │   - Save pipeline_manifest.json to artifacts/<account>/<run>/          │
  │   - Return pipeline_stats dict with all timing, counts, config         │
  └────────────────────────────────────────────────────────────────────────┘

  OUTPUT:
    - DuckDB: agreements, chunks, clauses, domains, clusters,
              cluster_assignments, fields, extractions, clause_intents,
              centroids, cluster_layers
    - LanceDB: chunk vectors (N_chunks, 768), document vectors (N_docs, 768)
    - JSON artifacts: cluster profiles, pipeline manifest
    - Sidecar DuckDB (feedback): feedback events, personalization state

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. DATA STORES & I/O PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ DuckDB (main) ─────────────────────────────────────────────────────────┐
  │ Path: data/<account_id>.duckdb                                          │
  │ Access: core.store.ClusteringStore                                      │
  │ Tables: agreements, chunks, clauses, domains, clusters,                 │
  │         cluster_assignments, fields, extractions, clause_intents,       │
  │         centroids, cluster_layers, pipeline_runs                        │
  │ Read-heavy in: Stage 4 (chunk queries per cluster), Stage 6 (centroids)│
  │ Write-heavy in: Stage 2 (bulk chunk insert), Stage 3 (cluster create), │
  │                 Stage 4 (field + extraction inserts)                    │
  │ Pattern: single connection, sync operations via store wrapper           │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ LanceDB ───────────────────────────────────────────────────────────────┐
  │ Path: data/<account_id>.lance                                           │
  │ Access: core.lancedb_store.LanceVectorStore                             │
  │ Tables: chunks (vector + metadata), documents (vector + metadata)       │
  │ Written once in Stage 2; read by RAG extractor for nearest-neighbor     │
  │ chunk retrieval during extraction.                                      │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ DuckDB sidecar (feedback) ─────────────────────────────────────────────┐
  │ Path: data/feedback_<tenant_id>.duckdb                                  │
  │ Access: core.feedback_store.FeedbackStore                               │
  │ Tables: feedback_events, virtual_clusters, personalization_state        │
  │ 8 action types: field_accept, field_reject, field_rename, threshold_adj,│
  │                 cluster_merge, cluster_split, intent_override, custom   │
  │ 6 entity types: field, cluster, extraction, intent, threshold, custom   │
  │ Supports temporal decay scoring (recent feedback weighted higher)       │
  │ Read: at pipeline init (get_context), at assign_new_documents init      │
  │ Write: from external API / UI (not written by this pipeline directly)   │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ JSON Artifacts ────────────────────────────────────────────────────────┐
  │ Path: data/artifacts/<account_id>/<run_id>/                             │
  │   profiles/<cluster_id>.json  — ClusterProfile (one per cluster)        │
  │   pipeline_manifest.json      — full run stats, timings, config         │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ Filesystem (parsed text cache) ────────────────────────────────────────┐
  │ Path: data/parsed_text/                                                 │
  │   *.json per PDF, manifest.json keyed on source_dir path                │
  │ Avoids re-parsing PDFs across runs.                                     │
  └─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. EXTERNAL DEPENDENCIES (import graph from this file)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Standard lib: asyncio, json, logging, os, time, uuid, math, re,
                concurrent.futures, dataclasses, functools, pathlib, typing

  Third-party:
    - numpy               Array operations, cosine similarity, centroid math
    - dotenv              Load .env for Azure credentials
    - openai (AzureOpenAI) LLM calls via Azure OpenAI SDK

  Internal core modules:
    - config              DATA_DIR path constant
    - core.store          ClusteringStore (DuckDB wrapper)
    - core.lancedb_store  LanceVectorStore (LanceDB wrapper)
    - core.nomic_embedder NomicEmbedder (Nomic 768-d CUDA embeddings)
    - core.embedder       Embedder singleton (used by RAG + assigner)
    - core.pdf_parser     PDF parsing (parse_pdf_directory, parse_pdf)
    - core.chunker        3-type chunking (SemanticChunkerConfig, etc.)
    - core.macro_clusterer get_field_summaries (document summary generation)
    - core.evoc_pipeline  EVoC clustering (evoc_cluster, evoc_optimize, etc.)
    - core.topic_merger   HierarchicalTopicMerger (imported but used by EVoC)
    - core.rlm_v3         RLM V3 discovery engine (agentic REPL, 12 tools)
    - core.rag_extractor  RAGExtractor (agentic RAG with retries + reviewer)
    - core.field_features FieldFeatureExtractor (pattern/keyword matching)
    - core.intent_extractor IntentRegistry (canonical intent types)
    - core.feedback_store FeedbackStore (sidecar DuckDB for human feedback)
    - core.personalization PersonalizationEngine (context from feedback)
    - core.cluster_assigner IncrementalClusterAssigner (3-tier routing)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. CONCURRENCY MODEL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Framework: asyncio event loop (single-threaded, cooperative)

  LLM calls (AsyncLLMClient):
    - asyncio.Semaphore(max_llm_concurrency) limits in-flight LLM requests
    - ThreadPoolExecutor(max_workers=max_concurrency) runs sync OpenAI SDK
      calls via loop.run_in_executor()
    - Exponential backoff retry: 1.5s * (attempt + 1), up to 3 retries

  Cluster processing (Stage 4):
    - asyncio.Semaphore(CLUSTER_CONCURRENCY, default 3) limits parallel
      _process_cluster coroutines
    - asyncio.gather() dispatches all clusters concurrently (bounded by sem)
    - return_exceptions=True — errors logged, not fatal

  RLM V3 calls:
    - Sync execution offloaded via loop.run_in_executor(None, _run_sync)
    - Each RLM V3 session is single-threaded (REPL state machine)

  Embedding:
    - NomicEmbedder: batched GPU inference (batch_size=8)
    - Singleton pattern via get_instance()

  NOTE for microservices: The current async model assumes in-process
  concurrency.  Decomposing into services requires replacing the
  Semaphore-based throttling with proper distributed rate limiting
  (e.g., token bucket via Redis, or queue-based backpressure).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. LOGICAL MODULE GROUPS (suggested microservice boundaries)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─ MODULE A: Ingestion Service ───────────────────────────────────────────┐
  │ Responsibilities:                                                       │
  │   - PDF parsing (parse_pdf_directory, parse_single_pdf)                 │
  │   - 3-type chunking (generate_typed_chunks)                             │
  │   - Field feature extraction (FieldFeatureExtractor)                    │
  │   - Clause header extraction                                            │
  │   - Parsed text caching                                                 │
  │                                                                         │
  │ In this file (functions/classes):                                        │
  │   - Stage 1 block in run_minimal_pipeline                               │
  │   - Stage 2 block (chunk generation + field features portion)           │
  │                                                                         │
  │ External deps: core.pdf_parser, core.chunker, core.field_features       │
  │ Input: PDF files (filesystem or blob)                                   │
  │ Output: list[dict] chunks with metadata, list[dict] clause records      │
  │ Communication pattern: batch job, request-reply, or event-driven        │
  │   (emit "documents.ingested" event with chunk payloads)                │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ MODULE B: Embedding Service ───────────────────────────────────────────┐
  │ Responsibilities:                                                       │
  │   - Nomic embedding (768-d, L2-normalised)                              │
  │   - GPU resource management (singleton, warmup, batching)               │
  │   - Document summary generation (get_field_summaries)                   │
  │   - LanceDB vector persistence                                         │
  │                                                                         │
  │ In this file (functions/classes):                                        │
  │   - Stage 2 embed portion + _get_embedder()                             │
  │   - LanceDB upsert calls                                                │
  │                                                                         │
  │ External deps: core.nomic_embedder, core.lancedb_store,                 │
  │                core.macro_clusterer                                      │
  │ Input: list[str] texts                                                  │
  │ Output: np.ndarray (N, 768)                                             │
  │ Communication pattern: gRPC or REST with numpy serialization            │
  │   (request: texts → response: embeddings as bytes/arrow)               │
  │   This is a stateless, GPU-bound service — ideal for autoscaling.      │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ MODULE C: Clustering Service ──────────────────────────────────────────┐
  │ Responsibilities:                                                       │
  │   - Macro EVoC (document-level domains)                                 │
  │   - Micro EVoC (chunk-level clause clusters)                            │
  │   - Optuna hyperparameter search                                        │
  │   - Guided clustering (attractor centroids from keyword groups)         │
  │   - Multi-dimensional signal aggregation                                │
  │   - Soft membership computation                                         │
  │   - BM25 keyword extraction                                             │
  │   - LLM cluster labelling                                               │
  │   - Centroid computation + persistence                                  │
  │                                                                         │
  │ In this file (functions/classes):                                        │
  │   - extract_cluster_keywords(), _get_stopwords(), _tokenize_for_bm25() │
  │   - compute_guided_embeddings()                                         │
  │   - compute_multidimensional_embeddings()                               │
  │   - compute_soft_memberships(), identify_low_confidence_chunks()        │
  │   - Stage 3 block in run_minimal_pipeline                               │
  │                                                                         │
  │ External deps: core.evoc_pipeline, core.topic_merger, numpy             │
  │ Input: embeddings (N, 768), config params                               │
  │ Output: cluster assignments, centroids, labels, soft memberships        │
  │ Communication pattern: batch job (triggered after embedding complete)    │
  │   Event: "embeddings.ready" → runs clustering → emits "clusters.ready" │
  │   CPU-heavy (HDBSCAN, Optuna) — can run on dedicated compute.          │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ MODULE D: Discovery & Extraction Service ──────────────────────────────┐
  │ Responsibilities:                                                       │
  │   - RLM V3 agentic field discovery (12-tool REPL)                       │
  │   - Standard single-shot LLM discovery (fallback)                       │
  │   - Confidence-gated extraction (Cochran sampling)                      │
  │   - Agentic RAG extraction (with retries + reviewer)                    │
  │   - ClusterProfile assembly                                             │
  │   - Personalization context injection                                   │
  │   - Field schema normalization                                          │
  │                                                                         │
  │ In this file (functions/classes):                                        │
  │   - ClusterProfile dataclass + to_dict/from_dict                        │
  │   - AsyncLLMClient (rate-limited Azure OpenAI wrapper)                  │
  │   - STANDARD_DISCOVERY_PROMPT template                                  │
  │   - _normalize_field()                                                  │
  │   - _discover_fields_standard(), _discover_fields_rlm_v3(),            │
  │     _discover_and_extract_rlm_v3()                                      │
  │   - _process_cluster() (3-phase: discovery → gate → extraction)        │
  │   - _build_rlm_context_section(), _run_rag_extraction()                │
  │   - _build_cluster_profile()                                            │
  │   - _diversity_sample_chunks()                                          │
  │   - _cochran_sample_size(), _compute_extraction_confidence()            │
  │   - Stage 4 block in run_minimal_pipeline                               │
  │                                                                         │
  │ External deps: core.rlm_v3, core.rag_extractor, core.embedder,         │
  │                openai (AzureOpenAI)                                      │
  │ Input: cluster metadata + chunk texts + field features                  │
  │ Output: field definitions, extractions, ClusterProfiles                 │
  │ Communication pattern: fan-out per cluster (queue-based recommended)    │
  │   Each cluster is an independent work item → natural parallelization.   │
  │   Currently bounded by asyncio.Semaphore; in microservices, use a       │
  │   task queue (e.g., Celery, Azure Service Bus, SQS).                   │
  │   LLM calls are the bottleneck — needs distributed rate limiting.      │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ MODULE E: Intent Service ──────────────────────────────────────────────┐
  │ Responsibilities:                                                       │
  │   - Legal intent discovery per clause                                   │
  │   - Intent type registry (canonical matching)                           │
  │   - Intent persistence                                                  │
  │                                                                         │
  │ In this file (functions/classes):                                        │
  │   - INTENT_PROMPT template                                              │
  │   - _process_intents()                                                  │
  │   - Stage 5 block in run_minimal_pipeline                               │
  │                                                                         │
  │ External deps: core.intent_extractor                                    │
  │ Input: clause records from DuckDB                                       │
  │ Output: clause_intents records                                          │
  │ Communication pattern: optional stage, can be async event-driven        │
  │   Triggered by "extraction.complete" event.                            │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ MODULE F: Assignment Service ──────────────────────────────────────────┐
  │ Responsibilities:                                                       │
  │   - IncrementalClusterAssigner initialization (from centroids)          │
  │   - Single-document assignment (parse → embed → cosine match)           │
  │   - 3-tier confidence routing (high / tentative / novel)               │
  │   - Running-mean centroid updates for high-confidence assignments       │
  │   - Novel document buffering for re-clustering trigger                  │
  │   - Personalization threshold loading from feedback store               │
  │                                                                         │
  │ In this file (functions/classes):                                        │
  │   - Stage 6 block in run_minimal_pipeline                               │
  │   - assign_new_documents() entry point                                  │
  │                                                                         │
  │ External deps: core.cluster_assigner, core.nomic_embedder,              │
  │                core.pdf_parser, core.feedback_store, core.personalization│
  │ Input: PDF file paths + account_id                                      │
  │ Output: list[dict] assignment results with confidence tiers             │
  │ Communication pattern: low-latency REST API (<500ms SLA)                │
  │   This is the hot path for real-time document intake.                   │
  │   Needs: in-memory centroid cache, pre-loaded embedder, connection pool.│
  │   Ideal as a standalone service with health checks and autoscaling.    │
  └─────────────────────────────────────────────────────────────────────────┘

  ┌─ MODULE G: Feedback & Personalization Service ──────────────────────────┐
  │ Responsibilities:                                                       │
  │   - FeedbackStore sidecar DB management                                 │
  │   - PersonalizationEngine context computation                           │
  │   - Tenant threshold overrides, correction examples, field weights      │
  │   - Virtual cluster management                                          │
  │   - Temporal decay scoring                                              │
  │                                                                         │
  │ In this file: wiring in run_minimal_pipeline init +                     │
  │               assign_new_documents feedback loading                     │
  │                                                                         │
  │ External deps: core.feedback_store, core.personalization                │
  │ Input: tenant_id, feedback events (from UI / API)                       │
  │ Output: PersonalizationContext (thresholds, corrections, weights)        │
  │ Communication pattern: read-heavy, low-write.  Could be a shared       │
  │   microservice with per-tenant caching.  Other services query it for    │
  │   context before processing.                                            │
  └─────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10. INTER-MODULE DATA CONTRACTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A → B:  chunks with texts + metadata (JSON serializable dicts)
  B → C:  embeddings as np.ndarray(N, 768) + chunk_ids for alignment
  C → D:  cluster assignments (cluster_id → [chunk_ids]), centroids,
          soft_memberships, keywords
  D → DuckDB: field definitions + extraction rows (persisted directly)
  D → JSON:   ClusterProfile per cluster (agreement genome)
  C → F:  centroids dict (cluster_id → np.ndarray(768))
  G → D:  PersonalizationContext (thresholds, correction_examples, weights)
  G → F:  PersonalizationContext (threshold overrides for assignment)

  Shared state (currently via DuckDB connection):
    - Stages 2-4 share the ClusteringStore connection for reads + writes.
    - In microservices: replace with API calls or shared DB with connection
      pooling.  Consider CQRS: write via events, read via materialized views.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
11. FUNCTION INDEX (line-approximate, alphabetical by section)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DATA STRUCTURES
    ClusterProfile (dataclass)           — lines ~45-125
    MinimalPipelineConfig (dataclass)    — lines ~665-730

  BM25 KEYWORDS
    _get_stopwords()                     — lines ~135-175
    _tokenize_for_bm25()                 — lines ~185-190
    extract_cluster_keywords()           — lines ~193-260

  EMBEDDING SPACE TRANSFORMS
    compute_guided_embeddings()          — lines ~270-360
    compute_multidimensional_embeddings() — lines ~405-510
    compute_soft_memberships()           — lines ~512-575
    identify_low_confidence_chunks()     — lines ~578-595

  CONFIDENCE GATE
    _cochran_sample_size()               — lines ~600-620
    _compute_extraction_confidence()     — lines ~622-655

  SAMPLING
    _diversity_sample_chunks()           — lines ~362-400

  LLM CLIENT
    AsyncLLMClient (class)               — lines ~735-800

  DISCOVERY & EXTRACTION
    STANDARD_DISCOVERY_PROMPT            — lines ~805-830
    _normalize_field()                   — lines ~840-865
    _discover_fields_standard()          — lines ~870-895
    _discover_fields_rlm_v3()            — lines ~897-930
    _discover_and_extract_rlm_v3()       — lines ~933-970
    _process_cluster()                   — lines ~978-1195
    _build_rlm_context_section()         — lines ~1198-1225
    _run_rag_extraction()                — lines ~1228-1265
    _build_cluster_profile()             — lines ~1268-1310

  UTILITY
    _get_embedder()                      — lines ~1313-1316

  INTENT DISCOVERY
    INTENT_PROMPT                        — lines ~1325-1355
    _process_intents()                   — lines ~1358-1430

  PIPELINE ORCHESTRATION
    _StageTimer (class)                  — lines ~1438-1445
    run_minimal_pipeline()               — lines ~1448-2200
    assign_new_documents()               — lines ~2210-2380

  CLI
    main()                               — lines ~2385-2438

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
12. NOTES FOR MICROSERVICES DECOMPOSITION AGENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. SHARED STATE: Currently all stages share a single DuckDB connection
     (ClusteringStore) and a single asyncio event loop.  This is the main
     coupling point.  Decomposition requires either:
       a. A shared database with connection pooling + row-level locking, or
       b. Event-sourced architecture where each service owns its data and
          publishes events for downstream consumers to materialise.

  2. GPU AFFINITY: NomicEmbedder and EVoC clustering are GPU-bound.
     Modules B and C should co-locate on GPU instances.  Module D is
     LLM-bound (network I/O) and can run on CPU instances.

  3. LATENCY-CRITICAL PATH: assign_new_documents() (Module F) is the
     real-time path with <500ms SLA.  It needs pre-warmed embedder,
     in-memory centroid cache, and should NOT share resources with the
     batch pipeline.

  4. LLM RATE LIMITING: The current Semaphore-based approach does not
     survive process boundaries.  Use a distributed token bucket (Redis)
     or queue-based backpressure with per-tenant fairness.

  5. NATURAL PARALLELISATION BOUNDARIES:
     - Per-cluster: Stage 4 already processes clusters independently.
       Each cluster = one queue message.
     - Per-document: Stage 1-2 can process documents independently.
     - Stage 3 (clustering) is inherently global (needs all embeddings)
       and cannot be parallelised at the document level.

  6. IDEMPOTENCY: run_ids and entity IDs use uuid4 hex — no idempotency
     keys.  For reliable message processing, add deterministic ID
     generation (e.g., hash of content + account_id).

  7. FEEDBACK LOOP: The FeedbackStore → PersonalizationEngine → extraction
     path crosses Modules G → D.  In microservices, Module D should fetch
     PersonalizationContext from Module G via API before processing each
     cluster (cache with TTL to avoid per-cluster network calls).

  8. PROFILE PERSISTENCE: ClusterProfiles are written as JSON files.
     Consider moving to a profiles table in DuckDB or a dedicated
     document store (CosmosDB, S3+metadata) for queryability.

Usage:
    python -m core.minimal_pipeline run /path/to/pdfs --account-id acct_001
    python -m core.minimal_pipeline assign doc1.pdf doc2.pdf --account-id acct_001
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ── ClusterProfile — the "agreement genome" per cluster ──────────────────────


@dataclass
class ClusterProfile:
    """Structured profile for a single cluster — the 'agreement genome'.

    Bundles every signal the pipeline has discovered about a cluster into one
    serializable object.  Persisted as JSON so downstream consumers (API,
    incremental assigner, feedback loop) have a single source of truth for
    what defines this cluster.

    Fields:
        cluster_id:      Unique cluster identifier (e.g. 'c_abcd1234').
        domain_id:       Parent macro-domain ID.
        label:           Human-readable cluster label (from LLM labeller).
        description:     One-sentence description of the cluster.
        keywords:        BM25-weighted thematic keywords.
        field_definitions: Field schemas discovered by RLM / standard fallback.
        clause_types:    Sub-clause-type descriptors from RLM meta.
        obligations:     Key obligations found in RLM meta.
        centroid:        L2-normalised cluster centroid (768-d Nomic).
        chunk_count:     Number of chunks assigned to this cluster.
        agreement_count: Number of distinct agreements in this cluster.
        extraction_stats: Aggregate extraction quality metrics.
        soft_membership:  {chunk_id: membership_probability} for borderline chunks.
        confidence_gate:  Result of confidence-gated scaling check.
    """

    cluster_id: str = ""
    domain_id: str = ""
    label: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    field_definitions: list[dict] = field(default_factory=list)
    clause_types: list[dict] = field(default_factory=list)
    obligations: list[dict] = field(default_factory=list)
    centroid: Any = None   # np.ndarray, serialized as list for JSON
    chunk_count: int = 0
    agreement_count: int = 0
    extraction_stats: dict[str, Any] = field(default_factory=dict)
    soft_membership: dict[str, float] = field(default_factory=dict)
    confidence_gate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict (converts ndarray centroid to list)."""
        d = {
            "cluster_id": self.cluster_id,
            "domain_id": self.domain_id,
            "label": self.label,
            "description": self.description,
            "keywords": self.keywords,
            "field_definitions": self.field_definitions,
            "clause_types": self.clause_types,
            "obligations": self.obligations,
            "centroid": self.centroid.tolist() if isinstance(self.centroid, np.ndarray) else self.centroid,
            "chunk_count": self.chunk_count,
            "agreement_count": self.agreement_count,
            "extraction_stats": self.extraction_stats,
            "soft_membership": self.soft_membership,
            "confidence_gate": self.confidence_gate,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ClusterProfile":
        """Deserialize from JSON dict."""
        centroid = d.get("centroid")
        if centroid is not None and not isinstance(centroid, np.ndarray):
            centroid = np.array(centroid, dtype=np.float32)
        return cls(
            cluster_id=d.get("cluster_id", ""),
            domain_id=d.get("domain_id", ""),
            label=d.get("label", ""),
            description=d.get("description", ""),
            keywords=d.get("keywords", []),
            field_definitions=d.get("field_definitions", []),
            clause_types=d.get("clause_types", []),
            obligations=d.get("obligations", []),
            centroid=centroid,
            chunk_count=d.get("chunk_count", 0),
            agreement_count=d.get("agreement_count", 0),
            extraction_stats=d.get("extraction_stats", {}),
            soft_membership=d.get("soft_membership", {}),
            confidence_gate=d.get("confidence_gate", {}),
        )


# ── BM25-weighted keyword extraction ─────────────────────────────────────────

# Minimal legal-aware stopwords (standard English + common legal boilerplate)
_STOPWORDS: set[str] | None = None


def _get_stopwords() -> set[str]:
    """Return a cached set of stopwords for BM25 tokenization.

    Includes standard English stopwords plus common legal boilerplate terms
    (e.g. 'herein', 'thereof', 'pursuant') that appear everywhere in contracts
    and add no discriminative signal for clustering.
    """
    global _STOPWORDS
    if _STOPWORDS is not None:
        return _STOPWORDS
    _STOPWORDS = {
        # Standard English
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can", "need",
        "it", "its", "this", "that", "these", "those", "he", "she", "they",
        "we", "you", "i", "me", "him", "her", "us", "them", "my", "your",
        "his", "our", "their", "which", "who", "whom", "what", "where",
        "when", "how", "why", "if", "then", "than", "so", "no", "not",
        "only", "very", "just", "also", "more", "most", "any", "all",
        "each", "every", "both", "few", "some", "such", "other", "own",
        "same", "about", "above", "after", "again", "against", "below",
        "between", "during", "into", "through", "under", "until", "upon",
        "over", "out", "up", "down", "off", "here", "there", "now",
        # Legal boilerplate that appears everywhere and adds no signal
        "shall", "herein", "hereof", "hereby", "thereof", "therein",
        "thereto", "whereas", "pursuant", "notwithstanding", "provided",
        "including", "without", "limitation", "respect", "accordance",
        "subject", "applicable", "otherwise", "foregoing", "hereunder",
        "set", "forth", "whether", "unless", "except", "prior",
        "following", "upon", "within", "under", "between",
        "section", "article", "paragraph", "clause", "subsection",
    }
    return _STOPWORDS


import math
import re as _re
from collections import Counter as _Counter


def _tokenize_for_bm25(text: str) -> list[str]:
    """Lowercase, split on non-alpha, drop short tokens and stopwords."""
    stops = _get_stopwords()
    return [
        w for w in _re.findall(r'[a-z]{3,}', text.lower())
        if w not in stops and len(w) >= 3
    ]


def extract_cluster_keywords(
    cluster_chunk_texts: list[str],
    all_corpus_texts: list[str] | None = None,
    corpus_df: dict[str, int] | None = None,
    corpus_size: int | None = None,
    top_k: int = 20,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[str]:
    """Extract thematic keywords for a cluster using BM25-style TF-IDF weighting.

    Words are scored by how important they are *within this cluster* relative
    to the full corpus. High BM25 score = frequent in cluster, rare globally.

    Args:
        cluster_chunk_texts: Chunk texts belonging to this cluster.
        all_corpus_texts: Full corpus texts (all chunks across all clusters).
            Used to compute document frequency if corpus_df not provided.
        corpus_df: Pre-computed {term: doc_frequency} counts across the corpus.
        corpus_size: Total number of documents in the corpus.
        top_k: Number of keywords to return.
        k1, b: BM25 saturation and length normalization parameters.

    Returns:
        List of up to top_k keywords sorted by BM25 score descending.
    """
    if not cluster_chunk_texts:
        return []

    # Build corpus DF if not provided
    N = corpus_size or (len(all_corpus_texts) if all_corpus_texts else len(cluster_chunk_texts))
    if corpus_df is None:
        if all_corpus_texts:
            corpus_df = {}
            for doc in all_corpus_texts:
                for term in set(_tokenize_for_bm25(doc)):
                    corpus_df[term] = corpus_df.get(term, 0) + 1
        else:
            # Fallback: use cluster as its own corpus (less discriminative)
            corpus_df = {}
            for doc in cluster_chunk_texts:
                for term in set(_tokenize_for_bm25(doc)):
                    corpus_df[term] = corpus_df.get(term, 0) + 1

    # Concatenate cluster texts as "the cluster document"
    cluster_tokens = []
    for t in cluster_chunk_texts:
        cluster_tokens.extend(_tokenize_for_bm25(t))

    if not cluster_tokens:
        return []

    tf = _Counter(cluster_tokens)
    doc_len = len(cluster_tokens)
    avgdl = doc_len  # single-doc BM25, but we still use the formula for saturation

    # Score each term with BM25-style weighting
    scores: dict[str, float] = {}
    for term, freq in tf.items():
        df = corpus_df.get(term, 0)
        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        # TF saturation
        tf_norm = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * doc_len / max(avgdl, 1)))
        scores[term] = idf * tf_norm

    # Sort by score, return top_k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in ranked[:top_k]]


# ── Guided (seeded) clustering ────────────────────────────────────────────────


def compute_guided_embeddings(
    embeddings: np.ndarray,
    chunk_texts: list[str],
    keyword_groups: dict[str, list[str]],
    embed_fn,
    alpha: float = 0.3,
) -> np.ndarray:
    """Augment embeddings to steer clustering toward user-specified domains.

    For each keyword group, we embed the keywords to form an "attractor"
    centroid. Then for each chunk, we compute its cosine similarity to
    every attractor centroid. Chunks with high similarity get their
    embedding nudged toward that attractor.

    This modifies the geometry of the embedding space so that EVoC/HDBSCAN
    naturally forms clusters aligned with the user's domains — without
    changing the algorithm itself.

    Args:
        embeddings: (N, D) L2-normalized chunk embeddings.
        chunk_texts: Parallel list of chunk texts (used for keyword matching boost).
        keyword_groups: {domain_name: [keyword1, keyword2, ...]}.
            E.g. {"termination": ["termination", "cancellation", "expiry"],
                  "payment": ["payment", "fees", "compensation", "invoice"]}.
        embed_fn: Callable that takes list[str] → np.ndarray (N, D).
        alpha: Blending strength (0 = no guidance, 1 = full pull toward attractor).
            Recommended: 0.2–0.4. Higher = stronger clustering bias.

    Returns:
        (N, D) modified embeddings, L2-normalized.
    """
    if not keyword_groups or alpha <= 0:
        return embeddings

    n, d = embeddings.shape

    # 1. Embed each keyword group → attractor centroids
    attractor_names = list(keyword_groups.keys())
    attractor_centroids = []
    for name in attractor_names:
        kw_list = keyword_groups[name]
        # Embed all keywords in the group, average to get centroid
        kw_embeddings = embed_fn(kw_list)
        centroid = kw_embeddings.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        attractor_centroids.append(centroid)

    attractor_matrix = np.stack(attractor_centroids)  # (K, D)

    # 2. Compute cosine similarity of each chunk to each attractor
    # embeddings: (N, D), attractor_matrix: (K, D) → sim: (N, K)
    sim = embeddings @ attractor_matrix.T

    # 3. BM25-style keyword-presence boost: chunks containing keywords
    #    from a group get a bonus similarity to that group's attractor.
    #    This acts as a "hard match" complement to the soft embedding similarity.
    stops = _get_stopwords()
    for ki, name in enumerate(attractor_names):
        kw_set = {w.lower() for phrase in keyword_groups[name]
                  for w in _re.findall(r'[a-z]{3,}', phrase.lower())
                  if w not in stops}
        if not kw_set:
            continue
        for ci, text in enumerate(chunk_texts):
            text_tokens = set(_re.findall(r'[a-z]{3,}', text.lower()))
            overlap = len(kw_set & text_tokens)
            if overlap > 0:
                # Boost proportional to fraction of keywords matched
                boost = min(0.3, 0.1 * overlap / len(kw_set))
                sim[ci, ki] = min(1.0, sim[ci, ki] + boost)

    # 4. For each chunk, find its best-matching attractor and blend
    best_attractor_idx = sim.argmax(axis=1)  # (N,)
    best_sim = sim[np.arange(n), best_attractor_idx]  # (N,)

    # Only blend chunks with meaningful similarity (> 0.2)
    # Scale alpha by similarity so weakly-matching chunks aren't distorted
    modified = embeddings.copy()
    for i in range(n):
        if best_sim[i] > 0.2:
            effective_alpha = alpha * best_sim[i]
            target = attractor_centroids[best_attractor_idx[i]]
            modified[i] = (1.0 - effective_alpha) * modified[i] + effective_alpha * target

    # 5. Re-normalize
    norms = np.linalg.norm(modified, axis=1, keepdims=True)
    modified /= norms + 1e-12

    return modified


def _diversity_sample_chunks(
    chunk_rows: list[dict],
    max_sample: int,
) -> list[dict]:
    """Select a diverse subset of chunks for RLM discovery.

    Instead of taking the first N chunks (which may all be from the same
    agreement or the same section), this samples evenly across agreements
    and then fills remaining slots with the longest unseen chunks.

    Returns at most max_sample rows in a deterministic order.
    """
    if len(chunk_rows) <= max_sample:
        return chunk_rows

    # Group by agreement
    by_agr: dict[str, list[dict]] = {}
    for r in chunk_rows:
        by_agr.setdefault(r["agreement_id"], []).append(r)

    # Round-robin pick across agreements
    selected: list[dict] = []
    selected_set: set[int] = set()
    agr_keys = sorted(by_agr.keys())
    per_agr = max(1, max_sample // len(agr_keys))

    for agr_id in agr_keys:
        rows = by_agr[agr_id]
        # Take evenly spaced indices within this agreement's chunks
        step = max(1, len(rows) // per_agr)
        for idx in range(0, len(rows), step):
            if len(selected) >= max_sample:
                break
            row_idx = chunk_rows.index(rows[idx])
            if row_idx not in selected_set:
                selected.append(rows[idx])
                selected_set.add(row_idx)
        if len(selected) >= max_sample:
            break

    # Fill remaining slots with longest unseen chunks
    if len(selected) < max_sample:
        remaining = [(i, r) for i, r in enumerate(chunk_rows) if i not in selected_set]
        remaining.sort(key=lambda x: len(x[1].get("chunk_text", "")), reverse=True)
        for i, r in remaining:
            if len(selected) >= max_sample:
                break
            selected.append(r)

    return selected


# ── Multi-dimensional signal aggregation ──────────────────────────────────────


def compute_multidimensional_embeddings(
    embeddings: np.ndarray,
    chunk_texts: list[str],
    field_features_per_chunk: list[dict | None],
    field_overlap_weight: float = 0.15,
    obligation_weight: float = 0.10,
) -> np.ndarray:
    """Augment embeddings with field-overlap and obligation-pattern dimensions.

    Goes beyond simple keyword-based guidance by incorporating two extra
    signal types into the embedding space:

    1. **Field-overlap signal**: Chunks that match similar sets of field
       patterns (from apr-fields-processor) should cluster together.  We build
       a binary vector of matched-field flags, project it down to a small
       number of dimensions via random projection, and blend it in.

    2. **Obligation-pattern signal**: Chunks containing obligation language
       ('shall', 'must', 'agrees to', 'is required to') get a small
       directional boost that separates obligation-heavy clauses from
       declarative/definitional ones.

    Args:
        embeddings: (N, D) L2-normalised chunk embeddings.
        chunk_texts: Parallel list of chunk texts.
        field_features_per_chunk: Parallel list of field_features dicts
            (or None) per chunk.  Each dict has 'matched_fields': [str].
        field_overlap_weight: Blending weight for field-overlap signal.
        obligation_weight: Blending weight for obligation-pattern signal.

    Returns:
        (N, D) modified embeddings, L2-normalised.
    """
    n, d = embeddings.shape
    modified = embeddings.copy()

    # ── 1. Field-overlap signal ────────────────────────────────────────────
    if field_overlap_weight > 0:
        # Collect all unique field names across all chunks
        all_fields: set[str] = set()
        for ff in field_features_per_chunk:
            if ff and ff.get("matched_fields"):
                all_fields.update(ff["matched_fields"])
        field_list = sorted(all_fields)

        if len(field_list) >= 2:
            field_idx = {f: i for i, f in enumerate(field_list)}
            # Build sparse binary matrix (N, F)
            binary = np.zeros((n, len(field_list)), dtype=np.float32)
            for ci, ff in enumerate(field_features_per_chunk):
                if ff and ff.get("matched_fields"):
                    for fname in ff["matched_fields"]:
                        if fname in field_idx:
                            binary[ci, field_idx[fname]] = 1.0

            # Random projection F → min(16, D) for dimensionality reduction
            proj_dim = min(16, d)
            rng = np.random.RandomState(42)
            proj_matrix = rng.randn(len(field_list), proj_dim).astype(np.float32)
            proj_matrix /= np.linalg.norm(proj_matrix, axis=0, keepdims=True) + 1e-12

            field_signal = binary @ proj_matrix  # (N, proj_dim)
            # Pad to D dimensions
            padded = np.zeros((n, d), dtype=np.float32)
            padded[:, :proj_dim] = field_signal
            # Normalise the padded signal
            norms = np.linalg.norm(padded, axis=1, keepdims=True)
            padded /= norms + 1e-12

            modified = (1.0 - field_overlap_weight) * modified + field_overlap_weight * padded

    # ── 2. Obligation-pattern signal ──────────────────────────────────────
    if obligation_weight > 0:
        obligation_patterns = _re.compile(
            r'\b(shall|must|agrees?\s+to|is\s+required\s+to|'
            r'will\s+be\s+obligated|covenants?\s+to|undertakes?\s+to)\b',
            _re.IGNORECASE,
        )
        obligation_scores = np.zeros(n, dtype=np.float32)
        for ci, text in enumerate(chunk_texts):
            matches = obligation_patterns.findall(text)
            # Saturate at 5 matches → score in [0, 1]
            obligation_scores[ci] = min(1.0, len(matches) / 5.0)

        # Create a directional vector for obligation language
        rng = np.random.RandomState(7)
        obligation_dir = rng.randn(d).astype(np.float32)
        obligation_dir /= np.linalg.norm(obligation_dir) + 1e-12

        # Blend: only for chunks with obligation language
        for ci in range(n):
            if obligation_scores[ci] > 0.1:
                alpha = obligation_weight * obligation_scores[ci]
                modified[ci] = (1.0 - alpha) * modified[ci] + alpha * obligation_dir

    # ── Re-normalise ─────────────────────────────────────────────────────
    norms = np.linalg.norm(modified, axis=1, keepdims=True)
    modified /= norms + 1e-12
    return modified


def compute_soft_memberships(
    chunk_embeddings: np.ndarray,
    cluster_centroids: dict[str, np.ndarray],
    top_k: int = 3,
    temperature: float = 5.0,
) -> dict[str, dict[str, float]]:
    """Compute probabilistic (soft) cluster memberships for every chunk.

    Instead of hard-assigning each chunk to exactly one cluster, this
    computes a softmax distribution over the top-K nearest centroids.
    Chunks near cluster boundaries get split membership, which enables:
      - Detecting ambiguous assignments for human review
      - Probabilistic routing (extract using multiple cluster schemas)
      - Smarter incremental assignment (consider second-best cluster)

    Args:
        chunk_embeddings: (N, D) L2-normalised chunk embeddings.
        cluster_centroids: {cluster_id: centroid_vector} for all clusters.
        top_k: Number of nearest clusters to keep per chunk.
        temperature: Softmax temperature (higher = more uniform distribution).

    Returns:
        {chunk_index_str: {cluster_id: membership_probability, ...}}
        Only includes clusters with probability > 0.01.
    """
    if not cluster_centroids:
        return {}

    cids = list(cluster_centroids.keys())
    centroid_matrix = np.stack([cluster_centroids[c] for c in cids])  # (K, D)
    # cosine similarity: (N, K)
    sim = chunk_embeddings @ centroid_matrix.T

    soft: dict[str, dict[str, float]] = {}
    for ci in range(chunk_embeddings.shape[0]):
        row = sim[ci]
        # Top-K indices
        top_idx = np.argsort(row)[-top_k:][::-1]
        top_sims = row[top_idx]
        # Softmax with temperature
        exp_vals = np.exp((top_sims - top_sims.max()) * temperature)
        probs = exp_vals / (exp_vals.sum() + 1e-12)

        memberships = {}
        for ki, prob in zip(top_idx, probs):
            if prob > 0.01:
                memberships[cids[ki]] = round(float(prob), 4)
        if memberships:
            soft[str(ci)] = memberships

    return soft


def identify_low_confidence_chunks(
    soft_memberships: dict[str, dict[str, float]],
    low_threshold: float = 0.40,
) -> list[int]:
    """Identify chunk indices where the best cluster membership is below threshold.

    These chunks are near cluster boundaries and should be routed to:
      - Extraction using schemas from multiple candidate clusters
      - A human review queue, or
      - A fallback generic extraction pipeline

    Args:
        soft_memberships: Output of compute_soft_memberships().
        low_threshold: Max-membership below this → low confidence.

    Returns:
        List of chunk indices with low-confidence assignments.
    """
    low_conf = []
    for chunk_idx_str, memberships in soft_memberships.items():
        max_prob = max(memberships.values()) if memberships else 0.0
        if max_prob < low_threshold:
            low_conf.append(int(chunk_idx_str))
    return low_conf


# ── Confidence-gated extraction helpers ───────────────────────────────────────


def _cochran_sample_size(
    total: int,
    fraction: float = 0.3,
    min_sample: int = 5,
    max_sample: int = 30,
) -> int:
    """Compute example-set size for confidence-gated extraction.

    Uses a simple fraction-based approach clamped to [min_sample, max_sample].
    The idea: extract a small example-set first, measure aggregate confidence,
    and only scale to the full cluster if the confidence gate passes.

    Args:
        total: Total number of chunks in the cluster.
        fraction: Fraction of chunks for the example set.
        min_sample: Floor on sample size.
        max_sample: Ceiling on sample size.

    Returns:
        Number of chunks to include in the example set.
    """
    return max(min_sample, min(max_sample, int(total * fraction)))


def _compute_extraction_confidence(extractions: list[dict]) -> dict:
    """Aggregate extraction-level confidence metrics.

    Computes mean/median/p25 confidence across all extractions in the
    example set.  The pipeline uses mean_confidence vs the gate threshold
    to decide whether to proceed with full-scale extraction.

    Args:
        extractions: List of extraction dicts, each with a 'confidence' key.

    Returns:
        Dict with mean_confidence, median_confidence, p25_confidence,
        n_extractions, and n_high (confidence >= 0.7).
    """
    if not extractions:
        return {"mean_confidence": 0.0, "median_confidence": 0.0,
                "p25_confidence": 0.0, "n_extractions": 0, "n_high": 0}
    confs = [e.get("confidence", 0.0) for e in extractions]
    arr = np.array(confs)
    return {
        "mean_confidence": float(np.mean(arr)),
        "median_confidence": float(np.median(arr)),
        "p25_confidence": float(np.percentile(arr, 25)),
        "n_extractions": len(confs),
        "n_high": int(np.sum(arr >= 0.7)),
    }


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class MinimalPipelineConfig:
    """Tunables for the minimal pipeline."""

    # Clustering
    device: str = "cuda"
    use_optuna: bool = True
    optuna_trials: int = 30
    evoc_noise_level: float = 0.3

    # Chunking (3 chunk types)
    min_chunk_chars: int = 100
    max_chunk_chars: int = 2000                # ~2000 tok embedding chunks
    extraction_chunk_chars: int = 16000        # ~4000 tok extraction chunks
    max_chunks_per_doc: int = 80

    # LLM concurrency
    max_llm_concurrency: int = 8

    # Discovery (rlm_v3 only, standard fallback)
    discovery_mode: str = "rlm_v3_unified"     # rlm_v3 | rlm_v3_unified
    max_sample_chunks: int = 30
    max_fields_per_cluster: int = 15
    min_cluster_chunks: int = 5
    discovery_temperature: float = 0.2

    # Extraction (RAG only)
    min_extraction_confidence: float = 0.5
    extract_values: bool = True

    # Optional stages
    run_intents: bool = False
    min_intent_clause_chars: int = 80
    intent_match_threshold: float = 0.85

    # Paths
    db_path: str | None = None
    lance_path: str | None = None
    artifacts_dir: str | None = None
    max_docs: int | None = None

    # Legacy field features (apr-fields-processor)
    legacy_fields_dir: str | None = None   # path to apr-fields-processor root; auto-detected if None
    enable_field_features: bool = True      # run pattern/keyword/type extraction on chunks

    # Guided clustering: user-specified keyword groups steer cluster formation
    # E.g. {"termination": ["termination", "cancellation", "expiry"],
    #        "payment": ["payment", "fees", "compensation", "invoice"]}
    guided_keywords: dict[str, list[str]] | None = None
    guided_alpha: float = 0.3              # 0=no guidance, 1=full pull (0.2-0.4 recommended)

    # ── Feedback & Personalization ──
    enable_feedback: bool = True            # wire FeedbackStore + PersonalizationEngine
    tenant_id: str | None = None            # sidecar DB tenant; defaults to account_id
    feedback_data_dir: str | None = None    # dir for sidecar DuckDB; defaults to data/

    # ── Confidence-gated scaling ──
    enable_confidence_gate: bool = True     # extract on Cochran sample first, gate before scaling
    confidence_gate_threshold: float = 0.65 # min avg confidence to proceed with full extraction
    cochran_sample_fraction: float = 0.3    # fraction of cluster chunks for example-set extraction
    cochran_min_sample: int = 5             # minimum example-set size
    cochran_max_sample: int = 30            # maximum example-set size

    # ── Probabilistic routing ──
    enable_soft_membership: bool = True     # store soft (probabilistic) cluster memberships
    soft_membership_k: int = 3              # top-K clusters for soft membership per chunk
    low_confidence_threshold: float = 0.40  # below this → route to fallback / human review

    # ── Multi-dimensional signal aggregation ──
    enable_multidim_signals: bool = False   # use field-overlap + obligation similarity in clustering
    field_overlap_weight: float = 0.15      # weight for field-overlap dimension
    obligation_weight: float = 0.10         # weight for obligation-pattern dimension


# ── Async LLM Client ─────────────────────────────────────────────────────────


class AsyncLLMClient:
    """Rate-limited async wrapper around Azure OpenAI with JSON-mode responses.

    Uses a semaphore to limit concurrent LLM calls and a ThreadPoolExecutor
    to run synchronous OpenAI SDK calls in an async context.  Tracks total
    call count and latency for pipeline stats.
    """

    def __init__(self, max_concurrency: int = 8):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency)
        self._client = None
        self._deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
        self._call_count = 0
        self._total_latency = 0.0

    @property
    def client(self):
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )
        return self._client

    async def complete_json(
        self, prompt: str, temperature: float = 0.2,
        max_tokens: int = 2000, retries: int = 3, model: str | None = None,
    ) -> dict | None:
        deployment = model or self._deployment
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            for attempt in range(retries):
                try:
                    t0 = time.time()
                    result = await loop.run_in_executor(
                        self._executor,
                        partial(self._sync_call, prompt, temperature, max_tokens, deployment),
                    )
                    self._call_count += 1
                    self._total_latency += time.time() - t0
                    return result
                except json.JSONDecodeError:
                    logger.warning(f"JSON parse error (attempt {attempt + 1})")
                except Exception as e:
                    logger.warning(f"LLM call error (attempt {attempt + 1}): {e}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1.5 * (attempt + 1))
        return None

    def _sync_call(self, prompt: str, temperature: float, max_tokens: int, deployment: str) -> dict:
        response = self.client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._call_count,
            "total_latency_s": round(self._total_latency, 1),
            "avg_latency_s": round(self._total_latency / max(1, self._call_count), 2),
        }


# ── Prompts ───────────────────────────────────────────────────────────────────

STANDARD_DISCOVERY_PROMPT = """You are a legal contract analyst. Analyze these clause excerpts that all belong to the same clause type: "{cluster_label}".

Keywords: {keywords}

CLAUSE EXCERPTS:
{chunks_text}

Identify the recurring **fields** (structured data points) across these clauses. For each field provide:
1. A snake_case field name (max 40 chars)
2. Data type: text, date, duration, currency, percentage, integer, boolean, entity_name, or enum
3. One-sentence description
4. 2-3 example values from the excerpts

Rules:
- Only include fields appearing in at least 2 excerpts
- Return between 2 and {max_fields} fields
- Focus on extractable structured data, not free text
- Prefer specific fields (e.g., "renewal_term" over "terms")

Respond with ONLY JSON: {{"fields": [...]}}
Each field: {{"name": "...", "type": "...", "description": "...", "examples": ["...", "..."]}}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

import re

VALID_FIELD_TYPES = {"text", "date", "duration", "currency", "percentage",
                     "integer", "boolean", "entity_name", "enum"}


def _normalize_field(f: dict, max_name_len: int = 40) -> dict | None:
    """Normalize a raw field dict from LLM output into a canonical schema.

    Cleans the field name to snake_case, maps type aliases (e.g. 'money' → 'currency'),
    validates the type against VALID_FIELD_TYPES, and structures examples.
    Returns None if the field has no usable name.
    """
    name = (f.get("name") or f.get("field_name") or f.get("field") or "").strip()
    if not name:
        return None
    name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:max_name_len]
    ftype = f.get("type", f.get("value_type", "text")).strip().lower()
    type_map = {"string": "text", "money": "currency", "ref": "text"}
    ftype = type_map.get(ftype, ftype)
    if ftype not in VALID_FIELD_TYPES:
        ftype = "text"
    return {
        "name": name, "type": ftype,
        "description": f.get("description", ""),
        "examples": f.get("examples") or ([f.get("span_text")] if f.get("span_text") else []),
        "source": f.get("source", "standard"),
    }


# ── Discovery ─────────────────────────────────────────────────────────────────


async def _discover_fields_standard(
    llm: AsyncLLMClient, cluster_label: str, keywords: list[str],
    chunk_texts: list[str], config: MinimalPipelineConfig,
) -> list[dict]:
    """Single-shot LLM field discovery — fast fallback when RLM V3 fails.

    Sends a sample of up to max_sample_chunks excerpts to the LLM with a
    structured prompt asking it to identify recurring fields.  Returns
    normalized field dicts.  Used as fallback when RLM V3 errors out or
    discovers no fields.
    """
    sample = chunk_texts[:config.max_sample_chunks]
    chunks_text = "\n\n".join(f"[Excerpt {i+1}]\n{text[:1500]}" for i, text in enumerate(sample))
    prompt = STANDARD_DISCOVERY_PROMPT.format(
        cluster_label=cluster_label,
        keywords=", ".join(keywords[:10]),
        chunks_text=chunks_text,
        max_fields=config.max_fields_per_cluster,
    )
    parsed = await llm.complete_json(prompt, temperature=config.discovery_temperature)
    if not parsed:
        return []
    raw_fields = parsed.get("fields", parsed) if isinstance(parsed, dict) else parsed
    if not isinstance(raw_fields, list):
        return []
    return [f for f in (_normalize_field(f) for f in raw_fields[:config.max_fields_per_cluster]) if f]


async def _discover_fields_rlm_v3(
    cluster_label: str, keywords: list[str], chunk_texts: list[str],
    config: MinimalPipelineConfig,
) -> tuple[list[dict], dict]:
    """RLM V3 discovery-only mode — agentic REPL with 12 tools.

    Runs the RLM V3 agent (structured state machine + dual-mode workers)
    on a sample of cluster chunks.  The agent iteratively discovers fields
    using REPL tools (grep, regex, summarize, etc.).  Returns (fields, meta)
    where meta contains clause_types, obligations, and strategy info.
    """
    loop = asyncio.get_event_loop()

    def _run_sync():
        from core.rlm_v3 import RLMV3, RLMV3Config
        v3_config = RLMV3Config(
            max_iterations=int(os.getenv("RLM_MAX_ITERATIONS", "25")),
            max_llm_calls=int(os.getenv("RLM_MAX_LLM_CALLS", "50")),
            max_sample_chunks=config.max_sample_chunks,
            max_fields=config.max_fields_per_cluster,
        )
        v3 = RLMV3(config=v3_config)
        return v3.discover(cluster_label, keywords, chunk_texts)

    try:
        return await loop.run_in_executor(None, _run_sync)
    except Exception as e:
        logger.warning(f"RLM V3 discovery failed for '{cluster_label}': {e}")
        return [], {"error": str(e)}


async def _discover_and_extract_rlm_v3(
    cluster_label: str, keywords: list[str], chunk_texts: list[str],
    chunk_ids: list[str], agreement_ids: list[str],
    config: MinimalPipelineConfig,
) -> tuple[list[dict], list[dict], dict]:
    """RLM V3 unified mode — discover fields AND extract values in one REPL pass.

    Combines discovery and extraction into a single agent session, producing
    'free' inline extractions as a byproduct of discovery.  Returns
    (fields, inline_extractions, meta).  The inline_extractions are persisted
    without costing additional LLM calls.
    """
    loop = asyncio.get_event_loop()

    def _run_sync():
        from core.rlm_v3 import RLMV3, RLMV3Config
        v3_config = RLMV3Config(
            max_iterations=int(os.getenv("RLM_MAX_ITERATIONS", "25")),
            max_llm_calls=int(os.getenv("RLM_MAX_LLM_CALLS", "50")),
            max_sample_chunks=config.max_sample_chunks,
            max_fields=config.max_fields_per_cluster,
            extract_inline=True,
            min_extraction_confidence=config.min_extraction_confidence,
        )
        v3 = RLMV3(config=v3_config)
        return v3.discover_and_extract(
            cluster_label, keywords, chunk_texts,
            chunk_ids=chunk_ids, agreement_ids=agreement_ids,
        )

    try:
        return await loop.run_in_executor(None, _run_sync)
    except Exception as e:
        logger.warning(f"RLM V3 unified failed for '{cluster_label}': {e}")
        return [], [], {"error": str(e)}


# ── Cluster Processing ────────────────────────────────────────────────────────


async def _process_cluster(
    cluster: dict,
    chunk_rows: list[dict],
    store,
    llm: AsyncLLMClient,
    config: MinimalPipelineConfig,
    cluster_idx: int,
    total_clusters: int,
    progress,
    shared_embedder=None,
    shared_rag_config=None,
    personalization_ctx=None,
) -> dict:
    """Process a single cluster through the full discovery → extraction lifecycle.

    Implements a 3-phase approach:
      Phase 1 (Discovery):  RLM V3 agentic field discovery on a diversity-
                             sampled subset.  Falls back to single-shot LLM
                             if RLM fails.
      Phase 2 (Example-set): If confidence gating is enabled, extract on a
                              small Cochran sample first and measure aggregate
                              confidence.  If below threshold, skip full-scale
                              extraction (saves LLM cost on low-quality clusters).
      Phase 3 (Production):  Agentic RAG extraction on remaining chunks.
                              Incorporates personalization context (tenant
                              threshold overrides, few-shot correction examples)
                              when available.

    Args:
        cluster:           Cluster metadata dict from DB.
        chunk_rows:        All chunk rows assigned to this cluster.
        store:             ClusteringStore for persistence.
        llm:               AsyncLLMClient instance.
        config:            Pipeline configuration.
        cluster_idx:       Index of this cluster (for progress logging).
        total_clusters:    Total number of clusters being processed.
        progress:          Progress callback(stage, message).
        shared_embedder:   Pre-created embedder singleton (avoids re-init).
        shared_rag_config: Pre-created RAGConfig (avoids re-init).
        personalization_ctx: Optional PersonalizationContext from feedback loop.

    Returns:
        Dict with cluster_id, label, fields, extractions, confidence_gate,
        and profile (ClusterProfile).
    """
    cluster_id = cluster["cluster_id"]
    cluster_label = cluster["label"] or "Unknown"
    try:
        keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(
            cluster.get("keywords"), str) else (cluster.get("keywords") or [])
    except json.JSONDecodeError:
        keywords = []

    # Apply personalization threshold overrides if available
    min_confidence = config.min_extraction_confidence
    if personalization_ctx and personalization_ctx.tentative_threshold:
        min_confidence = max(min_confidence, personalization_ctx.tentative_threshold)

    # Use diversity-sampled chunks for RLM to see variety across agreements
    sampled_rows = _diversity_sample_chunks(chunk_rows, config.max_sample_chunks)
    chunk_texts = [r["chunk_text"] for r in sampled_rows]
    all_chunk_texts = [r["chunk_text"] for r in chunk_rows]
    stats = {"cluster_id": cluster_id, "label": cluster_label, "fields": 0,
             "extractions": 0, "confidence_gate": {}, "profile": None}
    progress("discovery", f"  [{cluster_idx+1}/{total_clusters}] {cluster_label} "
             f"({len(chunk_rows)} chunks, {len(sampled_rows)} sampled)")

    # ── Phase 1: Discover fields (+ inline extractions for unified mode) ──
    rlm_meta = None
    inline_extractions: list[dict] = []
    already_extracted_indices: set[int] = set()

    if config.discovery_mode == "rlm_v3_unified":
        chunk_ids = [r["chunk_id"] for r in sampled_rows]
        agr_ids = [r["agreement_id"] for r in sampled_rows]
        fields, inline_extractions, rlm_meta = await _discover_and_extract_rlm_v3(
            cluster_label, keywords, chunk_texts, chunk_ids, agr_ids, config)
        if rlm_meta:
            extracted_sample_indices = set(range(len(sampled_rows))) - set(
                rlm_meta.get("remaining_chunk_indices", []))
            sampled_row_ids = {r["chunk_id"] for i, r in enumerate(sampled_rows)
                               if i in extracted_sample_indices}
            already_extracted_indices = {i for i, r in enumerate(chunk_rows)
                                         if r["chunk_id"] in sampled_row_ids}
        if not fields:
            fields = await _discover_fields_standard(llm, cluster_label, keywords,
                                                     all_chunk_texts, config)
    else:
        fields, rlm_meta = await _discover_fields_rlm_v3(
            cluster_label, keywords, chunk_texts, config)
        if not fields:
            fields = await _discover_fields_standard(llm, cluster_label, keywords,
                                                     all_chunk_texts, config)

    if not fields:
        progress("discovery", f"    No fields discovered")
        return stats

    # Persist field definitions
    field_id_map = {}
    for f in fields:
        field_id = f"fld_{uuid.uuid4().hex[:10]}"
        field_id_map[f["name"]] = field_id
        store.upsert_field(
            field_id=field_id, cluster_id=cluster_id,
            name=f["name"], field_type=f["type"],
            description=f["description"], source=f.get("source", "auto_discovered"),
            examples=f.get("examples", []),
        )
    stats["fields"] = len(fields)

    strategy = rlm_meta.get("strategy", config.discovery_mode) if rlm_meta else config.discovery_mode
    progress("discovery", f"    {len(fields)} fields ({strategy}): {[f['name'] for f in fields]}")

    if not config.extract_values:
        # Build profile even without extraction
        stats["profile"] = _build_cluster_profile(
            cluster, fields, keywords, rlm_meta, stats, {})
        return stats

    cluster_extractions = 0

    # ── Persist inline extractions from unified discovery (free extractions) ──
    for ext in inline_extractions:
        field_id = field_id_map.get(ext["field_name"])
        if not field_id:
            continue
        store.insert_extraction(
            extraction_id=f"ext_{uuid.uuid4().hex[:10]}",
            agreement_id=ext.get("agreement_id", ""),
            field_id=field_id,
            value=ext["value"],
            confidence=ext.get("confidence", 0.5),
            source_chunk_id=ext.get("chunk_id", ""),
        )
        cluster_extractions += 1

    if inline_extractions:
        progress("extraction", f"    {cluster_extractions} free inline extractions from discovery")

    remaining_rows = [r for i, r in enumerate(chunk_rows) if i not in already_extracted_indices]
    if not remaining_rows:
        stats["extractions"] = cluster_extractions
        stats["profile"] = _build_cluster_profile(
            cluster, fields, keywords, rlm_meta, stats, {})
        progress("extraction", f"    {cluster_extractions} total (all inline)")
        return stats

    # Build context from RLM meta
    context_section = _build_rlm_context_section(rlm_meta)

    # Inject personalization few-shot corrections into context if available
    if personalization_ctx and personalization_ctx.correction_examples:
        correction_lines = ["\n## Tenant Corrections (apply these naming preferences):"]
        for ex in personalization_ctx.correction_examples[:5]:
            old_l = ex.get("old_label", "")
            new_l = ex.get("new_label", "")
            if old_l and new_l:
                correction_lines.append(f"- Instead of \"{old_l}\", use \"{new_l}\"")
        context_section += "\n".join(correction_lines) + "\n"

    # ── Phase 2: Confidence-gated example-set extraction ──────────────────
    gate_result: dict[str, Any] = {"enabled": config.enable_confidence_gate}

    if config.enable_confidence_gate and len(remaining_rows) > config.cochran_min_sample * 2:
        example_size = _cochran_sample_size(
            len(remaining_rows),
            config.cochran_sample_fraction,
            config.cochran_min_sample,
            config.cochran_max_sample,
        )
        example_rows = _diversity_sample_chunks(remaining_rows, example_size)
        progress("confidence_gate",
                 f"    Confidence gate: extracting {len(example_rows)}/{len(remaining_rows)} example set")

        # Run RAG on example set
        example_extractions = await _run_rag_extraction(
            store, llm, config, cluster_id, fields, cluster_label,
            keywords, context_section, shared_embedder, shared_rag_config,
        )
        gate_metrics = _compute_extraction_confidence(example_extractions)
        gate_result.update(gate_metrics)
        gate_result["example_size"] = len(example_rows)
        gate_result["threshold"] = config.confidence_gate_threshold
        gate_result["passed"] = gate_metrics["mean_confidence"] >= config.confidence_gate_threshold

        if not gate_result["passed"]:
            # Gate failed: skip full-scale extraction to save LLM cost
            progress("confidence_gate",
                     f"    GATE FAILED: mean_conf={gate_metrics['mean_confidence']:.2f} "
                     f"< threshold={config.confidence_gate_threshold:.2f} — skipping full extraction")
            stats["extractions"] = cluster_extractions
            stats["confidence_gate"] = gate_result
            stats["profile"] = _build_cluster_profile(
                cluster, fields, keywords, rlm_meta, stats, gate_result)
            return stats

        progress("confidence_gate",
                 f"    GATE PASSED: mean_conf={gate_metrics['mean_confidence']:.2f} — "
                 f"proceeding with full extraction")

    stats["confidence_gate"] = gate_result

    # ── Phase 3: Production RAG extraction on remaining chunks ────────────
    from core.rag_extractor import RAGExtractor, RAGConfig

    embedder = shared_embedder or _get_embedder(config)
    rag_config = shared_rag_config or RAGConfig(
        max_sample_docs=int(os.getenv("RAG_MAX_SAMPLE_DOCS", "10")),
        max_retry_rounds=int(os.getenv("RAG_MAX_RETRY_ROUNDS", "3")),
        enable_reviewer=os.getenv("RAG_ENABLE_REVIEWER", "1") == "1",
        min_extraction_confidence=min_confidence,
    )
    rag_extractor = RAGExtractor(
        store=store, llm=llm, embed_fn=embedder.embed, config=rag_config,
    )
    try:
        batch = await rag_extractor.extract_cluster(
            cluster_id=cluster_id,
            fields=fields,
            cluster_label=cluster_label,
            cluster_keywords=keywords,
            context_section=context_section,
        )
        rag_extractions = batch.meta.get("n_persisted", 0)
        cluster_extractions += rag_extractions
        stats["extractions"] = cluster_extractions
        stats["rag_meta"] = batch.meta
        n_inline = len(inline_extractions)
        progress(
            "extraction",
            f"    {cluster_extractions} extractions "
            f"(RAG: {batch.meta.get('n_agreements_sampled', 0)} docs, "
            f"{batch.meta.get('total_retries', 0)} retries"
            f"{f', {n_inline} inline' if n_inline else ''})"
        )
    except Exception as e:
        logger.warning(f"RAG extraction failed for '{cluster_label}': {e}")
        stats["extractions"] = cluster_extractions

    # Build and return cluster profile
    stats["profile"] = _build_cluster_profile(
        cluster, fields, keywords, rlm_meta, stats, gate_result)

    return stats


def _build_rlm_context_section(rlm_meta: dict | None) -> str:
    """Build a context string from RLM V3 discovery metadata.

    Extracts clause sub-types and key obligations from the RLM meta dict
    and formats them into a text section that can be prepended to RAG
    extraction prompts for better field-context.

    Args:
        rlm_meta: RLM V3 discovery metadata dict (may be None).

    Returns:
        Formatted context string, or empty string if no useful meta.
    """
    if not rlm_meta:
        return ""
    parts = []
    clause_types = rlm_meta.get("clause_types", [])
    if clause_types:
        ct_str = "; ".join(
            f"{ct.get('name', '')}: {ct.get('emergent_description', '')}"
            for ct in clause_types[:3]
        )
        parts.append(f"Clause sub-types: {ct_str}")
    obligations = rlm_meta.get("obligations", [])
    if obligations:
        ob_str = "; ".join(
            f"{ob.get('obligor', '')} must {ob.get('action', '')}"
            for ob in obligations[:3]
        )
        parts.append(f"Key obligations: {ob_str}")
    return ("\n".join(parts) + "\n") if parts else ""


async def _run_rag_extraction(
    store, llm, config, cluster_id, fields, cluster_label,
    keywords, context_section, shared_embedder, shared_rag_config,
) -> list[dict]:
    """Run RAG extraction and return the raw extraction dicts (for confidence measurement).

    This is a thin wrapper around RAGExtractor.extract_cluster() that returns
    the list of persisted extractions so the confidence gate can measure
    aggregate quality before committing to full-scale extraction.

    Returns:
        List of extraction dicts with at least a 'confidence' key each.
    """
    from core.rag_extractor import RAGExtractor, RAGConfig

    embedder = shared_embedder or _get_embedder(config)
    rag_config = shared_rag_config or RAGConfig(
        max_sample_docs=int(os.getenv("RAG_MAX_SAMPLE_DOCS", "10")),
        max_retry_rounds=int(os.getenv("RAG_MAX_RETRY_ROUNDS", "3")),
        enable_reviewer=os.getenv("RAG_ENABLE_REVIEWER", "1") == "1",
        min_extraction_confidence=config.min_extraction_confidence,
    )
    rag = RAGExtractor(store=store, llm=llm, embed_fn=embedder.embed, config=rag_config)
    try:
        batch = await rag.extract_cluster(
            cluster_id=cluster_id, fields=fields,
            cluster_label=cluster_label, cluster_keywords=keywords,
            context_section=context_section,
        )
        return batch.extractions if hasattr(batch, "extractions") else []
    except Exception as e:
        logger.warning(f"Example-set RAG extraction failed: {e}")
        return []


def _build_cluster_profile(
    cluster: dict,
    fields: list[dict],
    keywords: list[str],
    rlm_meta: dict | None,
    stats: dict,
    gate_result: dict,
) -> ClusterProfile:
    """Assemble a ClusterProfile from discovery + extraction results.

    Bundles everything the pipeline knows about this cluster into a single
    structured object: field schemas, BM25 keywords, clause sub-types,
    obligations, extraction quality metrics, and confidence-gate outcome.
    This profile is the 'agreement genome' — downstream consumers use it
    for incremental assignment, feedback routing, and API responses.

    Args:
        cluster:     Cluster metadata dict from DB.
        fields:      Discovered field definitions.
        keywords:    BM25-weighted thematic keywords.
        rlm_meta:    RLM V3 discovery metadata (clause_types, obligations).
        stats:       Current extraction stats dict.
        gate_result: Confidence gate outcome dict.

    Returns:
        Populated ClusterProfile instance.
    """
    return ClusterProfile(
        cluster_id=cluster.get("cluster_id", ""),
        domain_id=cluster.get("domain_id", ""),
        label=cluster.get("label", ""),
        description=cluster.get("description", ""),
        keywords=keywords,
        field_definitions=fields,
        clause_types=rlm_meta.get("clause_types", []) if rlm_meta else [],
        obligations=rlm_meta.get("obligations", []) if rlm_meta else [],
        chunk_count=cluster.get("chunk_count", 0),
        agreement_count=cluster.get("agreement_count", 0),
        extraction_stats={
            "fields": stats.get("fields", 0),
            "extractions": stats.get("extractions", 0),
        },
        confidence_gate=gate_result,
    )


def _get_embedder(config: MinimalPipelineConfig):
    """Return the shared Embedder singleton for RAG and assignment operations."""
    from core.embedder import Embedder
    return Embedder.get_instance()


# ── Intent Discovery (optional Stage 5) ──────────────────────────────────────


INTENT_PROMPT = """You are a legal contract analyst. Analyze this clause and identify every distinct legal intent it contains.

A "legal intent" is a single obligation, right, prohibition, condition, or declaration that has legal effect.

Clause type: "{clause_type}"
Clause text:
{clause_text}

For each intent found, provide:
- label: a short snake_case name
- summary: one sentence describing what this intent does
- party_from: who bears the obligation/restriction (or "none" if declarative)
- party_to: who benefits (or "none" if declarative)
- attributes: a dict of intent-specific properties stated in the text

If the clause has NO legal intent, return an empty array.

Respond with ONLY a JSON object:
{{"intents": [
  {{"label": "...", "summary": "...", "party_from": "...", "party_to": "...", "attributes": {{...}}}},
]}}"""


async def _process_intents(store, llm: AsyncLLMClient, config: MinimalPipelineConfig, progress):
    """Discover legal intents for all clustered clauses (optional Stage 5).

    For each clause not yet processed, sends the clause text to the LLM
    to identify obligations, rights, prohibitions, conditions, and declarations.
    Intents are matched against an IntentRegistry to build canonical types.
    """
    from core.intent_extractor import IntentRegistry, _normalize_intent

    registry = IntentRegistry(store)
    clauses = store.conn.execute("""
        SELECT cl.clause_id, cl.agreement_id, cl.clause_type_id,
               cl.full_text, c.label as clause_type_label
        FROM clauses cl
        JOIN clusters c ON cl.clause_type_id = c.cluster_id
        WHERE cl.clause_type_id IS NOT NULL
          AND cl.clause_id NOT IN (SELECT DISTINCT clause_id FROM clause_intents)
          AND LENGTH(cl.full_text) >= ?
        ORDER BY c.label, cl.clause_id
    """, [config.min_intent_clause_chars]).fetchdf().to_dict("records")

    if not clauses:
        progress("intent", "No new clauses for intent extraction")
        return {"total_clauses": 0, "total_intents": 0}

    progress("intent", f"Extracting intents from {len(clauses)} clauses...")
    total_intents = 0

    for ci, clause in enumerate(clauses):
        clause_type = clause.get("clause_type_label", "Unknown")
        prompt = INTENT_PROMPT.format(
            clause_type=clause_type,
            clause_text=clause["full_text"][:3000],
        )
        parsed = await llm.complete_json(prompt, temperature=0.0, max_tokens=1500)
        if not parsed or not isinstance(parsed, dict):
            continue

        raw_intents = parsed.get("intents", [])
        if not isinstance(raw_intents, list):
            continue

        for raw in raw_intents:
            if not isinstance(raw, dict):
                continue
            label = raw.get("label", "").strip()
            if not label:
                continue
            label = label.lower().replace(" ", "_").replace("-", "_")
            attrs = raw.get("attributes", {})
            if not isinstance(attrs, dict):
                attrs = {}

            intent_type_id = registry.match_or_register(label, raw.get("summary", ""))
            intent_id = f"int_{uuid.uuid4().hex[:10]}"
            store.insert_clause_intent(
                intent_id=intent_id,
                clause_id=clause["clause_id"],
                intent_type_id=intent_type_id,
                summary=str(raw.get("summary", "")).strip(),
                party_from=str(raw.get("party_from", "none")).strip(),
                party_to=str(raw.get("party_to", "none")).strip(),
                attributes=attrs,
            )
            total_intents += 1

        if (ci + 1) % 50 == 0:
            progress("intent", f"  {ci+1}/{len(clauses)} clauses, {total_intents} intents")

    progress("intent", f"  {total_intents} intents from {len(clauses)} clauses")
    return {"total_clauses": len(clauses), "total_intents": total_intents}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


class _StageTimer:
    """Context manager that records elapsed time for a named pipeline stage."""
    def __init__(self, name: str, timings: dict):
        self.name = name
        self.timings = timings
    def __enter__(self):
        self._t0 = time.time()
        return self
    def __exit__(self, *args):
        self.timings[self.name] = round(time.time() - self._t0, 2)


async def run_minimal_pipeline(
    pdf_dir: str | Path,
    account_id: str = "default",
    config: MinimalPipelineConfig | None = None,
    callback=None,
) -> dict:
    """Run the full minimal pipeline: parse → chunk+embed → cluster → discover → extract.

    7-stage pipeline (5 core + 2 optional):
      Stage 1: Parse PDFs from directory (with caching).
      Stage 2: Chunk into 3 types (embedding, extraction, clause) + Nomic embed
               + optional field feature extraction from apr-fields-processor.
      Stage 3: Parallel EVoC clustering — macro on summaries, micro on chunks.
               Includes guided clustering, multi-dimensional signals, and
               soft (probabilistic) membership computation.
      Stage 4: RLM V3 discovery + confidence-gated RAG extraction with
               personalization context from FeedbackStore.
               Builds ClusterProfile per cluster.
      Stage 5: [Optional] Intent discovery on clause texts.
      Stage 6: Wire IncrementalClusterAssigner with cluster centroids for
               future single-document ingestion.
      Post:    Persist cluster profiles, assigner state, and pipeline manifest.

    Args:
        pdf_dir:    Path to directory containing PDF files.
        account_id: Tenant/account identifier for DB partitioning.
        config:     Pipeline configuration (defaults to MinimalPipelineConfig).
        callback:   Optional progress callback(stage: str, message: str).

    Returns:
        Dict with run statistics, timing, cluster profiles, and error info.
    """
    config = config or MinimalPipelineConfig()
    t0 = time.time()
    stage_timings: dict[str, float] = {}

    def progress(stage: str, msg: str):
        elapsed = time.time() - t0
        logger.info(f"[{elapsed:6.1f}s] [{stage}] {msg}")
        if callback:
            callback(stage, msg)

    def _time(name):
        return _StageTimer(name, stage_timings)

    # ── Paths ──
    from config import DATA_DIR
    artifacts_root = Path(config.artifacts_dir) if config.artifacts_dir else DATA_DIR / "artifacts"
    db = Path(config.db_path) if config.db_path else DATA_DIR / f"{account_id}.duckdb"
    lance = Path(config.lance_path) if config.lance_path else DATA_DIR / f"{account_id}.lance"

    from core.store import ClusteringStore
    from core.lancedb_store import LanceVectorStore
    from core.nomic_embedder import NomicEmbedder

    store = ClusteringStore(db)
    lance_store = LanceVectorStore(lance)
    llm = AsyncLLMClient(max_concurrency=config.max_llm_concurrency)

    # ── Wire FeedbackStore + PersonalizationEngine ──
    # Creates a sidecar DuckDB per tenant that stores feedback events,
    # personalization weights, virtual clusters, and intent overrides.
    # The PersonalizationEngine reads this sidecar and produces a
    # PersonalizationContext for each cluster's extraction phase.
    feedback_store = None
    personalization_engine = None
    personalization_ctx = None
    if config.enable_feedback:
        try:
            from core.feedback_store import FeedbackStore
            from core.personalization import PersonalizationEngine
            tenant = config.tenant_id or account_id
            fb_dir = config.feedback_data_dir or str(DATA_DIR)
            feedback_store = FeedbackStore(tenant_id=tenant, data_dir=fb_dir)
            personalization_engine = PersonalizationEngine(
                feedback_store, main_db_path=str(db))
            personalization_ctx = personalization_engine.get_context()
            progress("init", f"Feedback loop active: tenant={tenant}, "
                     f"events={feedback_store.summary().get('feedback_events', 0)}")
        except Exception as e:
            logger.warning(f"Feedback/personalization init failed (non-fatal): {e}")
            progress("init", f"Feedback loop skipped: {e}")

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    store.start_run(run_id, "init")
    progress("init", f"Minimal pipeline: discovery={config.discovery_mode}, "
             f"account={account_id}, device={config.device}")

    pipeline_stats: dict[str, Any] = {
        "run_id": run_id, "account_id": account_id,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_")},
    }

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Parse PDFs
    # ══════════════════════════════════════════════════════════════════════
    with _time("parse"):
        from core.pdf_parser import parse_pdf_directory, load_parsed_agreements, PARSED_TEXT_DIR

        cached = load_parsed_agreements()
        pdf_dir_resolved = str(Path(pdf_dir).resolve())
        manifest_path = PARSED_TEXT_DIR / "manifest.json"
        cache_hit = False
        if cached and manifest_path.exists():
            with open(manifest_path) as _f:
                _manifest = json.load(_f)
            if str(Path(_manifest.get("source_dir", "")).resolve()) == pdf_dir_resolved:
                cache_hit = True

        if cache_hit:
            progress("parse", f"Using {len(cached)} cached PDFs")
            parsed_docs = cached
        else:
            progress("parse", f"Parsing PDFs from {pdf_dir}...")
            parsed_docs = parse_pdf_directory(pdf_dir)

        if config.max_docs and len(parsed_docs) > config.max_docs:
            parsed_docs = parsed_docs[:config.max_docs]

        agreements = [
            {"agreement_id": d["agreement_id"], "filename": d["filename"], "raw_text": d["raw_text"]}
            for d in parsed_docs if len(d.get("raw_text", "")) > 100
        ]
        progress("parse", f"{len(agreements)} valid PDFs")

    if not agreements:
        return {"error": "no_documents", **pipeline_stats}

    pipeline_stats["n_documents"] = len(agreements)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Chunk (3 types) + Embed
    # ══════════════════════════════════════════════════════════════════════
    with _time("chunk_embed"):
        embedder = NomicEmbedder.get_instance(device=config.device)
        embedder.warmup()

        from core.chunker import (
            SemanticChunkerConfig, extract_clause_header,
            _approx_token_count, generate_typed_chunks,
        )

        progress("chunk", "Generating 3 chunk types (embedding, extraction, clause)...")
        chunker_config = SemanticChunkerConfig(
            buffer_size=1, breakpoint_percentile=90.0,
            min_chunk_chars=config.min_chunk_chars,
            max_chunk_chars=config.max_chunk_chars,
            extraction_max_chunk_chars=config.extraction_chunk_chars,
            clause_split=True,
        )
        _embed_fn = lambda texts: embedder.embed_for_clustering(texts, batch_size=8)

        all_embedding_chunks = []     # chunk_type="embedding" — for clustering + RAG
        all_extraction_chunks = []    # chunk_type="extraction" — larger, for sequential extraction (future)
        all_clauses = []              # clause records with metadata

        for agr in agreements:
            typed = generate_typed_chunks(agr["raw_text"], embed_fn=_embed_fn, config=chunker_config)

            # Embedding chunks
            for c in typed["embedding"]:
                c["agreement_id"] = agr["agreement_id"]
                c["clause_id"] = None
            all_embedding_chunks.extend(typed["embedding"])

            # Extraction chunks (stored for future sequential extraction path)
            for c in typed["extraction"]:
                c["agreement_id"] = agr["agreement_id"]
            all_extraction_chunks.extend(typed["extraction"])

            # Clause chunks → clause records
            for ci, cc in enumerate(typed["clause"]):
                cc["agreement_id"] = agr["agreement_id"]
                header = extract_clause_header(cc["text"])
                clause_id = f"cls_{uuid.uuid4().hex[:10]}"
                cc["clause_id"] = clause_id

                # Link embedding chunks that overlap this clause
                linked_ids = []
                for ec in typed["embedding"]:
                    if ec["char_start"] < cc["char_end"] and ec["char_end"] > cc["char_start"]:
                        linked_ids.append(ec["chunk_id"])
                        if ec.get("clause_id") is None:
                            ec["clause_id"] = clause_id

                all_clauses.append({
                    "clause_id": clause_id,
                    "agreement_id": agr["agreement_id"],
                    "clause_number": header["clause_number"] if header else "",
                    "clause_title": header["clause_title"] if header else "",
                    "heading_level": header["heading_level"] if header else 0,
                    "clause_index": ci,
                    "full_text": cc["text"],
                    "token_count": cc["token_count"],
                    "chunk_ids": linked_ids,
                })

            # Cap per doc: remove excess from THIS doc only (not from global list)
            doc_count = len(typed["embedding"])
            if doc_count > config.max_chunks_per_doc:
                excess = doc_count - config.max_chunks_per_doc
                all_embedding_chunks = all_embedding_chunks[:-excess]

        progress("chunk", f"{len(all_embedding_chunks)} embedding, "
                 f"{len(all_extraction_chunks)} extraction, "
                 f"{len(all_clauses)} clause chunks from {len(agreements)} docs")

        # ── Field Feature Extraction (pattern + keyword + type) ──
        field_feature_extractor = None
        doc_feature_summaries: dict[str, str] = {}   # agreement_id → feature summary line

        if config.enable_field_features:
            try:
                from core.field_features import FieldFeatureExtractor
                field_feature_extractor = FieldFeatureExtractor.from_legacy_dir(
                    config.legacy_fields_dir)
                progress("field_features",
                         f"Loaded {field_feature_extractor.total_count} field configs "
                         f"({field_feature_extractor.predefined_count} predefined, "
                         f"{field_feature_extractor.business_count} business)")

                # Run feature extraction per document on embedding chunks
                chunks_by_agr: dict[str, list[dict]] = {}
                for c in all_embedding_chunks:
                    chunks_by_agr.setdefault(c["agreement_id"], []).append(c)

                total_feature_matches = 0
                for agr_id, agr_chunks in chunks_by_agr.items():
                    texts = [c["text"] for c in agr_chunks]
                    features_list = field_feature_extractor.batch_extract_features(texts)

                    # Annotate each chunk with its feature matches
                    for chunk, feat in zip(agr_chunks, features_list):
                        chunk["field_features"] = {
                            "matched_fields": list(feat.matched_field_names),
                            "matched_categories": list(feat.matched_categories),
                            "top_confidence": feat.top_confidence,
                            "n_matches": len(feat.matches),
                            "summary": feat.summary_line(),
                        }
                        total_feature_matches += len(feat.matches)

                    # Build per-doc feature summary for enriching macro summaries
                    doc_stats = field_feature_extractor.get_field_feature_summary(features_list)
                    if doc_stats["top_fields"]:
                        top_str = ", ".join(f"{fn}" for fn, cnt in doc_stats["top_fields"][:8])
                        cat_str = ", ".join(doc_stats["category_distribution"].keys())
                        doc_feature_summaries[agr_id] = (
                            f"Detected fields: {top_str}\n"
                            f"Categories: {cat_str}"
                        )

                progress("field_features",
                         f"{total_feature_matches} feature matches across "
                         f"{len(all_embedding_chunks)} chunks, "
                         f"{len(doc_feature_summaries)} docs with features")
            except Exception as e:
                logger.warning(f"Field feature extraction failed (non-fatal): {e}")
                progress("field_features", f"Skipped: {e}")

        # ── Embed ──
        chunk_texts = [c["text"] for c in all_embedding_chunks]
        chunk_ids = [c["chunk_id"] for c in all_embedding_chunks]

        progress("embed", f"Embedding {len(all_embedding_chunks)} chunks on {config.device}...")
        chunk_embeddings = embedder.embed_for_clustering(chunk_texts, show_progress=True, batch_size=8)

        # Field-aware document summaries for macro clustering
        from core.macro_clusterer import get_field_summaries
        summaries = get_field_summaries(agreements, all_embedding_chunks, chunk_embeddings, embed_fn=_embed_fn)

        # Enrich summaries with field feature annotations
        if doc_feature_summaries:
            for i, agr in enumerate(agreements):
                feat_summary = doc_feature_summaries.get(agr["agreement_id"])
                if feat_summary:
                    summaries[i] = summaries[i] + "\n\n" + feat_summary

        summary_embeddings = embedder.embed_for_clustering(summaries, batch_size=8)

        progress("embed", f"Embedded: {chunk_embeddings.shape[0]} chunks + {summary_embeddings.shape[0]} summaries")

    # Persist to DB
    with _time("persist"):
        for agr in agreements:
            from core.chunker import get_document_summary
            summary = get_document_summary(agr["raw_text"], agr.get("filename", ""))
            store.insert_agreement(agr["agreement_id"], agr.get("filename", ""), agr["raw_text"], summary)

        for i, chunk in enumerate(all_embedding_chunks):
            store.insert_chunk(
                chunk["chunk_id"], chunk["agreement_id"], chunk["text"],
                chunk["chunk_index"], chunk_embeddings[i], chunk.get("token_count", 0),
                clause_id=chunk.get("clause_id"),
            )
        for clause in all_clauses:
            store.insert_clause(
                clause["clause_id"], clause["agreement_id"],
                clause.get("clause_number", ""), clause.get("clause_title", ""),
                clause.get("heading_level", 0), clause.get("clause_index", 0),
                clause.get("full_text", ""), clause.get("token_count", 0),
            )
        lance_store.upsert_chunks(
            chunk_ids=chunk_ids, vectors=chunk_embeddings, texts=chunk_texts,
            agreement_ids=[c["agreement_id"] for c in all_embedding_chunks],
        )
        lance_store.upsert_documents(
            agreement_ids=[a["agreement_id"] for a in agreements],
            vectors=summary_embeddings,
            filenames=[a.get("filename", "") for a in agreements],
            summaries=summaries,
        )
        progress("persist", f"Stored {len(all_embedding_chunks)} chunks + {len(all_clauses)} clauses")

    pipeline_stats["n_embedding_chunks"] = len(all_embedding_chunks)
    pipeline_stats["n_extraction_chunks"] = len(all_extraction_chunks)
    pipeline_stats["n_clauses"] = len(all_clauses)
    pipeline_stats["field_features_enabled"] = config.enable_field_features
    if field_feature_extractor:
        pipeline_stats["field_features"] = {
            "predefined_fields": field_feature_extractor.predefined_count,
            "business_fields": field_feature_extractor.business_count,
            "total_fields": field_feature_extractor.total_count,
            "docs_with_features": len(doc_feature_summaries),
        }
    store.update_run(run_id, "embed_complete")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: Parallel EVoC Clustering (macro ∥ micro)
    # ══════════════════════════════════════════════════════════════════════
    with _time("cluster"):
        from core.evoc_pipeline import (
            evoc_cluster, evoc_optimize, _select_best_layer, _llm_label_clusters,
        )
        from core.topic_merger import HierarchicalTopicMerger

        n_docs = len(agreements)
        n_chunks = len(all_embedding_chunks)

        # ── 3a: Macro EVoC on document summaries ──
        progress("cluster", f"Macro EVoC clustering {n_docs} documents...")
        domain_map = {}

        if n_docs < 10:
            # Too few docs → single domain
            macro_labels = np.zeros(n_docs, dtype=int)
            domain_id = f"d_{uuid.uuid4().hex[:8]}"
            centroid = summary_embeddings.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            store.upsert_domain(domain_id, "All Documents", "Single domain (few docs)",
                                n_docs, 1.0, centroid)
            for i in range(n_docs):
                store.update_agreement_domain(agreements[i]["agreement_id"], domain_id,
                                              float(np.dot(summary_embeddings[i], centroid)))
            domain_map = {0: {"domain_id": domain_id, "agreement_indices": list(range(n_docs)),
                              "label": "All Documents"}}
        else:
            if config.use_optuna:
                progress("cluster", f"  Macro Optuna ({config.optuna_trials} trials)...")
                best_params, macro_labels, best_score = evoc_optimize(
                    summary_embeddings, n_trials=min(config.optuna_trials, 15),
                    min_clusters=3, progress_fn=progress,
                )
                progress("cluster", f"  Macro Optuna best: score={best_score:.4f}")
            else:
                macro_labels, macro_layers, macro_persist = evoc_cluster(
                    summary_embeddings,
                    base_min_cluster_size=max(3, n_docs // 50),
                    noise_level=config.evoc_noise_level,
                    n_neighbors=min(15, n_docs - 1),
                )
                macro_labels, _, _ = _select_best_layer(
                    macro_layers, summary_embeddings, macro_persist, prefer_granularity=True)

            llm_labels = _llm_label_clusters(macro_labels, summaries, "legal agreement summaries")

            for tid in sorted(set(t for t in macro_labels if t != -1)):
                mask = [i for i, t in enumerate(macro_labels) if t == tid]
                centroid = summary_embeddings[mask].mean(axis=0)
                centroid /= np.linalg.norm(centroid) + 1e-12
                info = llm_labels.get(tid, {"label": f"Domain_{tid}", "description": ""})
                did = f"d_{uuid.uuid4().hex[:8]}"
                domain_map[tid] = {"domain_id": did, "agreement_indices": mask, "label": info["label"]}
                store.upsert_domain(did, info["label"], info.get("description", ""),
                                    len(mask),
                                    float(np.mean([np.dot(summary_embeddings[i], centroid) for i in mask])),
                                    centroid)
                for idx in mask:
                    store.update_agreement_domain(
                        agreements[idx]["agreement_id"], did,
                        float(np.dot(summary_embeddings[idx], centroid)))

            # Outlier domain
            outlier_mask = [i for i, t in enumerate(macro_labels) if t == -1]
            if outlier_mask:
                odid = f"d_{uuid.uuid4().hex[:8]}"
                store.upsert_domain(odid, "Uncategorized", "Unassigned", len(outlier_mask))
                domain_map[-1] = {"domain_id": odid, "agreement_indices": outlier_mask, "label": "Uncategorized"}
                for idx in outlier_mask:
                    store.update_agreement_domain(agreements[idx]["agreement_id"], odid, 0.0)

        n_domains = len([k for k in domain_map if k != -1])
        progress("cluster", f"  {n_domains} macro domains")

        # ── 3b: Micro EVoC on clause/embedding chunks ──
        progress("cluster", f"Micro EVoC clustering {n_chunks} chunks...")
        total_clusters = 0

        # Apply guided clustering if user provided keyword groups
        clustering_embeddings = chunk_embeddings  # default: use raw embeddings
        if config.guided_keywords:
            progress("cluster", f"  Applying guided clustering "
                     f"({len(config.guided_keywords)} keyword groups, alpha={config.guided_alpha})")
            clustering_embeddings = compute_guided_embeddings(
                embeddings=chunk_embeddings,
                chunk_texts=chunk_texts,
                keyword_groups=config.guided_keywords,
                embed_fn=lambda texts: embedder.embed(texts),
                alpha=config.guided_alpha,
            )
            pipeline_stats["guided_clustering"] = {
                "keyword_groups": {k: len(v) for k, v in config.guided_keywords.items()},
                "alpha": config.guided_alpha,
            }

        # Apply multi-dimensional signal aggregation (field-overlap + obligation patterns)
        if config.enable_multidim_signals:
            field_features_list = [
                c.get("field_features") for c in all_embedding_chunks
            ]
            progress("cluster", f"  Applying multi-dimensional signals "
                     f"(field_overlap={config.field_overlap_weight}, "
                     f"obligation={config.obligation_weight})")
            clustering_embeddings = compute_multidimensional_embeddings(
                embeddings=clustering_embeddings,
                chunk_texts=chunk_texts,
                field_features_per_chunk=field_features_list,
                field_overlap_weight=config.field_overlap_weight,
                obligation_weight=config.obligation_weight,
            )
            pipeline_stats["multidim_signals"] = {
                "field_overlap_weight": config.field_overlap_weight,
                "obligation_weight": config.obligation_weight,
            }

        if config.use_optuna:
            progress("cluster", f"  Micro Optuna ({config.optuna_trials} trials)...")
            best_params, clause_labels, best_score = evoc_optimize(
                clustering_embeddings, n_trials=config.optuna_trials,
                min_clusters=3, progress_fn=progress,
            )
            progress("cluster", f"  Micro Optuna best: score={best_score:.4f}, params={best_params}")
        else:
            clause_labels, clause_layers, clause_persist = evoc_cluster(
                clustering_embeddings,
                base_min_cluster_size=max(3, n_chunks // 100),
                noise_level=config.evoc_noise_level,
                n_neighbors=min(15, n_chunks - 1),
            )
            clause_labels, _, _ = _select_best_layer(clause_layers, clustering_embeddings, clause_persist)

        # Save layers
        store.save_cluster_layers("clause", [clause_labels], chunk_ids, [1.0], 0, [{}])

        # LLM-label clusters
        clause_cluster_labels = _llm_label_clusters(clause_labels, chunk_texts, "legal contract clauses")

        # Pre-compute corpus-wide document frequency for BM25 keyword extraction
        corpus_df: dict[str, int] = {}
        for txt in chunk_texts:
            for term in set(_tokenize_for_bm25(txt)):
                corpus_df[term] = corpus_df.get(term, 0) + 1
        corpus_size = len(chunk_texts)

        # Create cluster records with domain linkage
        for tid in sorted(set(t for t in clause_labels if t != -1)):
            mask = [i for i, t in enumerate(clause_labels) if t == tid]
            centroid = chunk_embeddings[mask].mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            info = clause_cluster_labels.get(tid, {"label": f"Clause_{tid}", "description": ""})
            cid = f"c_{uuid.uuid4().hex[:8]}"

            # Find most common domain for chunks in this cluster (majority vote)
            domain_id = ""
            if domain_map:
                agr_ids_in = [all_embedding_chunks[i]["agreement_id"] for i in mask]
                domain_votes: dict[str, int] = {}
                for aid in agr_ids_in:
                    for dtid, dr in domain_map.items():
                        if dtid == -1:
                            continue
                        dr_agr_ids = set(agreements[idx]["agreement_id"] for idx in dr["agreement_indices"])
                        if aid in dr_agr_ids:
                            domain_votes[dr["domain_id"]] = domain_votes.get(dr["domain_id"], 0) + 1
                            break
                if domain_votes:
                    domain_id = max(domain_votes, key=domain_votes.get)

            # Extract BM25-weighted thematic keywords for this cluster
            cluster_texts = [chunk_texts[i] for i in mask]
            cluster_kw = extract_cluster_keywords(
                cluster_texts, corpus_df=corpus_df, corpus_size=corpus_size, top_k=20,
            )

            store.upsert_cluster(
                cluster_id=cid, domain_id=domain_id or None,
                label=info["label"], description=info.get("description", ""),
                keywords=cluster_kw, chunk_count=len(mask),
                agreement_count=len(set(all_embedding_chunks[i]["agreement_id"] for i in mask)),
                quality_score=0.0, centroid=centroid,
            )
            store.save_centroid(cid, centroid, len(mask))

            for idx in mask:
                store.insert_cluster_assignment(
                    all_embedding_chunks[idx]["chunk_id"], cid,
                    float(np.dot(chunk_embeddings[idx], centroid)),
                )
                cls_id = all_embedding_chunks[idx].get("clause_id")
                if cls_id:
                    try:
                        store.conn.execute(
                            "UPDATE clauses SET clause_type_id = ? WHERE clause_id = ? AND clause_type_id IS NULL",
                            [cid, cls_id])
                    except Exception:
                        pass
            total_clusters += 1

        progress("cluster", f"  {total_clusters} micro clusters, {n_domains} macro domains")

        # ── 3c: Soft (probabilistic) cluster memberships ──
        # Compute softmax membership distribution over top-K clusters for each
        # chunk.  Used for: detecting ambiguous boundary chunks, probabilistic
        # routing to multiple extraction schemas, and human review queuing.
        soft_memberships: dict[str, dict[str, float]] = {}
        low_confidence_chunks: list[int] = []
        all_centroids: dict[str, np.ndarray] = {}

        if config.enable_soft_membership and total_clusters >= 2:
            # Collect all centroids we just created
            all_centroids = store.get_all_centroids()
            centroid_vecs = {cid: vec for cid, (vec, _) in all_centroids.items()}
            soft_memberships = compute_soft_memberships(
                chunk_embeddings, centroid_vecs,
                top_k=config.soft_membership_k,
            )
            low_confidence_chunks = identify_low_confidence_chunks(
                soft_memberships, config.low_confidence_threshold)
            progress("cluster", f"  Soft memberships: {len(soft_memberships)} chunks, "
                     f"{len(low_confidence_chunks)} low-confidence (< {config.low_confidence_threshold})")
            pipeline_stats["soft_membership"] = {
                "total_chunks_with_soft": len(soft_memberships),
                "low_confidence_count": len(low_confidence_chunks),
                "threshold": config.low_confidence_threshold,
            }

    pipeline_stats["n_clusters"] = total_clusters
    pipeline_stats["n_domains"] = n_domains
    store.update_run(run_id, "cluster_complete")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 4: RLM V3 Discovery + Agentic RAG Extraction
    # ══════════════════════════════════════════════════════════════════════
    with _time("discovery_extraction"):
        clusters = store.conn.execute("""
            SELECT c.cluster_id, c.domain_id, c.label, c.keywords, c.chunk_count
            FROM clusters c WHERE c.chunk_count >= ?
            ORDER BY c.chunk_count DESC
        """, [config.min_cluster_chunks]).fetchdf().to_dict("records")

        progress("discovery", f"Processing {len(clusters)} clusters (RLM V3 discovery + RAG extraction)...")

        # Build in-memory field_features lookup (chunk_id → features dict)
        # so we can re-attach features to DB-queried chunk rows
        chunk_field_features: dict[str, dict] = {}
        for c in all_embedding_chunks:
            ff = c.get("field_features")
            if ff:
                chunk_field_features[c["chunk_id"]] = ff

        # Pre-create shared embedder and RAG config to avoid per-cluster re-instantiation
        shared_embedder = _get_embedder(config)
        from core.rag_extractor import RAGConfig
        shared_rag_config = RAGConfig(
            max_sample_docs=int(os.getenv("RAG_MAX_SAMPLE_DOCS", "10")),
            max_retry_rounds=int(os.getenv("RAG_MAX_RETRY_ROUNDS", "3")),
            enable_reviewer=os.getenv("RAG_ENABLE_REVIEWER", "1") == "1",
            min_extraction_confidence=config.min_extraction_confidence,
        )

        # Prepare all cluster chunk rows up front (single DB scan)
        all_cluster_chunks = store.conn.execute("""
            SELECT ca.cluster_id, ch.chunk_id, ch.chunk_text, ch.agreement_id
            FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            ORDER BY ca.cluster_id, ch.chunk_index
        """).fetchdf().to_dict("records")

        cluster_chunk_map: dict[str, list[dict]] = {}
        for row in all_cluster_chunks:
            cid = row.pop("cluster_id")
            # Re-attach field features from in-memory store
            row["field_features"] = chunk_field_features.get(row["chunk_id"])
            cluster_chunk_map.setdefault(cid, []).append(row)

        # Process clusters concurrently with a semaphore to limit parallelism
        max_concurrent = int(os.getenv("CLUSTER_CONCURRENCY", "3"))
        sem = asyncio.Semaphore(max_concurrent)
        total_fields = 0
        total_extractions = 0
        cluster_profiles: list[ClusterProfile] = []

        async def _guarded_process(ci: int, cluster: dict) -> dict:
            """Run discovery + confidence-gated extraction for one cluster under semaphore."""
            cid = cluster["cluster_id"]
            chunk_rows = cluster_chunk_map.get(cid, [])
            if len(chunk_rows) < config.min_cluster_chunks:
                return {"fields": 0, "extractions": 0}
            async with sem:
                return await _process_cluster(
                    cluster, chunk_rows, store, llm, config,
                    ci, len(clusters), progress,
                    shared_embedder=shared_embedder,
                    shared_rag_config=shared_rag_config,
                    personalization_ctx=personalization_ctx,
                )

        results = await asyncio.gather(
            *[_guarded_process(ci, cl) for ci, cl in enumerate(clusters)],
            return_exceptions=True,
        )

        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Cluster processing error: {r}")
                continue
            total_fields += r["fields"]
            total_extractions += r["extractions"]
            profile = r.get("profile")
            if profile and isinstance(profile, ClusterProfile):
                cluster_profiles.append(profile)

        progress("discovery", f"Done: {total_fields} fields, {total_extractions} extractions "
                 f"across {len(clusters)} clusters, {len(cluster_profiles)} profiles built")

    pipeline_stats["n_fields"] = total_fields
    pipeline_stats["n_extractions"] = total_extractions
    pipeline_stats["n_cluster_profiles"] = len(cluster_profiles)
    store.update_run(run_id, "extraction_complete")

    # ── Persist cluster profiles as JSON artifacts ──
    # Each profile is the 'agreement genome' — field definitions, keywords,
    # clause sub-types, obligations, extraction quality, confidence gate outcome.
    if cluster_profiles:
        profiles_dir = artifacts_root / account_id / run_id / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        for profile in cluster_profiles:
            # Attach soft membership data to the profile
            if soft_memberships:
                # Find chunk indices assigned to this cluster
                profile_soft = {}
                for idx_str, memberships in soft_memberships.items():
                    if profile.cluster_id in memberships:
                        profile_soft[idx_str] = memberships
                profile.soft_membership = profile_soft

            pf_path = profiles_dir / f"{profile.cluster_id}.json"
            with open(pf_path, "w") as f:
                json.dump(profile.to_dict(), f, indent=2, default=str)
        progress("profiles", f"Persisted {len(cluster_profiles)} cluster profiles to {profiles_dir}")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 5: [Optional] Intent Discovery
    # ══════════════════════════════════════════════════════════════════════
    if config.run_intents:
        with _time("intents"):
            intent_stats = await _process_intents(store, llm, config, progress)
            pipeline_stats["intent_stats"] = intent_stats
        store.update_run(run_id, "intent_complete")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 6: Wire IncrementalClusterAssigner
    # ══════════════════════════════════════════════════════════════════════
    # Load all cluster centroids into the assigner so new documents can
    # be assigned to existing clusters in <500ms without re-clustering.
    # The assigner uses a 3-tier confidence model:
    #   HIGH (>0.85): assign + update centroid running mean
    #   TENTATIVE (0.60-0.85): assign + flag for review
    #   NOVEL (<0.60): buffer for periodic re-clustering
    incremental_assigner = None
    with _time("assigner_init"):
        try:
            from core.cluster_assigner import IncrementalClusterAssigner
            all_centroids_for_assigner = store.get_all_centroids()
            if all_centroids_for_assigner:
                # Use personalization thresholds if available
                hi_thresh = 0.85
                tent_thresh = 0.60
                if personalization_ctx:
                    hi_thresh = personalization_ctx.high_confidence_threshold
                    tent_thresh = personalization_ctx.tentative_threshold

                incremental_assigner = IncrementalClusterAssigner(
                    embedder=_get_embedder(config),
                    high_confidence_threshold=hi_thresh,
                    tentative_threshold=tent_thresh,
                )
                incremental_assigner.load_centroids(all_centroids_for_assigner)
                progress("assigner",
                         f"IncrementalClusterAssigner loaded with "
                         f"{len(all_centroids_for_assigner)} centroids "
                         f"(hi={hi_thresh}, tent={tent_thresh})")
                pipeline_stats["incremental_assigner"] = {
                    "n_centroids": len(all_centroids_for_assigner),
                    "high_threshold": hi_thresh,
                    "tentative_threshold": tent_thresh,
                }
        except Exception as e:
            logger.warning(f"IncrementalClusterAssigner init failed (non-fatal): {e}")
            progress("assigner", f"Skipped: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - t0
    store.update_run(run_id, "complete", status="complete")

    # Close feedback store connection
    if feedback_store:
        try:
            feedback_store.close()
        except Exception:
            pass

    pipeline_stats.update({
        "stage_timings": stage_timings,
        "total_elapsed_s": round(total_elapsed, 1),
        "llm_stats": llm.stats,
        "db_path": str(db),
        "feedback_enabled": config.enable_feedback and feedback_store is not None,
        "confidence_gate_enabled": config.enable_confidence_gate,
        "soft_membership_enabled": config.enable_soft_membership,
        "profiles_dir": str(artifacts_root / account_id / run_id / "profiles") if cluster_profiles else None,
    })

    progress("done", f"Pipeline complete in {total_elapsed:.0f}s — "
             f"{pipeline_stats['n_documents']} docs, {total_clusters} clusters, "
             f"{total_fields} fields, {total_extractions} extractions"
             f"{f', {len(cluster_profiles)} profiles' if cluster_profiles else ''}")

    # Save manifest
    manifest_dir = artifacts_root / account_id / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_dir / "pipeline_manifest.json", "w") as f:
        json.dump(pipeline_stats, f, indent=2, default=str)

    return pipeline_stats


# ── Incremental Document Assignment (post-pipeline entry point) ───────────────


async def assign_new_documents(
    pdf_paths: list[str | Path],
    account_id: str = "default",
    config: MinimalPipelineConfig | None = None,
) -> list[dict]:
    """Assign new documents to existing clusters without re-running the full pipeline.

    This is the incremental ingestion path.  For each new PDF:
      1. Parse and chunk the document.
      2. Embed chunks using the same Nomic embedder.
      3. Use IncrementalClusterAssigner to find the nearest cluster centroid.
      4. Return assignment results with confidence tier (high/tentative/novel).

    Documents assigned with HIGH confidence get their centroids updated
    (running mean).  TENTATIVE assignments are flagged for human review.
    NOVEL documents are buffered; when the buffer exceeds threshold, a
    re-clustering is recommended.

    Args:
        pdf_paths:  List of paths to individual PDF files.
        account_id: Tenant/account identifier (must match previous pipeline run).
        config:     Pipeline configuration (for DB path and embedding settings).

    Returns:
        List of assignment result dicts, one per document:
        {
            "filename": str,
            "agreement_id": str,
            "cluster_id": str | None,
            "cluster_label": str,
            "confidence": float,
            "assignment_type": "high_confidence" | "tentative" | "novel",
            "top_matches": [(cluster_id, similarity), ...],
        }
    """
    config = config or MinimalPipelineConfig()
    from config import DATA_DIR
    from core.store import ClusteringStore
    from core.nomic_embedder import NomicEmbedder
    from core.cluster_assigner import IncrementalClusterAssigner
    from core.pdf_parser import parse_pdf as parse_single_pdf

    db = Path(config.db_path) if config.db_path else DATA_DIR / f"{account_id}.duckdb"
    store = ClusteringStore(db)
    embedder = NomicEmbedder.get_instance(device=config.device)

    # Load personalization thresholds if feedback is enabled
    hi_thresh = 0.85
    tent_thresh = 0.60
    if config.enable_feedback:
        try:
            from core.feedback_store import FeedbackStore
            from core.personalization import PersonalizationEngine
            tenant = config.tenant_id or account_id
            fb_dir = config.feedback_data_dir or str(DATA_DIR)
            fb_store = FeedbackStore(tenant_id=tenant, data_dir=fb_dir)
            pe = PersonalizationEngine(fb_store, main_db_path=str(db))
            ctx = pe.get_context()
            hi_thresh = ctx.high_confidence_threshold
            tent_thresh = ctx.tentative_threshold
            fb_store.close()
        except Exception:
            pass

    # Load centroids and initialise assigner
    all_centroids = store.get_all_centroids()
    if not all_centroids:
        return [{"error": "no_centroids", "message": "Run full pipeline first"}]

    assigner = IncrementalClusterAssigner(
        embedder=embedder,
        high_confidence_threshold=hi_thresh,
        tentative_threshold=tent_thresh,
    )
    assigner.load_centroids(all_centroids)

    # Build cluster_id → label lookup
    cluster_labels = {}
    try:
        rows = store.conn.execute("SELECT cluster_id, label FROM clusters").fetchall()
        cluster_labels = {r[0]: r[1] for r in rows}
    except Exception:
        pass

    results = []
    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        try:
            doc = parse_single_pdf(str(pdf_path))
            text = doc.get("raw_text", "")
            if len(text) < 100:
                results.append({"filename": pdf_path.name, "error": "too_short"})
                continue

            # Embed the full document text (use first 4000 chars for assignment)
            assignment = assigner.assign_document(text[:4000], doc_id=doc.get("agreement_id", ""))

            cid = assignment["cluster_id"]
            results.append({
                "filename": pdf_path.name,
                "agreement_id": doc.get("agreement_id", ""),
                "cluster_id": cid,
                "cluster_label": cluster_labels.get(cid, "Unknown") if cid else "Novel",
                "confidence": assignment["confidence"],
                "assignment_type": assignment["assignment_type"].value,
                "top_matches": assignment["top_matches"][:5],
            })

            # Persist novel documents for re-clustering buffer
            if assignment["assignment_type"].value == "novel":
                try:
                    emb = embedder.embed_single(text[:4000], normalize=True)
                    store.insert_novel_document(doc.get("agreement_id", ""), text, emb)
                except Exception:
                    pass

        except Exception as e:
            results.append({"filename": pdf_path.name, "error": str(e)})

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Minimal clustering → discovery → RAG extraction pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline mode")

    # ── Full pipeline command (default) ──
    run_parser = subparsers.add_parser("run", help="Run full pipeline on PDF directory")
    run_parser.add_argument("pdf_dir", help="Path to PDF directory")
    run_parser.add_argument("--account-id", default="default")
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--max-docs", type=int, default=None)
    run_parser.add_argument("--max-concurrency", type=int, default=8)
    run_parser.add_argument("--optuna-trials", type=int, default=30)
    run_parser.add_argument("--no-optuna", action="store_true")
    run_parser.add_argument("--discovery", default="rlm_v3_unified", choices=["rlm_v3", "rlm_v3_unified"])
    run_parser.add_argument("--no-extraction", action="store_true")
    run_parser.add_argument("--run-intents", action="store_true", help="Enable optional intent discovery")
    run_parser.add_argument("--no-field-features", action="store_true", help="Disable legacy field feature extraction")
    run_parser.add_argument("--legacy-fields-dir", default=None, help="Path to apr-fields-processor root")
    run_parser.add_argument("--guided-keywords", default=None,
                        help='JSON file or inline JSON: {"termination": ["termination", "cancellation"], ...}')
    run_parser.add_argument("--guided-alpha", type=float, default=0.3,
                        help="Guided clustering strength (0=none, 1=full pull). Default: 0.3")
    run_parser.add_argument("--no-feedback", action="store_true", help="Disable feedback/personalization loop")
    run_parser.add_argument("--tenant-id", default=None, help="Tenant ID for feedback sidecar DB")
    run_parser.add_argument("--no-confidence-gate", action="store_true", help="Disable confidence-gated extraction")
    run_parser.add_argument("--confidence-threshold", type=float, default=0.65,
                        help="Confidence gate threshold. Default: 0.65")
    run_parser.add_argument("--no-soft-membership", action="store_true", help="Disable probabilistic memberships")
    run_parser.add_argument("--enable-multidim", action="store_true",
                        help="Enable multi-dimensional signal aggregation (field-overlap + obligations)")
    run_parser.add_argument("--db", default=None)
    run_parser.add_argument("--lance", default=None)
    run_parser.add_argument("--log-level", default="INFO")

    # ── Incremental assign command ──
    assign_parser = subparsers.add_parser("assign", help="Assign new PDFs to existing clusters")
    assign_parser.add_argument("pdf_paths", nargs="+", help="Paths to individual PDF files")
    assign_parser.add_argument("--account-id", default="default")
    assign_parser.add_argument("--device", default="cuda")
    assign_parser.add_argument("--db", default=None)
    assign_parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    # Default to 'run' if no subcommand and pdf_dir given positionally
    if args.command is None:
        # Backward compat: treat as 'run' with pdf_dir as first positional
        parser.print_help()
        return 1

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)-20s %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "assign":
        config = MinimalPipelineConfig(device=args.device, db_path=args.db)
        result = asyncio.run(assign_new_documents(
            pdf_paths=args.pdf_paths,
            account_id=args.account_id,
            config=config,
        ))
        print(json.dumps(result, indent=2, default=str))
        return 0

    # ── 'run' command ──
    # Parse guided keywords from file or inline JSON
    guided_kw = None
    if args.guided_keywords:
        gk = args.guided_keywords
        if os.path.isfile(gk):
            with open(gk) as f:
                guided_kw = json.load(f)
        else:
            guided_kw = json.loads(gk)

    config = MinimalPipelineConfig(
        device=args.device,
        discovery_mode=args.discovery,
        max_docs=args.max_docs,
        max_llm_concurrency=args.max_concurrency,
        use_optuna=not args.no_optuna,
        optuna_trials=args.optuna_trials,
        extract_values=not args.no_extraction,
        run_intents=args.run_intents,
        enable_field_features=not args.no_field_features,
        legacy_fields_dir=args.legacy_fields_dir,
        guided_keywords=guided_kw,
        guided_alpha=args.guided_alpha,
        enable_feedback=not args.no_feedback,
        tenant_id=args.tenant_id,
        enable_confidence_gate=not args.no_confidence_gate,
        confidence_gate_threshold=args.confidence_threshold,
        enable_soft_membership=not args.no_soft_membership,
        enable_multidim_signals=args.enable_multidim,
        db_path=args.db,
        lance_path=args.lance,
    )

    result = asyncio.run(run_minimal_pipeline(
        pdf_dir=args.pdf_dir,
        account_id=args.account_id,
        config=config,
    ))

    print(json.dumps(result, indent=2, default=str))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())
