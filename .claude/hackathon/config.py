"""
Configuration module for Clustering V2 MVP.

Replaces legacy cloud dependencies (AIDB, AIFlow, Azure ServiceBus, Redis)
with local-first alternatives (DuckDB, sentence-transformers, direct OpenAI).

Legacy pain point: 8+ env vars for Azure connections, gRPC service URLs, Redis.
V2 approach: Single config file, local storage, optional cloud LLM.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample_agreements"
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "evoc_200_refined.duckdb")))

# ── LLM Configuration ─────────────────────────────────────────────────────────
# Supports: "openai", "ollama", "litellm", "gemini"
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# Gemini (Vertex AI) — requires GOOGLE_APPLICATION_CREDENTIALS pointing at a
# service-account JSON, GOOGLE_CLOUD_PROJECT, and optionally GOOGLE_CLOUD_LOCATION.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# ── Embedding Configuration ───────────────────────────────────────────────────
# Legacy: AIFlow gRPC → all-MiniLM-L12-v3 (384-dim), cloud-hosted
# V2: Local sentence-transformers, no network dependency
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # MiniLM-L6-v2 produces 384-dim vectors
CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── GPU Embedding Configuration ───────────────────────────────────────────────
# Nomic: 768-dim (full) or 256/512 via Matryoshka truncation
NOMIC_MODEL = os.getenv("NOMIC_MODEL", "nomic-ai/nomic-embed-text-v1.5")
NOMIC_DEVICE = os.getenv("NOMIC_DEVICE", "cuda")
NOMIC_DIM = int(os.getenv("NOMIC_DIM", "768"))

# ── GPU Pipeline Configuration ────────────────────────────────────────────────
LANCEDB_PATH = os.getenv("LANCEDB_PATH", str(DATA_DIR / "vectors.lance"))
OPTIMIZE_TRIALS = int(os.getenv("OPTIMIZE_TRIALS", "30"))
OPTIMIZE_PARALLEL = int(os.getenv("OPTIMIZE_PARALLEL", "4"))
USE_RAY = os.getenv("USE_RAY", "true").lower() in ("1", "true", "yes")

# ── Azure OpenAI (Field Discovery) ────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")

# ── RLM (Recursive Language Model via dspy) ───────────────────────────────────
# RLM_SUB_MODEL: cheaper/faster model for llm_query() sub-calls inside the REPL.
# The root LM (AZURE_OPENAI_DEPLOYMENT) orchestrates; the sub-LM does snippet analysis.
# Set to "" to use the same model for both.
RLM_SUB_MODEL = os.getenv("RLM_SUB_MODEL", "")
RLM_MAX_ITERATIONS = int(os.getenv("RLM_MAX_ITERATIONS", "25"))
RLM_MAX_LLM_CALLS = int(os.getenv("RLM_MAX_LLM_CALLS", "50"))

# ── Azure Blob Snapshots ──────────────────────────────────────────────────────
# Set AZURE_STORAGE_CONNECTION_STRING to enable blob snapshots.
# Container is created automatically on first use.
SNAPSHOT_CONTAINER = os.getenv("SNAPSHOT_CONTAINER", "clustering-snapshots")
SNAPSHOT_BLOB_PREFIX = os.getenv("SNAPSHOT_BLOB_PREFIX", "v2")

# ── Chunking ───────────────────────────────────────────────────────────────────
# Legacy: Paragraph-level chunking via fields_processor
# V2: Sliding window with configurable overlap
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "256"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "2000"))  # ~500 tokens for macro summaries

# ── Macro Clustering (NEW — replaces flat BERTopic) ───────────────────────────
# Legacy: Single-level BERTopic on field-summary embeddings
# V2: Two-stage — macro domains from doc summaries, then micro per domain
MACRO_MIN_CLUSTER_SIZE = int(os.getenv("MACRO_MIN_CLUSTER_SIZE", "5"))
MACRO_MIN_SAMPLES = int(os.getenv("MACRO_MIN_SAMPLES", "2"))

# ── Micro Clustering ──────────────────────────────────────────────────────────
# Legacy: UMAP(n_neighbors=15-50, n_components=5-15) + HDBSCAN(min_cluster_size=5-20)
# V2: Same algorithms, applied per-domain instead of globally
MICRO_MIN_CLUSTER_SIZE = int(os.getenv("MICRO_MIN_CLUSTER_SIZE", "3"))
MICRO_MIN_SAMPLES = int(os.getenv("MICRO_MIN_SAMPLES", "2"))
UMAP_N_NEIGHBORS = int(os.getenv("UMAP_N_NEIGHBORS", "15"))
UMAP_N_COMPONENTS = int(os.getenv("UMAP_N_COMPONENTS", "5"))
UMAP_MIN_DIST = float(os.getenv("UMAP_MIN_DIST", "0.0"))

# ── Topic Merging (FIELD-310 — NEW) ───────────────────────────────────────────
MERGE_SIMILARITY_THRESHOLD = float(os.getenv("MERGE_SIMILARITY_THRESHOLD", "0.82"))
SYNONYM_SIMILARITY_THRESHOLD = float(os.getenv("SYNONYM_SIMILARITY_THRESHOLD", "0.78"))

# ── Incremental Assignment (FIELD-801 — NEW) ──────────────────────────────────
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.85"))
TENTATIVE_THRESHOLD = float(os.getenv("TENTATIVE_THRESHOLD", "0.60"))
NOVEL_BUFFER_SIZE = int(os.getenv("NOVEL_BUFFER_SIZE", "500"))

# ── KeyBERT Clause Fingerprinting ────────────────────────────────────────────
KEYBERT_FINGERPRINTS_PATH = Path(os.getenv(
    "KEYBERT_FINGERPRINTS_PATH", str(DATA_DIR / "models" / "clause_fingerprints.json")
))
KEYBERT_CLASSIFICATION_THRESHOLD = float(os.getenv("KEYBERT_CLASSIFICATION_THRESHOLD", "0.3"))
KEYBERT_UMAP_TARGET_WEIGHT = float(os.getenv("KEYBERT_UMAP_TARGET_WEIGHT", "0.3"))
KEYBERT_PRIOR_WEIGHT = float(os.getenv("KEYBERT_PRIOR_WEIGHT", "0.15"))
KEYBERT_AUGMENT_TOP_K = int(os.getenv("KEYBERT_AUGMENT_TOP_K", "3"))

# ── Association Rule Mining (ARM) ────────────────────────────────────────────
ENABLE_ARM_ENRICHMENT = os.getenv("ENABLE_ARM_ENRICHMENT", "true").lower() in ("1", "true", "yes")
ARM_MIN_SUPPORT = float(os.getenv("ARM_MIN_SUPPORT", "0.20"))
ARM_MIN_CONFIDENCE = float(os.getenv("ARM_MIN_CONFIDENCE", "0.7"))
ARM_MIN_LIFT = float(os.getenv("ARM_MIN_LIFT", "1.8"))
FIELD_ARM_MIN_SUPPORT = float(os.getenv("FIELD_ARM_MIN_SUPPORT", "0.05"))
FIELD_ARM_MIN_CONFIDENCE = float(os.getenv("FIELD_ARM_MIN_CONFIDENCE", "0.5"))
FIELD_ARM_MIN_LIFT = float(os.getenv("FIELD_ARM_MIN_LIFT", "1.5"))

# ── Retrieval ──────────────────────────────────────────────────────────────────
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))  # RRF constant

# ── Extraction ─────────────────────────────────────────────────────────────────
# Legacy: HyDE query → AIFlow embedding → AIDB ANN → AIFlow extraction (4 network hops)
# V2: Direct embedding + FAISS local search + direct LLM call (0 network hops for search)
MAX_CONCURRENT_LLM = int(os.getenv("MAX_CONCURRENT_LLM", "16"))
EXAMPLE_SET_SIZE = int(os.getenv("EXAMPLE_SET_SIZE", "20"))

# ── Legacy Legal Stopwords (from bertopic_clusterer.py) ────────────────────────
LEGAL_STOPWORDS = [
    "agreement", "party", "parties", "shall", "herein", "hereto", "hereby",
    "thereof", "therein", "pursuant", "notwithstanding", "foregoing",
    "whereas", "witnesseth", "undersigned", "aforementioned", "hereunder",
    "thereto", "hereof", "wherein", "thereby", "heretofore",
    "section", "article", "clause", "provision", "paragraph",
    "company", "corporation", "entity", "organization", "firm",
    "date", "day", "month", "year", "period", "term", "time",
    "written", "notice", "consent", "approval", "request",
    "applicable", "reasonable", "material", "sole", "prior",
]
