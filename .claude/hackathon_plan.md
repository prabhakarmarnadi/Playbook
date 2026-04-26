# Clustering V2: Hackathon MVP Plan

## Goal

Build a **working demo of V2 capabilities that don't exist in V1** — in ~2 days, with 2 engineers. The demo must show things the audience has *never seen* from the current system.

**What V1 already does (DON'T rebuild):**
* BERTopic clustering (UMAP + HDBSCAN + c-TF-IDF)
* Bayesian hyperparameter optimization
* LLM-based cluster naming
* HyDE → snippet retrieval → LLM extraction
* Cochran sampling, AIDB integration, Service Bus orchestration

**What V2 shows that's NEW (the hackathon focus):**

| V2 Capability | Why It's Impressive | V1 Equivalent |
|---------------|---------------------|---------------|
| Macro → Micro two-level clustering | "The system knows NDA and SaaS are different *before* clustering" | V1 clusters everything flat |
| Auto ontology construction | "Clusters become a structured knowledge graph with fields and relationships" | V1 outputs topic labels and keywords only |
| Cluster-conditioned extraction | "Extraction prompts know what clause type they're looking at" | V1 uses generic prompts |
| Example Set → Review → Scale UX | "Users validate on 20 docs, then the system scales to 10K" | V1 runs on everything or nothing |
| Composite fields with feedback propagation | "Fix one field, three composites auto-improve" | Doesn't exist |
| BM25 + Dense hybrid search with RRF | "Catches exact statute numbers that semantic search misses" | Doesn't exist |
| Cross-encoder reranking with score visualization | "See why this result ranked higher than that one" | Doesn't exist |
| Field definition CRUD (rename, retype, examples) | "Users own their schema" | Doesn't exist |
| Ontology tree visualization | "See the full knowledge graph: domains → clause types → fields → composites" | V1 outputs flat JSON |
| Cluster quality dashboard with drift detection | "Live cluster health, not just a score in a log file" | V1 computes metrics but doesn't visualize them |
| **Clause-level semantic embeddings** | "Each clause gets its own embedding — no more averaging away critical terms" | V1 embeds field summaries only |
| **Hierarchical topic merging & synonym resolution** | "Payment Terms and Net Payment Terms auto-merge; the system knows they're the same" | V1 treats similar clusters as distinct |
| **Incremental cluster assignment** | "New docs join existing clusters instantly — no expensive re-clustering" | V1 requires full re-clustering for every batch |

> **Key design decision — Macro identification uses embedding-based summary clustering, NOT per-document LLM classification.**
>
> Per-document LLM classification (calling GPT-4o-mini once per uploaded document) is too expensive and slow for 100–500+ docs. Instead, macro domains are discovered by running BERTopic on **document-level summary embeddings** (embed title + first ~500 tokens per doc → UMAP + HDBSCAN → coarse clusters). Then we make only **5–10 LLM calls** to label the resulting macro clusters, not 100–500 calls to classify individual documents. This is ~50–100x cheaper, ~10x faster, and produces the same result: documents grouped into meaningful domains before micro-clustering begins.

**Audience impression target:** "V1 clusters agreements. V2 *understands* them."

---

## UX & Feasibility FAQ (Addressing Team Questions)

### Q1: "User uploads 100 (or 500) docs — how long before we can recommend the first action?"

**Short answer:** First meaningful result in ~30–60 seconds. Full pipeline complete in ~3–8 minutes.

| Step | 100 docs | 500 docs | When user sees something |
|------|----------|----------|-------------------------|
| **1. Parse + chunk** | ~5–10s | ~20–40s | Progress bar: "Processing documents..." |
| **2. Embed summaries** (MiniLM, local GPU/CPU) | ~2–5s | ~10–25s | Progress bar: "Generating embeddings..." |
| **3. Macro clustering** (UMAP + HDBSCAN on summary vectors) | ~1–2s | ~2–5s | **Domain bar chart appears (~30–60s total)** ← first action |
| **4. Label macro clusters** (5–10 LLM calls total, NOT per-doc) | ~5–10s | ~5–10s | Domain labels appear on chart |
| **5. Chunk-level embedding** (MiniLM on all chunks) | ~15–30s | ~60–120s | Progress bar: "Analyzing clause patterns..." |
| **6. Micro clustering per domain** (BERTopic) | ~3–5s/domain | ~5–10s/domain | UMAP scatter appears per domain |
| **7. Label micro clusters** (3–5 LLM calls per domain) | ~10–15s | ~15–25s | Cluster names appear |
| **8. Field discovery** (1 LLM call per cluster) | ~20–40s | ~30–60s | Field schemas appear |
| **Total to first action** | **~30s** | **~60s** | Macro domain chart visible |
| **Total to full ontology** | **~2–3 min** | **~5–8 min** | All clusters + fields visible |

**UX strategy:** Show results progressively. Don't make users wait for the full pipeline:
* **30s:** "Your 100 documents fall into 4 domains: SaaS (42), NDA (28), Employment (18), Vendor (12)"
* **90s:** "Within SaaS, we found 5 clause patterns: Payment Terms, Liability, Auto-Renewal..."
* **3 min:** "Here's the full ontology. Click any cluster to review the example set."

### Q2: "What additional metadata do we get per document beyond V1?"

V1 gives you: topic_id, topic keywords, topic probability. That's it.

V2 gives you a **rich document profile**:

| Metadata | Source | Example |
|----------|--------|---------|
| **Macro domain** + confidence | Summary clustering | `domain: "SaaS", confidence: 0.92` |
| **Domain description** | LLM label (per cluster, not per doc) | `"Software-as-a-Service agreements covering licensing, hosting, and support"` |
| **Clause type assignments** | Micro clustering (per chunk) | `["Payment Terms", "Liability", "Auto-Renewal", "Data Processing"]` |
| **Per-clause extracted fields** | Cluster-conditioned extraction | `Payment Terms: {payment_days: 30, method: "wire", late_fee: 1.5%}` |
| **Universal fields** | Domain-agnostic extraction | `{parties: ["Acme", "Widget Inc"], effective_date: "2024-01-15", governing_law: "Delaware"}` |
| **Composite / derived values** | Formula engine | `{TCV: $4,500,000, cost_per_month: $125,000}` |
| **Cluster typicality score** | Cosine to cluster centroid | `0.87` — how representative this doc is of its cluster |
| **Coverage %** | Fields found / fields expected | `85%` — 17 of 20 expected fields extracted |
| **Anomaly / outlier flags** | Distance from centroid | `"Unusual liability clause — unlimited cap, 95th percentile deviation"` |
| **Clause count distribution** | Chunk → cluster mapping | `{Payment: 2 sections, Liability: 1 section, Termination: 3 sections}` |

### Q3: "What does the clustering data format look like? Is it a tree?"

**Yes — a hierarchical tree (not flat topics).**

```
Account (root)
│
├── 📁 Domain: SaaS/Software (142 docs)
│   │   description: "Software licensing, hosting, and support agreements"
│   │   confidence: 0.89
│   │
│   ├── 💼 Clause: Payment Terms (47 chunks across 38 docs)
│   │   │   quality_score: 0.91, keywords: ["payment", "invoice", "net", "due"]
│   │   │
│   │   ├── 📋 payment_days: int            (coverage: 94%, distribution: {30: 40%, 45: 45%, 60: 15%})
│   │   ├── 📋 payment_method: enum         (coverage: 82%, values: ["wire", "ACH", "check"])
│   │   ├── 📋 late_fee_pct: float          (coverage: 61%, mean: 1.5%)
│   │   ├── 📋 currency: string             (coverage: 97%, dominant: "USD")
│   │   └── 🧮 TCV = unit_price × quantity × term_years   (composite)
│   │
│   ├── 💼 Clause: Liability Limitation (38 chunks across 35 docs)
│   │   ├── 📋 liability_cap: float         (coverage: 89%)
│   │   ├── 📋 cap_type: enum               (values: ["fixed", "multiplier", "unlimited"])
│   │   └── 📋 indemnification_scope: string (coverage: 76%)
│   │
│   └── 💼 Clause: Auto-Renewal (31 chunks across 28 docs)
│       ├── 📋 renewal_term: string          (coverage: 91%)
│       ├── 📋 notice_period: string         (coverage: 87%)
│       └── 🧮 last_opt_out = date_sub(expiration, notice_period)  (composite)
│
├── 📁 Domain: NDA/Confidentiality (89 docs)
│   ├── 💼 Clause: Confidentiality Scope (52 chunks)
│   └── 💼 Clause: Term & Survival (37 chunks)
│
└── 📁 Domain: Employment (71 docs)
    ├── 💼 Clause: Non-Compete (29 chunks)
    └── 💼 Clause: Compensation (42 chunks)
```

**JSON data format:**

```json
{
  "account_id": "acct_123",
  "domains": [
    {
      "domain_id": "d_saas",
      "label": "SaaS/Software",
      "description": "Software licensing, hosting, and support agreements",
      "agreement_count": 142,
      "confidence": 0.89,
      "clusters": [
        {
          "cluster_id": "c_payment",
          "label": "Payment Terms",
          "keywords": ["payment", "invoice", "net", "due"],
          "chunk_count": 47,
          "agreement_count": 38,
          "quality_score": 0.91,
          "fields": [
            {"name": "payment_days", "type": "int", "coverage": 0.94, "source": "auto_discovered"},
            {"name": "payment_method", "type": "enum", "coverage": 0.82, "source": "auto_discovered"}
          ],
          "composites": [
            {"name": "TCV", "expression": "unit_price * quantity * term_years", "result_type": "float"}
          ]
        }
      ]
    }
  ]
}
```

### Q4: "How does the learning loop work? How long does it take?"

**The loop is: Review → Correct → Re-run → Scale.**

| Step | What the user does | Time | What happens behind the scenes |
|------|-------------------|------|-------------------------------|
| **1. View example set** | Clicks a cluster → sees 20 representative docs with extracted fields | Instant | System pre-selected the most representative docs (nearest to centroid) |
| **2. Quick review** | Scans extracted values. Marks 2–3 as incorrect, corrects them | ~2–3 min | Corrections stored. System notes which fields are unreliable. |
| **3. Refine schema (optional)** | Renames a field, adds an example, changes a type | ~1 min | Field definition updated. Extraction prompt rebuilt with new examples. |
| **4. Re-extract on sample** | Clicks "Re-run on sample" | ~30–60s | Re-extracts 20 docs with improved prompt + examples. Shows diff. |
| **5. Review changes** | Confirms improvements look good | ~1 min | Corrections + schema changes locked in. |
| **6. Scale to full cluster** | Clicks "Scale to all 147 docs" | ~2–5 min (async) | Extraction runs on remaining docs using refined schema. Progress bar shown. |

**Total active user time: ~5–7 minutes per cluster.** That covers reviewing, correcting, and scaling.

**What compounds over time:**
* Gold examples from corrections are reused as few-shot examples in extraction prompts
* Field type corrections (string → int) produce cleaner extracted values
* Composite fields automatically benefit from improved base field accuracy
* The system does NOT need full re-clustering — corrections improve extraction within stable clusters

---

## Team Split

### Person A: Pipeline Engineer (Backend + ML)

Owns: macro→micro clustering, ontology construction, cluster-conditioned extraction, composite field engine, hybrid retrieval

### Person B: Product Engineer (UI + Integration)

Owns: Streamlit app, all pages, example set UX, field definition CRUD, ontology tree, search UI, demo polish

**Shared:** Data model (SQLite schema), LLM prompts, sample data prep

---

## Demo Script (What the Audience Sees)

### Act 1: "Domain Intelligence" (60 seconds)

01. Upload 300–500 sample agreements (pre-processed for speed)
02. System clusters them into **macro domains** (SaaS, NDA, Employment, Vendor, Lease) using **embedding-based summary clustering** — embed each doc's title + opening text → UMAP + HDBSCAN → coarse clusters → LLM labels only the clusters (5–10 LLM calls, not 500)
03. UI shows: domain distribution bar chart, cluster confidence per agreement, domain descriptions auto-generated
04. "V1 clusters everything together. V2 knows that indemnity in an NDA is different from indemnity in a SaaS agreement. And it figured that out from the document structure, not by asking an LLM 500 times."

### Act 2: "Auto Ontology" (90 seconds)

05. Within the SaaS domain, run micro-clustering → discover clause types (Payment Terms, Liability, Auto-Renewal, Data Processing, SLA)
06. Click "Payment Terms" cluster →  show **auto-discovered field schema**: `payment_days: int`, `payment_method: enum`, `late_fee: float`, `currency: string`
07. Click "Liability Limitation" → different fields: `liability_cap: float`, `cap_type: enum`, `indemnification_scope: string`
08. Show the **ontology tree**: SaaS → [Payment Terms, Liability, Auto-Renewal, ...] → fields per clause type
09. "The system built this ontology on its own. No predefined schema."

### Act 3: "Example Set Review" (60 seconds)

10. Select "Payment Terms" cluster → system shows **20 representative agreements** as an Example Set
11. Show extracted fields for these 20 docs in an editable table
12. User reviews, corrects 2–3 values, marks 1 field as irrelevant
13. Click **"Scale to Full Cluster"** → extraction runs on all 147 agreements using the refined schema
14. "Users validate on 20 documents. The system scales their decisions to thousands."

### Act 4: "Cluster-Conditioned Extraction" (45 seconds)

15. Side-by-side comparison:
    - **Left panel:** Generic extraction (V1-style) — "Extract structured fields from this section" → noisy, hallucinated fields, wasted tokens
    - **Right panel:** Cluster-conditioned extraction (V2) — "This is a Payment Terms clause. Extract: payment_days, payment_method, late_fee" → precise, clean, fewer tokens
16. Show token count difference (~40% fewer tokens), accuracy difference, hallucination count
17. "Same LLM, same document. But V2 knows what it's looking for."

### Act 5: "Composite Fields — Compound Learning" (60 seconds)

18. Define composite: `TCV = unit_price × quantity × term_years`
19. Values appear instantly — computed from existing extractions, zero LLM calls
20. **Correct `unit_price`** on one agreement → TCV auto-updates
21. Define another composite: `cost_per_month = TCV / (term_years × 12)` — references TCV (composite referencing composite)
22. Correct `unit_price` again → both `TCV` and `cost_per_month` cascade
23. "One correction. Three fields improved. Zero AI cost. The learning compounds."

### Act 6: "Hybrid Search" (60 seconds)

24. Search: "liability cap exceeding $5M in SaaS agreements"
25. Results show **three score columns**: Dense (semantic), BM25 (keyword), RRF Combined
26. Highlight: BM25 caught "$5M" exact match that dense search missed
27. Toggle on **cross-encoder reranker** → results re-sort, show cross-encoder score
28. Click a result → source clause highlighted with extracted fields overlaid
29. "Dense search understands meaning. BM25 catches exact terms. Together, they miss nothing."

### Act 7: "Field CRUD + Schema Ownership" (45 seconds)

30. Rename `payment_days` → `net_payment_terms`
31. Retype a field from `string` → `int` → show re-extraction with cleaner values
32. Add a gold example: `{"text": "net 30 days", "value": 30}`
33. Create a new field from scratch: "early_payment_discount" with description → runs extraction
34. "Customers own their schema. Every refinement makes the system smarter."

### Act 8: "The Big Picture" (30 seconds)

35. Full ontology tree: 5 domains → 25 clause types → 100+ fields → 10 composites
36. Dashboard: cluster quality scores, extraction coverage %, correction count, composite propagation stats
37. "300 contracts → 5 domains → 25 clause types → 100 fields → 10 computed metrics. All discovered automatically, refined by humans, compounding over time."

**Total demo time: ~8 minutes.** Meatier than a 5-min demo, every act shows something V1 can't do.

---

## Tech Stack (All Open Source, Zero Infrastructure)

| Component | Tool | Why |
|-----------|------|-----|
| **UI** | Streamlit | Fast to build, looks polished, interactive widgets, native charts |
| **Embeddings** | `sentence-transformers` (all-MiniLM-L6-v2 or nomic-embed-text-v1.5) | Local, fast, no API key needed |
| **Macro clustering** | BERTopic (UMAP + HDBSCAN) on document-level summary embeddings | Same proven approach as micro, no per-doc LLM cost |
| **Macro cluster labeling** | OpenAI GPT-4o-mini | Label 5–10 macro clusters (not 500 docs) |
| **Micro clustering** | BERTopic (UMAP + HDBSCAN + c-TF-IDF) | Our actual production approach |
| **Cluster labeling** | OpenAI GPT-4o-mini | Auto-name clusters |
| **Field discovery** | OpenAI GPT-4o-mini | Discover fields from representative chunks |
| **Field extraction** | OpenAI GPT-4o-mini | Cluster-conditioned extraction |
| **BM25 search** | `rank_bm25` or `bm25s` | Lightweight, pip install, no server |
| **Dense search** | FAISS ( `faiss-cpu` ) | Fast ANN, pip install |
| **Cross-encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker, sentence-transformers |
| **PDF parsing** | `pymupdf` or `pdfplumber` | Extract text from uploaded PDFs |
| **Data storage** | SQLite | Zero setup, sufficient for demo |
| **Viz: Cluster scatter** | Plotly (via Streamlit) | Interactive UMAP scatter |
| **Viz: Ontology tree** | `streamlit-agraph` or Plotly Treemap | Interactive tree/graph view |
| **Viz: Score comparison** | Streamlit columns + metrics | Side-by-side dense vs BM25 vs combined |
| **Composite engine** | Custom Python (~200 LOC) | AST-based formula parser |

**Total dependencies:** One `pip install` command. No Docker, no cloud, no external databases.

---

## Project Structure

```
hackathon/
├── app.py                          # Streamlit main app (sidebar nav, page routing)
├── requirements.txt
├── config.py                       # API keys, model names, constants
├── data/
│   └── sample_agreements/          # Pre-loaded sample data
├── core/                           # Person A's domain
│   ├── __init__.py
│   ├── chunker.py                  # Text chunking (sliding window, ~256 tokens)
│   ├── embedder.py                 # Embedding wrapper (MiniLM or Nomic)
│   ├── macro_clusterer.py          # Embedding-based macro domain clustering (summary-level BERTopic)
│   ├── micro_clusterer.py          # BERTopic within a macro domain
│   ├── labeler.py                  # LLM cluster labeling
│   ├── field_discovery.py          # LLM field discovery from cluster samples
│   ├── extractor.py                # Cluster-conditioned LLM extraction
│   ├── generic_extractor.py        # V1-style generic extraction (for comparison)
│   ├── composite.py                # Composite field engine (DSL + DAG eval)
│   ├── retrieval.py                # Hybrid search (dense + BM25 + cross-encoder)
│   └── store.py                    # SQLite storage layer (all tables)
├── ui/                             # Person B's domain
│   ├── __init__.py
│   ├── upload_page.py              # File upload + macro domain clustering
│   ├── domain_page.py              # Macro domain explorer
│   ├── cluster_page.py             # Micro cluster visualization + ontology
│   ├── example_set_page.py         # Example set review → scale UX
│   ├── extraction_page.py          # Side-by-side: generic vs conditioned
│   ├── field_crud_page.py          # Rename, retype, add examples, create fields
│   ├── composite_page.py           # Composite field builder + cascade demo
│   ├── search_page.py              # Hybrid search with score breakdown
│   ├── ontology_page.py            # Full ontology tree view
│   └── dashboard_page.py           # Stats, quality metrics, correction counts
└── prompts/
    ├── macro_cluster_label.txt     # "Label this cluster of agreements..."
    ├── cluster_label.txt           # "Name this cluster of legal clauses..."
    ├── field_discovery.txt         # "What fields can be extracted from..."
    ├── field_extraction_conditioned.txt  # "This is a {type} clause. Extract..."
    └── field_extraction_generic.txt     # "Extract structured fields..." (V1-style, for comparison)
```

---

## Build Plan: Hour by Hour (2 People)

### Day 1: Core Pipeline + Basic UI Shell

| Block | Hours | Person A (Backend) | Person B (UI) |
|-------|-------|--------------------|---------------|
| **D1-1** | 0–1 | Project setup, requirements.txt, config, SQLite schema, sample data | Streamlit skeleton: sidebar nav, 8 empty pages, theme/styling |
| **D1-2** | 1–3 | `chunker.py` + `embedder.py` + `macro_clusterer.py` (summary embeddings → BERTopic → macro domains) | `upload_page.py` : file upload, progress bar, macro domain display |
| **D1-3** | 3–5 | `micro_clusterer.py` + `labeler.py` (BERTopic within domain) | `domain_page.py` : domain distribution chart + `cluster_page.py` : UMAP scatter |
| **D1-4** | 5–7 | `field_discovery.py` + `extractor.py` (cluster-conditioned) + `generic_extractor.py` | `example_set_page.py` : show 20 representative docs, editable table, "Scale" button |
| **D1-5** | 7–9 | `store.py` (SQLite: domains, clusters, fields, extractions, corrections, composites) | `extraction_page.py` : side-by-side generic vs conditioned, token count comparison |
| **D1-6** | 9–10 | `composite.py` (formula engine + DAG) | `field_crud_page.py` : rename, retype, add examples, create new field |

**Day 1 exit criteria:** Both people can run `streamlit run app.py` and see data flowing through: upload → macro clusters → micro clusters → fields → extractions. UI is rough but functional.

### Day 2: Search + Composites + Polish

| Block | Hours | Person A (Backend) | Person B (UI) |
|-------|-------|--------------------|---------------|
| **D2-1** | 0–2 | `retrieval.py` : hybrid search (dense + BM25 + cross-encoder + RRF) | `composite_page.py` : formula builder, computed values table, cascade animation |
| **D2-2** | 2–4 | Pre-compute all LLM responses for demo data (cache in SQLite) | `search_page.py` : search bar, score breakdown columns, result detail view |
| **D2-3** | 4–5 | Optimize: pre-build FAISS indices, BM25 indices, embeddings for sample data | `ontology_page.py` : interactive tree (domains → clause types → fields → composites) |
| **D2-4** | 5–7 | Edge cases: null handling in composites, empty clusters, missing fields | `dashboard_page.py` : cluster quality, extraction coverage, correction stats, system health |
| **D2-5** | 7–8 | **BOTH**: Integration testing — full demo flow end to end, fix any broken handoffs |
| **D2-6** | 8–10 | **BOTH**: Demo rehearsal — practice the 8-minute script, polish transitions, fix UX papercuts |

**Day 2 exit criteria:** Full demo runs end-to-end without crashes. Both people can present any section.

---

## SQLite Schema (Shared Contract Between A & B)

```sql
-- Macro domains
CREATE TABLE domains (
    domain_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,                  -- "SaaS Agreements"
    description TEXT,
    agreement_count INTEGER DEFAULT 0
);

-- Agreements
CREATE TABLE agreements (
    agreement_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    domain_id TEXT REFERENCES domains(domain_id),
    raw_text TEXT,
    domain_confidence REAL
);

-- Chunks
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    agreement_id TEXT REFERENCES agreements(agreement_id),
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER,
    embedding BLOB                        -- numpy array serialized
);

-- Micro clusters (within a domain)
CREATE TABLE clusters (
    cluster_id TEXT PRIMARY KEY,
    domain_id TEXT REFERENCES domains(domain_id),
    label TEXT,                           -- "Payment Terms"
    description TEXT,
    keywords TEXT,                        -- JSON array
    chunk_count INTEGER DEFAULT 0,
    quality_score REAL
);

-- Cluster-chunk assignments
CREATE TABLE cluster_assignments (
    chunk_id TEXT REFERENCES chunks(chunk_id),
    cluster_id TEXT REFERENCES clusters(cluster_id),
    probability REAL,
    PRIMARY KEY (chunk_id, cluster_id)
);

-- Field definitions (auto-discovered or user-created)
CREATE TABLE field_definitions (
    field_id TEXT PRIMARY KEY,
    cluster_id TEXT REFERENCES clusters(cluster_id),
    name TEXT NOT NULL,                   -- "payment_days"
    field_type TEXT NOT NULL,             -- "int", "string", "float", "date", "enum", "bool"
    description TEXT,
    source TEXT DEFAULT 'auto_discovered', -- "auto_discovered" | "user_created" | "user_modified"
    examples TEXT,                        -- JSON array of gold examples
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Extracted field values
CREATE TABLE extractions (
    extraction_id TEXT PRIMARY KEY,
    agreement_id TEXT REFERENCES agreements(agreement_id),
    field_id TEXT REFERENCES field_definitions(field_id),
    value TEXT,                           -- extracted value (stored as text, cast by field_type)
    confidence REAL,
    source_chunk_id TEXT,                 -- which chunk the value came from
    was_corrected BOOLEAN DEFAULT FALSE,
    original_value TEXT,                  -- value before user correction
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Composite field definitions
CREATE TABLE composite_definitions (
    field_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,                   -- "total_contract_value"
    expression TEXT NOT NULL,             -- "unit_price * quantity * term_years"
    source_field_ids TEXT NOT NULL,       -- JSON array of field_id references
    result_type TEXT DEFAULT 'float',
    cluster_id TEXT REFERENCES clusters(cluster_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Correction log (for feedback tracking)
CREATE TABLE corrections (
    correction_id TEXT PRIMARY KEY,
    extraction_id TEXT REFERENCES extractions(extraction_id),
    field_id TEXT REFERENCES field_definitions(field_id),
    old_value TEXT,
    new_value TEXT,
    corrected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Person A writes to these tables. Person B reads from them.** This is the integration contract — agree on it in hour 0.

---

## Requirements File

```
# requirements.txt
streamlit>=1.30.0
sentence-transformers>=2.2.0
bertopic>=0.16.0
umap-learn>=0.5.0
hdbscan>=0.8.33
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
openai>=1.12.0
pymupdf>=1.23.0
plotly>=5.18.0
pandas>=2.1.0
scikit-learn>=1.3.0
tiktoken>=0.5.0
streamlit-agraph>=0.0.45
```

---

## Key Code Blueprints

### macro_clusterer.py (NEW — embedding-based, replaces per-doc LLM classification)

```python
import json
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from openai import OpenAI

def get_document_summaries(agreements: list[dict]) -> list[str]:
    """Build a summary string per document: title + first ~500 tokens of body text."""
    summaries = []
    for a in agreements:
        title = a.get("title", a.get("filename", ""))
        text = a.get("raw_text", "")[:2000]  # ~500 tokens
        summaries.append(f"{title}\n\n{text}")
    return summaries

def cluster_into_domains(
    summaries: list[str],
    embed_model: SentenceTransformer,
    min_cluster_size: int = 10,
    min_samples: int = 3,
) -> tuple[list[int], BERTopic]:
    """Run BERTopic on document-level summary embeddings to produce macro domains.

    Returns (topic_ids_per_doc, topic_model).
    Typically produces 3–10 macro clusters from 100–500 docs.
    """
    embeddings = embed_model.encode(summaries, show_progress_bar=True)
    topic_model = BERTopic(
        min_topic_size=min_cluster_size,
        hdbscan_model={"min_cluster_size": min_cluster_size, "min_samples": min_samples},
        nr_topics="auto",
        calculate_probabilities=True,
    )
    topics, probs = topic_model.fit_transform(summaries, embeddings)
    return topics, topic_model

def label_macro_clusters(client: OpenAI, topic_model: BERTopic) -> dict[int, dict]:
    """Label each macro cluster using a single LLM call per cluster.

    With 5–10 clusters, this costs 5–10 LLM calls total (vs 100–500 for per-doc classification).
    """
    labels = {}
    for topic_id in topic_model.get_topic_info()["Topic"]:
        if topic_id == -1:
            labels[-1] = {"label": "Uncategorized", "description": "Documents not assigned to a domain"}
            continue
        keywords = [w for w, _ in topic_model.get_topic(topic_id)]
        representative_docs = topic_model.get_representative_docs(topic_id)
        prompt = f"""You are labeling a group of legal agreements that were automatically clustered together.

Keywords for this cluster: {', '.join(keywords[:10])}

Representative document excerpts (first 200 chars each):
{chr(10).join(f'- {doc[:200]}' for doc in representative_docs[:5])}

What type of agreement domain is this? Respond in JSON:
{{"label": "short name like SaaS, NDA, Employment, Vendor, Lease", "description": "one sentence describing this domain"}}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        labels[topic_id] = json.loads(resp.choices[0].message.content)
    return labels
```

**Cost comparison:**

| Approach | 100 docs | 500 docs | LLM calls |
|----------|----------|----------|-----------|
| Per-doc LLM classification (original plan) | ~$0.10 | ~$0.50 | 100–500 |
| Embedding-based clustering + cluster labeling (new plan) | ~$0.005 | ~$0.005 | 5–10 |
| **Savings** | **~95%** | **~99%** | **~95–98%** |

### extractor.py — Cluster-Conditioned (NEW — V1 uses generic prompts)

```python
CONDITIONED_PROMPT = """You are extracting structured data from a legal agreement section.

CLAUSE TYPE: {cluster_label}
CLAUSE DESCRIPTION: {cluster_description}
KNOWN FIELDS FOR THIS CLAUSE TYPE:
{field_definitions}

REPRESENTATIVE EXAMPLES FROM SIMILAR AGREEMENTS:
{examples}

AGREEMENT TEXT TO EXTRACT FROM:
{text}

For each field, extract the value from the text. If a field is not present, return null.
Include a confidence score (0-1) for each extraction.

Respond in JSON: {{"field_name": {{"value": ..., "confidence": 0.0-1.0}}, ...}}"""

GENERIC_PROMPT = """You are extracting structured data from a legal agreement section.

Extract any structured fields you can identify from this text.

Text: {text}

Respond in JSON: {{"field_name": {{"value": ..., "confidence": 0.0-1.0}}, ...}}"""
```

### composite.py

```python
import ast
import operator

SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

def parse_and_eval(expression: str, field_values: dict) -> float | None:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body, field_values)

def _eval_node(node, values):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        val = values.get(node.id)
        return float(val) if val is not None else None
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, values)
        right = _eval_node(node.right, values)
        if left is None or right is None:
            return None
        op_func = SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        if isinstance(node.op, ast.Div) and right == 0:
            return None
        return op_func(left, right)
    raise ValueError(f"Unsupported: {type(node).__name__}")

def get_dependencies(expression: str) -> list[str]:
    tree = ast.parse(expression, mode="eval")
    return [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]

def detect_cycles(composites: dict[str, str]) -> list[str]:
    """Return cycle errors. composites = {name: expression}"""
    visited, in_stack = set(), set()
    errors = []
    deps = {name: get_dependencies(expr) for name, expr in composites.items()}

    def dfs(node):
        if node in in_stack:
            errors.append(f"Cycle detected involving '{node}'")
            return
        if node in visited:
            return
        in_stack.add(node)
        for dep in deps.get(node, []):
            if dep in composites:
                dfs(dep)
        in_stack.discard(node)
        visited.add(node)

    for name in composites:
        dfs(name)
    return errors
```

### retrieval.py

```python
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class HybridRetriever:
    def __init__(self, chunks, embeddings, embed_fn):
        self.chunks = chunks
        self.embed_fn = embed_fn
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Dense index
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        normed = embeddings.copy()
        faiss.normalize_L2(normed)
        self.faiss_index.add(normed)

        # Sparse index
        tokenized = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20, use_reranker: bool = False):
        # Dense
        q_vec = self.embed_fn(query).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        dense_scores, dense_ids = self.faiss_index.search(q_vec, top_k)

        # Sparse
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top = np.argsort(bm25_scores)[-top_k:][::-1]

        # RRF merge
        rrf = {}
        for rank, idx in enumerate(dense_ids[0]):
            rrf[int(idx)] = rrf.get(int(idx), 0) + 1.0 / (60 + rank)
        for rank, idx in enumerate(bm25_top):
            rrf[int(idx)] = rrf.get(int(idx), 0) + 1.0 / (60 + rank)

        # Build results with individual scores
        results = []
        for idx, rrf_score in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            results.append({
                "chunk": self.chunks[idx], "idx": idx,
                "rrf_score": rrf_score,
                "dense_score": float(dense_scores[0][list(dense_ids[0]).index(idx)]) if idx in dense_ids[0] else 0,
                "bm25_score": float(bm25_scores[idx]),
            })

        # Optional cross-encoder reranking
        if use_reranker and results:
            pairs = [(query, r["chunk"]) for r in results]
            ce_scores = self.reranker.predict(pairs)
            for i, r in enumerate(results):
                r["cross_encoder_score"] = float(ce_scores[i])
            results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        return results
```

---

## UI Design Highlights (Person B Focus)

### Ontology Tree (the "wow" visual)

```
📊 Agreement Intelligence Ontology
│
├── 📁 SaaS/Software (142 agreements)
│   ├── 💼 Payment Terms (47 chunks)
│   │   ├── 📋 payment_days: int (coverage: 94%)
│   │   ├── 📋 payment_method: enum (coverage: 82%)
│   │   ├── 📋 late_fee_pct: float (coverage: 61%)
│   │   └── 🧮 TCV = unit_price × quantity × term_years
│   ├── 💼 Liability Limitation (38 chunks)
│   │   ├── 📋 liability_cap: float (coverage: 89%)
│   │   └── 📋 cap_type: enum (coverage: 76%)
│   └── 💼 Auto-Renewal (31 chunks)
│       ├── 📋 renewal_term: string (coverage: 91%)
│       ├── 📋 notice_period: string (coverage: 87%)
│       └── 🧮 last_opt_out = date_sub(expiration, notice_period)
├── 📁 NDA/Confidentiality (89 agreements)
│   ├── 💼 Confidentiality Obligations (52 chunks)
│   └── 💼 Term & Survival (37 chunks)
└── 📁 Employment (71 agreements)
    ├── 💼 Non-Compete (29 chunks)
    └── 💼 Compensation (42 chunks)
```

Use `streamlit-agraph` for an interactive node-edge graph, or Plotly Treemap for a color-coded area chart. The tree is the money shot — it shows the system "thinking."

### Side-by-Side Extraction Comparison (the "V2 is better" proof)

```
┌─────────────────────────────┬─────────────────────────────┐
│  ❌ V1: Generic Extraction    │  ✅ V2: Cluster-Conditioned  │
├─────────────────────────────┼─────────────────────────────┤
│  Prompt: "Extract fields"   │  Prompt: "This is a Payment │
│                             │  Terms clause. Extract:      │
│                             │  payment_days, method, fee"  │
├─────────────────────────────┼─────────────────────────────┤
│  Fields found: 11           │  Fields found: 4            │
│  ⚠️ 6 hallucinated          │  ✅ 0 hallucinated           │
│  Tokens used: 1,847         │  Tokens used: 1,102 (↓40%)  │
│  Time: 2.3s                 │  Time: 1.4s (↓39%)          │
├─────────────────────────────┼─────────────────────────────┤
│  payment_days: 30 ✅         │  payment_days: 30 ✅         │
│  payment_method: "wire" ✅   │  payment_method: "wire" ✅   │
│  late_fee: 1.5% ✅           │  late_fee: 1.5% ✅           │
│  currency: "USD" ✅          │  currency: "USD" ✅          │
│  contract_type: "SaaS" ❌    │                             │
│  effective_date: null ❌     │                             │
│  jurisdiction: null ❌       │                             │
│  party_name: "Acme" ❌       │                             │
│  total_value: "N/A" ❌       │                             │
│  renewal: null ❌            │                             │
│  termination: null ❌        │                             │
└─────────────────────────────┴─────────────────────────────┘
  "Same LLM. Same document. V2 knows what it's looking for."
```

### Composite Cascade Animation

When user corrects a base field, show the cascade in real time:

```
✏️ unit_price corrected: $50,000 → $55,000

  ↓ Propagating...

  🧮 annual_cost = unit_price × quantity
     = $55,000 × 100 = $5,500,000  (was $5,000,000)  ↑10%

  🧮 TCV = annual_cost × term_years
     = $5,500,000 × 3 = $16,500,000  (was $15,000,000)  ↑10%

  🧮 cost_per_month = annual_cost / 12
     = $5,500,000 / 12 = $458,333  (was $416,667)  ↑10%

  ✅ 1 correction → 3 composites updated → 0 AI calls
```

Use `st.status()` or `st.expander()` with a slight `time.sleep(0.5)` to create a cascade animation effect.

---

## Sample Data Strategy

**Use CUAD dataset** (already in this repo at `clustering_output/cuad_clustering_output/` ) for real contract credibility, plus **30 synthetic agreements** to guarantee clean demo scenarios:

| Synthetic | Count | Purpose |
|-----------|-------|---------|
| SaaS with explicit pricing | 10 | Ensures `unit_price` , `quantity` , `term_years` extractable for composite demo |
| NDAs with clear terms | 5 | Clean confidentiality + non-compete for domain separation demo |
| Vendor agreements with liability caps | 5 | "$5M" and "$10M" for BM25 exact-match demo |
| Employment with non-compete | 5 | Different domain with different fields |
| Leases with dates | 5 | Date math composite demo ( `last_opt_out_date` ) |

Generate synthetics with GPT-4o the night before. Takes ~20 minutes.

---

## Presentation Flow (8-Minute Demo)

| Act | Duration | What You Say | What You Show | Who Presents |
|-----|----------|-------------|---------------|-------------|
| Title | 15s | "V1 clusters agreements. V2 understands them." | Title slide | Either |
| Domain Intelligence | 60s | "We upload 300 contracts. The system clusters them by document structure — not one LLM call per doc, but smart embedding-based grouping. NDA indemnity ≠ SaaS indemnity." | Upload → domain bar chart → macro domains | Person A |
| Auto Ontology | 90s | "Within SaaS, it discovered 5 clause types. Each has its own fields. This ontology was built automatically." | UMAP scatter → click cluster → field schema → ontology tree | Person A |
| Example Set | 60s | "Before scaling extraction, users validate on 20 representative docs. Review, correct, then scale." | Example set table → corrections → "Scale" button | Person B |
| Conditioned Extraction | 45s | "Same LLM, same doc, but V2 tells the model what to look for. 40% fewer tokens, zero hallucinated fields." | Side-by-side comparison panel | Person B |
| Composite Fields | 60s | "Define TCV as a formula. One correction cascades everywhere. Zero AI cost." | Formula builder → correction → cascade animation | Person A |
| Hybrid Search | 60s | "Find liability caps over $5M. BM25 catches the exact dollar amount. Dense catches the semantic meaning. Cross-encoder picks the best." | Search → score breakdown → toggle reranker | Person B |
| Field CRUD | 45s | "Rename a field. Add examples. Create new fields. Users own their schema." | Quick CRUD operations | Person B |
| Big Picture | 30s | "300 contracts. 5 domains. 25 clause types. 100 fields. 10 computed metrics. All automatic. All compounding." | Dashboard + ontology tree | Both |

---

## Risk Mitigation for Demo Day

| Risk | Mitigation |
|------|-----------|
| LLM API is slow/down | Pre-cache ALL LLM responses in SQLite. Demo runs from cache by default. Toggle "live mode" only if API is fast. Macro clustering runs locally (embeddings + BERTopic) with zero LLM dependency. |
| Clustering produces bad topics | Pre-tune BERTopic params on sample data night before. Save the model. Load it. |
| Macro clustering misgroups docs | Pre-cluster all sample agreements. Store domain assignments in SQLite. Tune `min_cluster_size` for clean separation. |
| Side-by-side comparison doesn't show clear difference | Cherry-pick 3 agreements where generic extraction hallucinates badly. Use these as default demo docs. |
| Composite cascade isn't visually obvious | Add `time.sleep(0.3)` between each cascade step. Use green/red arrows. Make the numbers big. |
| "V1 already does most of this" | Have the comparison table ready: "Here are 10 things V2 does that V1 cannot." Lead with domain intelligence + conditioned extraction. |
| "How is this different from ChatGPT?" | "ChatGPT answers one question. This builds a permanent, compounding ontology across every contract." |
| "Does it scale?" | "This demo is local Streamlit. The production design uses AIDB, FAISS, cluster-partitioned indices. Same algorithms, production infrastructure." Point to the engineering doc. |

---

## Quick Start Commands

```bash
# Setup
mkdir hackathon && cd hackathon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-..."

# Prep sample data (night before)
python scripts/prep_demo_data.py    # Classifies, clusters, extracts, caches everything

# Run
streamlit run app.py --server.port 8501 --theme.base dark

# Open browser
open http://localhost:8501
```

---

## What Makes This Demo Win

01. **Everything is new.** Every act shows something V1 can't do. No "we rebuilt what we already have."
02. **Domain intelligence, not just clustering.** Macro → micro is the first thing the audience sees. It reframes the entire system.
03. **The side-by-side kills.** Showing V1-style generic extraction producing 6 hallucinated fields next to V2's clean conditioned extraction is undeniable.
04. **Composite cascade is the "aha" moment.** When one correction cascades to three fields with no AI cost, the audience understands compounding intelligence.
05. **Hybrid search scores are visible.** Not "trust us, it's better" — the audience sees Dense: 0.82, BM25: 0.95, Combined: 0.91 on every result.
06. **Users own their data.** CRUD on field definitions + example set review shows this is a platform, not a script.
07. **Two people built this in two days.** That's the unspoken signal: this team ships.

---

## Core V2 Deliverables: Required Capabilities (NOT Stretch Goals)

> **⚠️ THESE ARE THE PRIMARY HACKATHON DELIVERABLES.** The following three capabilities are core V2 differentiators that must be demonstrated in the hackathon and implemented in production. They appear in the Goal table above and are marked as REQUIRED in the Priority Matrix.

| Deliverable | JIRA | Effort | Why Critical |
|-------------|------|--------|--------------|
| **Clause-level embeddings** | FIELD-110 | 3-4h | V2 architecture foundation — no more averaging away critical signals |
| **Topic merging + synonyms** | FIELD-310 | 3-4h | Solves known V1 quality problem — duplicate clusters auto-merge |
| **Incremental assignment** | FIELD-801 | 2-3h | Avoids expensive re-clustering — new docs assigned in <500ms |

### Deliverable A: Clause-Level Semantic Embeddings (FIELD-110)

**JIRA Reference:** FIELD-110 — Rearchitect Input Signal: Clause-Level Semantic Embeddings
**Effort:** 3-4 hours
**Where:** New `core/clause_embedder.py` + upgrade `core/micro_clusterer.py`
**Owner:** Person A (Backend)

**Why this is required (not optional):**
- Field-summary averaging loses critical clause-specific signals
- V1's embedding approach is fundamentally limited — clauses get blurred together
- This is the architectural foundation for V2's superior clustering quality

**Implementation:**

```python
# In core/clause_embedder.py — clause-level embedding instead of field-summary averaging
from sentence_transformers import SentenceTransformer
import numpy as np

class ClauseEmbedder:
    """Generate clause-level embeddings that preserve semantic granularity.

    Key insight: Field summaries average away critical clause-specific signals.
    A liability clause and a payment clause get blurred when summarized together.
    Clause-level embeddings preserve these distinctions.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_clauses(self, chunks: list[dict]) -> np.ndarray:
        """Embed each clause/chunk independently.

        Args:
            chunks: List of {chunk_id, text, section_heading, ...}

        Returns:
            Embeddings matrix (n_chunks × embedding_dim)
        """
        texts = [c["text"] for c in chunks]
        return self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    def embed_with_context(self, chunks: list[dict], context_window: int = 1) -> np.ndarray:
        """Embed each clause with surrounding context for better semantic anchoring.

        This helps with short clauses that lack sufficient context on their own.
        """
        contextualized = []
        for i, chunk in enumerate(chunks):
            context_before = " ".join(c["text"] for c in chunks[max(0, i-context_window):i])
            context_after = " ".join(c["text"] for c in chunks[i+1:i+1+context_window])
            full_context = f"{context_before} {chunk['text']} {context_after}".strip()
            contextualized.append(full_context)
        return self.model.encode(contextualized, show_progress_bar=True, normalize_embeddings=True)
```

**Field processor perf improvements:**

```python
# In core/extractor.py — batch clause processing for performance
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BatchClauseExtractor:
    """Performance-optimized clause extraction.

    Key optimizations:
    1. Batch embeddings (vectorized, not per-clause)
    2. Parallel LLM calls with rate limiting
    3. Clause-type-aware prompt caching
    4. Early termination for high-confidence extractions
    """

    def __init__(self, max_concurrent_llm: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrent_llm)
        self.prompt_cache = {}  # clause_type -> compiled prompt

    async def extract_batch(self, clauses: list[dict], cluster_label: str) -> list[dict]:
        """Extract fields from multiple clauses in parallel."""
        # Compile prompt once per cluster type
        if cluster_label not in self.prompt_cache:
            self.prompt_cache[cluster_label] = self._compile_prompt(cluster_label)

        prompt = self.prompt_cache[cluster_label]

        # Parallel extraction with rate limiting
        tasks = [self._extract_single(clause, prompt) for clause in clauses]
        return await asyncio.gather(*tasks)

    async def _extract_single(self, clause: dict, prompt: str) -> dict:
        async with self.semaphore:
            # Early termination: if clause is too short, skip LLM
            if len(clause["text"]) < 50:
                return {"clause_id": clause["chunk_id"], "fields": {}, "confidence": 0.0}
            # ... extraction logic
```

**Demo impact:**
* Show embedding quality comparison: clause-level vs field-summary
* Demonstrate extraction speed improvement with batch processing
* "V1 averages your clauses together. V2 preserves every nuance."

**Metrics to capture:**
* Cluster purity improvement (clause-level vs summary-level)
* Extraction latency reduction (batch vs sequential)
* Token efficiency (clause-conditioned prompts vs generic)

---

### Deliverable B: Hierarchical Topic Merging & Synonym Resolution (FIELD-310)

**JIRA Reference:** FIELD-310 — Hierarchical Topic Merging & Synonym Resolution
**Effort:** 3-4 hours
**Where:** New `core/topic_merger.py` + upgrade `ui/ontology_page.py`
**Owner:** Person A (Backend) + Person B (UI)

**Why this is required (not optional):**
- BERTopic consistently produces duplicate/near-duplicate clusters
- Without merging, users see fragmented ontologies ("Payment Terms" vs "Net Payment Terms")
- Field synonyms across clusters create confusion and duplicate extraction work
- This directly addresses a known V1 quality problem

**Problem this solves:**
BERTopic often produces near-duplicate clusters:
- "Payment Terms" and "Net Payment Terms" as separate clusters
- "Liability Limitation" and "Limitation of Liability" as separate clusters
- "Confidentiality" and "Non-Disclosure" as separate clusters

These should merge automatically. Additionally, field synonyms across clusters should resolve:
- `payment_days`, `net_terms`, `payment_period` → canonical `payment_days`

**Implementation:**

```python
# In core/topic_merger.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class HierarchicalTopicMerger:
    """Merge semantically similar topics and resolve field synonyms.

    Uses a two-phase approach:
    1. Topic merging: Combine clusters with high centroid similarity
    2. Synonym resolution: Unify field names across merged clusters
    """

    def __init__(self, merge_threshold: float = 0.85, synonym_threshold: float = 0.80):
        self.merge_threshold = merge_threshold
        self.synonym_threshold = synonym_threshold

    def merge_similar_topics(
        self,
        clusters: list[dict],
        embeddings: dict[str, np.ndarray]  # cluster_id -> centroid embedding
    ) -> list[dict]:
        """Hierarchically merge clusters that are semantically equivalent.

        Returns merged cluster structure with parent-child relationships.
        """
        cluster_ids = list(embeddings.keys())
        centroids = np.array([embeddings[cid] for cid in cluster_ids])

        # Compute pairwise similarity
        sim_matrix = cosine_similarity(centroids)

        # Build merge groups using single-linkage clustering
        merged_groups = self._single_linkage_merge(cluster_ids, sim_matrix)

        # Create hierarchical structure
        merged_clusters = []
        for group in merged_groups:
            if len(group) == 1:
                # No merge needed
                merged_clusters.append(clusters[cluster_ids.index(group[0])])
            else:
                # Merge: pick best label, combine keywords, note aliases
                merged = self._merge_cluster_group(
                    [clusters[cluster_ids.index(cid)] for cid in group]
                )
                merged_clusters.append(merged)

        return merged_clusters

    def _single_linkage_merge(self, ids: list[str], sim_matrix: np.ndarray) -> list[list[str]]:
        """Single-linkage clustering to find merge groups."""
        n = len(ids)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Merge pairs above threshold
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.merge_threshold:
                    union(i, j)

        # Group by root
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(ids[i])

        return list(groups.values())

    def _merge_cluster_group(self, clusters: list[dict]) -> dict:
        """Merge multiple clusters into one, preserving provenance."""
        # Pick the cluster with highest quality score as primary
        primary = max(clusters, key=lambda c: c.get("quality_score", 0))

        # Combine keywords (deduplicated)
        all_keywords = []
        for c in clusters:
            all_keywords.extend(c.get("keywords", []))
        combined_keywords = list(dict.fromkeys(all_keywords))[:15]  # Top 15, deduped

        # Record aliases (other cluster labels)
        aliases = [c["label"] for c in clusters if c["label"] != primary["label"]]

        return {
            **primary,
            "keywords": combined_keywords,
            "aliases": aliases,
            "merged_from": [c["cluster_id"] for c in clusters],
            "chunk_count": sum(c.get("chunk_count", 0) for c in clusters),
        }

    def resolve_field_synonyms(
        self,
        fields: list[dict],
        embed_fn
    ) -> dict[str, str]:
        """Map field name variants to canonical names.

        Returns: {variant_name: canonical_name}
        """
        # Embed field names + definitions
        field_texts = [f"{f['name']}: {f.get('description', '')}" for f in fields]
        field_embeddings = embed_fn(field_texts)

        # Cluster similar fields
        sim_matrix = cosine_similarity(field_embeddings)

        # Find synonym groups
        synonym_groups = self._single_linkage_merge(
            [f["name"] for f in fields],
            sim_matrix
        )

        # Build mapping to canonical (most frequent) name
        mapping = {}
        for group in synonym_groups:
            if len(group) > 1:
                # Pick canonical as most common or first alphabetically
                canonical = min(group)  # Alphabetical for determinism
                for name in group:
                    mapping[name] = canonical

        return mapping


class SynonymResolver:
    """Resolve synonyms using multiple strategies."""

    # Common legal field synonyms (seed list)
    KNOWN_SYNONYMS = {
        "payment_days": ["net_terms", "payment_period", "days_to_pay", "payment_terms_days"],
        "effective_date": ["start_date", "commencement_date", "agreement_date"],
        "expiration_date": ["end_date", "termination_date", "term_end"],
        "governing_law": ["applicable_law", "choice_of_law", "jurisdiction"],
        "liability_cap": ["limitation_of_liability", "max_liability", "liability_limit"],
        "notice_period": ["notice_days", "termination_notice", "advance_notice"],
        "auto_renewal": ["auto_renew", "automatic_renewal", "evergreen"],
        "confidentiality_period": ["nda_term", "confidentiality_term", "secrecy_period"],
    }

    @classmethod
    def resolve(cls, field_name: str) -> str:
        """Resolve a field name to its canonical form."""
        normalized = field_name.lower().strip().replace(" ", "_").replace("-", "_")

        # Check known synonyms
        for canonical, synonyms in cls.KNOWN_SYNONYMS.items():
            if normalized == canonical or normalized in synonyms:
                return canonical

        return normalized
```

**UI Enhancement:**

```python
# In ui/ontology_page.py — show merge visualization
def render_merged_clusters(clusters: list[dict]):
    """Render ontology tree with merge indicators."""
    st.markdown("### 🌳 Ontology Tree (with auto-merged topics)")

    for cluster in clusters:
        if cluster.get("aliases"):
            # Show merged cluster with alias badges
            with st.expander(f"💼 {cluster['label']} (merged)", expanded=False):
                st.markdown(f"**Also known as:** {', '.join(cluster['aliases'])}")
                st.markdown(f"**Merged from:** {len(cluster.get('merged_from', []))} clusters")
                st.markdown(f"**Keywords:** {', '.join(cluster['keywords'][:10])}")

                # Show fields with synonym resolution
                st.markdown("**Fields:**")
                for field in cluster.get("fields", []):
                    canonical = SynonymResolver.resolve(field["name"])
                    if canonical != field["name"]:
                        st.markdown(f"  - 📋 {field['name']} → `{canonical}` (normalized)")
                    else:
                        st.markdown(f"  - 📋 {field['name']}")
        else:
            # Regular cluster
            with st.expander(f"💼 {cluster['label']}", expanded=False):
                st.markdown(f"**Keywords:** {', '.join(cluster['keywords'][:10])}")
```

**Demo impact:**
* Before/after comparison: "5 duplicate clusters → 3 clean clusters"
* Show synonym resolution: "payment_days and net_terms are now unified"
* "V1 creates fragmented clusters. V2 intelligently merges them."

**Metrics to capture:**
* Cluster count reduction (before vs after merge)
* Field deduplication rate
* Ontology coherence improvement

---

### Deliverable C: Incremental Cluster Assignment (FIELD-801)

**JIRA Reference:** FIELD-801 — Incremental Cluster Assignment Without Re-clustering
**Effort:** 2-3 hours
**Where:** New `core/cluster_assigner.py` + upgrade `core/store.py`
**Owner:** Person A (Backend)

**Why this is required (not optional):**
- Full re-clustering is expensive (compute + time) and can't run on every new document
- Users upload documents continuously; they need instant cluster assignment
- Without this, the system is batch-only and can't scale to real-time workflows
- This is critical for production viability

**How it works:**

```
New Document Arrives
        ↓
    Embed document (MiniLM, <100ms)
        ↓
    Query stored cluster centroids (cosine similarity)
        ↓
    ┌───────────────────────────────────────────┐
    │  max_sim > 0.85    │  0.6 < max_sim < 0.85  │  max_sim < 0.6
    │  HIGH CONFIDENCE    │  TENTATIVE             │  NOVEL
    └───────────────────────────────────────────┘
        ↓                       ↓                      ↓
    Assign to best        Assign + flag         Add to novel buffer
    cluster instantly     for review            (periodic re-cluster)
        ↓
    Run extraction with cluster's field schema
        ↓
    Update cluster centroid (running mean)
```

**Implementation:**

```python
# In core/cluster_assigner.py
import numpy as np
from sentence_transformers import SentenceTransformer

class IncrementalClusterAssigner:
    """
    Assign new documents to existing clusters without re-clustering.

    Key insight: Once clusters are formed, their centroids are stable.
    New docs can be assigned by finding the nearest centroid.
    Only trigger re-clustering when the novel buffer exceeds threshold.
    """

    def __init__(
        self,
        embed_model: SentenceTransformer,
        high_confidence_threshold: float = 0.85,
        tentative_threshold: float = 0.60,
        novel_buffer_size: int = 500,
    ):
        self.embed_model = embed_model
        self.high_confidence_threshold = high_confidence_threshold
        self.tentative_threshold = tentative_threshold
        self.novel_buffer_size = novel_buffer_size

        self.cluster_centroids: dict[str, np.ndarray] = {}  # cluster_id -> centroid
        self.cluster_counts: dict[str, int] = {}  # for running mean update
        self.novel_buffer: list[dict] = []

    def load_centroids(self, clusters: list[dict]):
        """Load cluster centroids from clustering results."""
        for cluster in clusters:
            self.cluster_centroids[cluster["cluster_id"]] = np.array(cluster["centroid"])
            self.cluster_counts[cluster["cluster_id"]] = cluster.get("chunk_count", 100)

    def assign_document(self, doc: dict) -> dict:
        """
        Assign a new document to the best matching cluster.

        Returns:
            {
                "cluster_id": str or None,
                "confidence": float,
                "assignment_type": "high_confidence" | "tentative" | "novel",
                "top_matches": [(cluster_id, similarity), ...]
            }
        """
        # Embed the document
        doc_embedding = self.embed_model.encode(
            doc["summary_text"],
            normalize_embeddings=True
        )

        # Find nearest centroids
        similarities = []
        for cluster_id, centroid in self.cluster_centroids.items():
            sim = np.dot(doc_embedding, centroid)
            similarities.append((cluster_id, float(sim)))

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:5]

        best_cluster_id, best_sim = top_matches[0] if top_matches else (None, 0.0)

        # Determine assignment type
        if best_sim >= self.high_confidence_threshold:
            assignment_type = "high_confidence"
            self._update_centroid(best_cluster_id, doc_embedding)
        elif best_sim >= self.tentative_threshold:
            assignment_type = "tentative"
            # Still assign, but flag for review
        else:
            assignment_type = "novel"
            best_cluster_id = None
            self.novel_buffer.append(doc)
            self._check_recluster_trigger()

        return {
            "cluster_id": best_cluster_id,
            "confidence": best_sim,
            "assignment_type": assignment_type,
            "top_matches": top_matches,
        }

    def _update_centroid(self, cluster_id: str, new_embedding: np.ndarray):
        """Update cluster centroid with running mean."""
        n = self.cluster_counts[cluster_id]
        old_centroid = self.cluster_centroids[cluster_id]
        # Running mean: new_centroid = (n * old + new) / (n + 1)
        new_centroid = (n * old_centroid + new_embedding) / (n + 1)
        # Re-normalize
        new_centroid = new_centroid / np.linalg.norm(new_centroid)
        self.cluster_centroids[cluster_id] = new_centroid
        self.cluster_counts[cluster_id] = n + 1

    def _check_recluster_trigger(self):
        """Check if novel buffer exceeds threshold for re-clustering."""
        if len(self.novel_buffer) >= self.novel_buffer_size:
            # In production: trigger async re-clustering job
            # For hackathon: just log a warning
            print(f"Novel buffer has {len(self.novel_buffer)} docs — re-clustering recommended")
```

**SQLite schema extension:**

```sql
-- Cluster centroids (for incremental assignment)
CREATE TABLE cluster_centroids (
    cluster_id TEXT PRIMARY KEY REFERENCES clusters(cluster_id),
    centroid BLOB NOT NULL,                -- numpy array serialized
    chunk_count INTEGER DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Novel document buffer
CREATE TABLE novel_documents (
    doc_id TEXT PRIMARY KEY,
    raw_text TEXT,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Demo impact:**
* Upload a new document after clustering is complete
* Show instant assignment: "This NDA was assigned to the Confidentiality cluster with 92% confidence"
* Show the top-5 candidate clusters with similarity scores
* "V1 would require re-clustering all 500 documents. V2 assigns in <500ms."

**Metrics to capture:**
* Assignment latency (target: <500ms)
* Assignment accuracy vs ground truth (target: ≥90%)
* Novel buffer growth rate

---

### Build Plan: Incorporating Core V2 Deliverables

**Day 1 — Include Clause-Level Embeddings (Person A focus):**

| Block | Hours | Task |
|-------|-------|------|
| **D1-2** | 1–3 | Implement `clause_embedder.py` with context window support |
| **D1-3** | 3–5 | Integrate clause embeddings into `micro_clusterer.py` |

**Day 2 — Include Topic Merging, Synonyms & Incremental Assignment (Both):**

| Block | Hours | Person A | Person B |
|-------|-------|----------|----------|
| **D2-1** | 0–2 | `topic_merger.py` + synonym resolution | Continue search UI |
| **D2-2** | 2–4 | Integrate merging into pipeline | `ontology_page.py` with merge visualization |
| **D2-3** | 4–6 | `cluster_assigner.py` + centroid storage | Demo page for "upload new doc" |
| **D2-4** | 6–8 | Integration testing | Polish + demo rehearsal |

---

### Priority Matrix

| Deliverable | Effort | Impact | Status |
|-------------|--------|--------|--------|
| **Clause-level embeddings (FIELD-110)** | 3-4h | **Critical — V2 architecture foundation** | **REQUIRED** |
| **Topic merging + synonyms (FIELD-310)** | 3-4h | **Critical — solves known V1 quality problem** | **REQUIRED** |
| **Incremental assignment (FIELD-801)** | 2-3h | **Critical — avoids expensive re-clustering** | **REQUIRED** |
| Kamradt chunking (FIELD-101) | 2-3h | High — better chunk boundaries | Stretch |
| Document TOC (FIELD-102) | 2-3h | Medium — nice-to-have | Stretch |
| Category labels (FIELD-702) | 1-2h | Medium — prettier ontology | Stretch |
| FIELD-401 + 402 bug fixes | 2h | Production fix | Pre-hackathon |

---

### Hackathon → Production Pipeline

Items prototyped in the hackathon feed directly into the FIELD-ONTOLOGY backlog:

```
Hackathon Prototype              →  Production Task
────────────────────────────────────────────────────
clause_embedder.py (REQUIRED)    →  FIELD-110 (clause-level embeddings)
topic_merger.py (REQUIRED)       →  FIELD-310 (topic merging & synonyms)
cluster_assigner.py (REQUIRED)   →  FIELD-801 (incremental cluster assignment)
macro_clusterer.py               →  FIELD-201 (hierarchical clustering)
retrieval.py (FAISS)             →  FIELD-202 (FAISS index infrastructure)
retrieval.py (BM25+RRF)          →  FIELD-501 (hybrid retrieval)
extractor.py (no HyDE)           →  FIELD-503 (HyDE removal validation)
example_set_page.py              →  FIELD-902 (review queue UX)
chunker.py (stretch: Kamradt)    →  FIELD-101 (production chunking)
labeler.py (stretch: categories) →  FIELD-702 (field taxonomy)
```

**After hackathon:** Hackathon code becomes the reference implementation for the first wave of FIELD-ONTOLOGY tasks. FIELD-110, FIELD-310, and FIELD-801 are **P0 critical** — they must ship in Phase 1–2.

---

## Stretch Goals: JIRA Backlog Items for Hackathon

> **Cross-reference:** See `cluster_v2_engineering.md` §21 for the full FIELD-ONTOLOGY backlog with 20 implementation tasks.

The hackathon plan already covers several JIRA tasks. Below are additional items that can be integrated **if time permits after the Core V2 Deliverables are complete**, plus quick wins that can be done in parallel (production code, not demo).

### Already Covered by Hackathon (No Extra Work)

These JIRA tasks are **validated or prototyped** by the core hackathon deliverables:

| JIRA Task | Hackathon Component | Validation |
|-----------|---------------------|------------|
| **FIELD-201** (Two-stage clustering) | `macro_clusterer.py` + `micro_clusterer.py` | Proves hierarchical architecture works at demo scale |
| **FIELD-202** (FAISS index) | `retrieval.py` FAISS dense search | Proves FAISS index approach is viable |
| **FIELD-501** (Hybrid BM25+dense retrieval) | `retrieval.py` with RRF fusion | Full prototype of hybrid retrieval — **Act 6** |
| **FIELD-503** (Remove HyDE dependency) | No HyDE used in hackathon | Validates raw queries work without HyDE |
| **FIELD-902** (Human review queue) | `example_set_page.py` | UX prototype for field review — **Act 3** |

### Stretch Goal 1: Kamradt Semantic Chunking (FIELD-101)

**JIRA Reference:** FIELD-101 — Replace naive text concatenation with Kamradt semantic chunking
**Effort:** +2-3 hours
**Where:** Upgrade `core/chunker.py`

**Current hackathon approach:** Sliding window chunking (~256 tokens, 50% overlap)

**Stretch upgrade:**

```python
# In core/chunker.py — replace sliding_window_chunk() with:
from semantic_chunkers import KamradtSemanticChunker

def semantic_chunk(text: str, embed_fn, min_chunk_chars: int = 250) -> list[dict]:
    """Chunk using Kamradt breakpoint detection on embedding similarity."""
    chunker = KamradtSemanticChunker(
        embed_fn=embed_fn,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )
    chunks = chunker.chunk(text)
    # Filter micro-fragments
    return [c for c in chunks if len(c["text"]) >= min_chunk_chars]
```

**Demo impact:**
* Better clause boundary detection → cleaner micro-clusters
* More impressive in Act 2: "The system preserves clause boundaries — no chunk spans two sections"

### Stretch Goal 2: Document Structure Tree (FIELD-102)

**JIRA Reference:** FIELD-102 — Build document structure tree from chunk metadata
**Effort:** +2-3 hours
**Where:** `core/chunker.py` + new `ui/doc_toc_page.py`

**Implementation:**

```python
# In core/chunker.py — add section header detection:
import re

SECTION_PATTERNS = [
    r'^(ARTICLE|Article)\s+[IVXLC\d]+[.:]\s*(.+)$',      # ARTICLE I: Definitions
    r'^(\d+\.)\s+([A-Z][^.]+)$',                          # 1. Definitions
    r'^(\d+\.\d+)\s+(.+)$',                               # 1.1 Purpose
    r'^(Section|SECTION)\s+\d+[.:]\s*(.+)$',              # Section 1: Terms
]

def extract_toc(text: str) -> list[dict]:
    """Extract section headings with hierarchy levels."""
    lines = text.split('\n')
    toc = []
    for i, line in enumerate(lines):
        for level, pattern in enumerate(SECTION_PATTERNS, start=1):
            if match := re.match(pattern, line.strip()):
                toc.append({
                    "heading": match.group(2) if match.lastindex >= 2 else match.group(1),
                    "level": level,
                    "line_number": i,
                })
                break
    return toc
```

**New UI page:** `ui/doc_toc_page.py` — shows extracted TOC alongside document text

**Demo impact:**
* Adds a "Document Structure" tab showing detected sections
* Makes ontology tree richer: Domain → Clause Type → **Document Sections** → Fields

### Stretch Goal 3: Auto Category Labels (FIELD-702)

**JIRA Reference:** FIELD-702 — Build hierarchical field taxonomy
**Effort:** +1-2 hours
**Where:** `core/labeler.py` + `ui/ontology_page.py`

**Implementation:**

```python
# In core/labeler.py — add category labeling:
def label_field_categories(client: OpenAI, fields: list[dict]) -> dict[str, str]:
    """Group fields into semantic categories using LLM."""
    field_names = [f["name"] for f in fields]
    prompt = f"""These are extracted fields from legal agreements:
{', '.join(field_names)}

Group them into 4-8 high-level categories (e.g., "Financial Terms", "Legal Provisions", "Dates & Timelines", "Parties & Roles").

Respond in JSON: {{"field_name": "category_name", ...}}"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)
```

**Demo impact:**
* Ontology tree gains a 4th level: **Category** → Domain → Clause Type → Fields
* Example: "Financial Terms" category groups Payment Terms, Pricing, Late Fees clusters
* More impressive in Act 8 (Big Picture): "Automatically organized into 8 semantic categories"

### Quick Wins (Production Code — Not Demo)

These FIELD-401 and FIELD-402 tasks are **production bug fixes** that should be done before/after hackathon (not part of the demo):

| JIRA | Task | Effort | When |
|------|------|--------|------|
| **FIELD-401** | Fix `representative_keywords=[]` bug in `schematize_handler.py` L692 | 1 hour | Pre-hackathon |
| **FIELD-402** | Switch `DEFAULT_SUBSET_STRATEGY` from `"top_membership"` to `"dpp"` | 1 hour | Pre-hackathon |

**Recommended:** Fix these in the main codebase on Day 0 (before hackathon starts) so the production pipeline also improves alongside the demo.

### Stretch Goals Decision Matrix

| Goal | Effort | Demo Impact | Recommend If... |
|------|--------|-------------|-----------------|
| Kamradt chunking (FIELD-101) | 2-3h | High — better clusters, cleaner extraction | Day 1 goes smoothly, spare 3h |
| Document TOC (FIELD-102) | 2-3h | Medium — adds a nice tab, not core demo | Both core features done early |
| Category labels (FIELD-702) | 1-2h | Medium — makes ontology tree prettier | Hackathon Day 2 morning, 1h free |
| FIELD-401 + 402 bug fixes | 2h | None (production only) | Always — do pre-hackathon |

### Updated Hour-by-Hour (If Adding Stretch Goals)

**Day 1 — Add FIELD-101 (Kamradt chunking) if ahead of schedule:**

| Block | Hours | Change |
|-------|-------|--------|
| D1-2 | 1–3 | Replace sliding window with Kamradt in `chunker.py` |
| D1-3 | 3–5 | Test that micro-clusters are cleaner with semantic chunks |

**Day 2 — Add FIELD-702 (Category labels) if ahead of schedule:**

| Block | Hours | Change |
|-------|-------|--------|
| D2-3 | 4–5 | Add `label_field_categories()` to labeler.py |
| D2-4 | 5–7 | Update ontology tree UI to show categories as top-level nodes |
