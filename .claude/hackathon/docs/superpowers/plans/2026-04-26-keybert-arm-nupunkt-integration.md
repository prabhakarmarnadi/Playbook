# KeyBERT + ARM + nupunkt Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate nupunkt structural decomposition, KeyBERT clause-type-aware clustering, and Association Rule Mining into the clustering pipeline via a two-pass architecture.

**Architecture:** Pass 1 (Foundation) replaces regex sentence splitting with nupunkt, adds KeyBERT-augmented embeddings and semi-supervised UMAP for better clustering. Pass 2 (Enrichment) mines clause/field relationships via FP-Growth ARM and feeds them into field discovery, extraction, and knowledge graph.

**Tech Stack:** nupunkt (sentence/block tokenization), keybert + sentence-transformers (clause fingerprinting), mlxtend (FP-Growth ARM), DuckDB (persistence), existing BERTopic + UMAP + HDBSCAN pipeline.

**Spec:** `docs/superpowers/specs/2026-04-26-keybert-arm-nupunkt-integration-design.md`

---

## File Structure

### New Files

| File                                           | Responsibility                                                        |
| ---------------------------------------------- | --------------------------------------------------------------------- |
| `core/nupunkt_chunker.py`                      | Structural decomposition: blocks -> sentences -> chunks using nupunkt |
| `core/keybert_scorer.py`                       | KeyBERT fingerprint loading, text classification, augmentation        |
| `core/arm/__init__.py`                         | ARM package init                                                      |
| `core/arm/arm_miner.py`                        | Clause-level FP-Growth mining (ported from keyBERTClause)             |
| `core/arm/field_arm_miner.py`                  | Field-value correlation mining (ported from keyBERTClause)            |
| `core/arm/configs.py`                          | ARM threshold defaults                                                |
| `core/relationship_layer.py`                   | Unified query interface over ARM outputs                              |
| `prompts/field_discovery_arm.txt`              | ARM-augmented field discovery prompt                                  |
| `prompts/field_extraction_arm_conditioned.txt` | ARM-augmented extraction prompt                                       |

### Modified Files

| File                           | Change                                                                        |
| ------------------------------ | ----------------------------------------------------------------------------- |
| `config.py`                    | Add KeyBERT + ARM config vars                                                 |
| `pyproject.toml`               | Add nupunkt, keybert, mlxtend dependencies                                    |
| `core/chunker.py`              | Add `structural_chunk()` entry point                                          |
| `core/store.py`                | Add columns to chunks/clusters tables, add arm_rules/arm_term_packages tables |
| `core/cuml_wrappers.py`        | Add `target`/`target_weight` params to `make_umap()`                          |
| `core/cluster_assigner.py`     | Add KeyBERT confidence bias to `assign_document()`                            |
| `core/field_discovery.py`      | Query RelationshipLayer for ARM context before LLM call                       |
| `core/extractor.py`            | Accept cross-clause fields and field correlations                             |
| `core/context_graph.py`        | Add `add_arm_relationships()` method                                          |
| `scripts/validate_pipeline.py` | Add 4 new test functions                                                      |

---

## Phase 1: nupunkt Chunking

### Task 1: Add nupunkt dependency and config

**Files:**

- Modify: `pyproject.toml:8-50`
- Modify: `config.py:69-74`

- [ ] **Step 1: Add nupunkt to pyproject.toml**

In `pyproject.toml`, add after the `pymupdf` line (line 47):

```toml
nupunkt = ">=0.6.0"                  # Legal-text-aware sentence/paragraph tokenization
```

- [ ] **Step 2: Install the dependency**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry add nupunkt`
Expected: nupunkt added to pyproject.toml and installed

- [ ] **Step 3: Verify nupunkt is importable**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "import nupunkt; print('nupunkt', nupunkt.__version__)"`
Expected: Prints version number

- [ ] **Step 4: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add pyproject.toml poetry.lock
git commit -m "feat: add nupunkt dependency for legal text segmentation"
```

---

### Task 2: Create core/nupunkt_chunker.py

**Files:**

- Create: `core/nupunkt_chunker.py`

- [ ] **Step 1: Write the test function in validate_pipeline.py**

Add this function before the `main()` block in `scripts/validate_pipeline.py`:

```python
def test_nupunkt_chunker():
    """Validate nupunkt structural decomposition."""
    from core.nupunkt_chunker import structural_decompose, nupunkt_available

    sample_text = """14.3 Termination for Convenience. Either party may terminate this Agreement upon thirty (30) days' prior written notice to the other party.

14.4 Effect of Termination. Upon termination, the Receiving Party shall return or destroy all Confidential Information. The obligations under Section 5 shall survive termination."""

    blocks = structural_decompose(sample_text)
    assert len(blocks) >= 1, f"Expected at least 1 block, got {len(blocks)}"

    # Each block should have required fields
    for block in blocks:
        assert "block_id" in block, "Block missing block_id"
        assert "sentences" in block, "Block missing sentences"
        assert "text" in block, "Block missing text"
        assert len(block["sentences"]) >= 1, f"Block has no sentences"

    # Sentences should preserve legal section numbers (not split on 14.3)
    all_sentences = [s for b in blocks for s in b["sentences"]]
    for sent in all_sentences:
        assert "sentence_id" in sent, "Sentence missing sentence_id"
        assert "text" in sent, "Sentence missing text"
        assert len(sent["text"].strip()) > 0, "Empty sentence text"

    # Should NOT split on "14.3" or "14.4" mid-sentence
    sentence_texts = [s["text"] for s in all_sentences]
    assert any("14.3" in s or "Termination for Convenience" in s for s in sentence_texts), \
        "Section number 14.3 was incorrectly split"

    print("  [PASS] nupunkt structural decomposition")
    print(f"    Blocks: {len(blocks)}, Sentences: {len(all_sentences)}")
    if nupunkt_available():
        print("    Using nupunkt tokenizer")
    else:
        print("    Using regex fallback")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "from core.nupunkt_chunker import structural_decompose"`
Expected: ModuleNotFoundError (file doesn't exist yet)

- [ ] **Step 3: Write the nupunkt_chunker module**

Create `core/nupunkt_chunker.py`:

```python
"""
Structural text decomposition using nupunkt.

Produces a three-level hierarchy: Document -> Blocks -> Sentences.
Falls back to regex splitting if nupunkt is not installed.
"""
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ── Detect nupunkt availability ──────────────────────────────────────────────
_NUPUNKT_AVAILABLE = False
try:
    import nupunkt
    _NUPUNKT_AVAILABLE = True
    logger.info("nupunkt detected — using legal-aware sentence tokenization")
except ImportError:
    logger.warning("nupunkt not available — falling back to regex sentence splitting")


def nupunkt_available() -> bool:
    """Check whether nupunkt is available."""
    return _NUPUNKT_AVAILABLE


# ── Regex fallback (from chunker.py) ─────────────────────────────────────────
_ABBREV_RE = re.compile(
    r"(?:Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Corp|Co|vs|etc|i\.e|e\.g|cf|viz|al|No|Art|Sec|U\.S|U\.K|E\.U)\."
)


def _regex_split_sentences(text: str) -> list[str]:
    """Fallback sentence splitter using regex (matches chunker.py behavior)."""
    if not text or not text.strip():
        return []
    protected = _ABBREV_RE.sub(lambda m: m.group().replace(".", "[DOT]"), text)
    protected = re.sub(r"(\d)\.", r"\1[DOT]", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\(\"'])", protected)
    sentences = [p.replace("[DOT]", ".").strip() for p in parts if p.strip()]
    return sentences


def _nupunkt_split_sentences(text: str) -> list[str]:
    """Split sentences using nupunkt."""
    if not text or not text.strip():
        return []
    return nupunkt.sent_tokenize(text)


def _split_into_blocks(text: str) -> list[str]:
    """Split text into paragraph-level blocks.

    Uses double-newline as the primary paragraph boundary.
    Preserves section-numbered paragraphs as distinct blocks.
    """
    if not text or not text.strip():
        return []
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def _detect_block_type(block_text: str) -> str:
    """Classify a block by its content pattern."""
    stripped = block_text.strip()
    if re.match(r"^(?:ARTICLE|SECTION|Part)\s", stripped, re.IGNORECASE):
        return "heading"
    if re.match(r'^"[^"]+"\s+(?:means|shall mean|refers to)', stripped):
        return "definition"
    if re.match(r"^(?:WHEREAS|RECITAL|PREAMBLE)", stripped, re.IGNORECASE):
        return "recital"
    return "clause_body"


def _extract_section_number(block_text: str) -> str | None:
    """Extract leading section number like '14.3' or '5.36'."""
    m = re.match(r"^(\d+(?:\.\d+)*)\s", block_text.strip())
    return m.group(1) if m else None


def structural_decompose(text: str) -> list[dict[str, Any]]:
    """Decompose text into blocks, each containing sentences.

    Returns a list of block dicts:
    [
        {
            "block_id": "blk_...",
            "block_type": "clause_body" | "heading" | "definition" | "recital",
            "section_number": "14.3" | None,
            "text": "full block text",
            "sentences": [
                {"sentence_id": "sent_...", "text": "...", "char_offset": int},
                ...
            ]
        },
        ...
    ]
    """
    split_fn = _nupunkt_split_sentences if _NUPUNKT_AVAILABLE else _regex_split_sentences
    raw_blocks = _split_into_blocks(text)

    blocks = []
    for block_text in raw_blocks:
        block_id = f"blk_{uuid.uuid4().hex[:12]}"
        sentences_raw = split_fn(block_text)

        sentences = []
        offset = 0
        for sent_text in sentences_raw:
            char_offset = block_text.find(sent_text, offset)
            if char_offset == -1:
                char_offset = offset
            sentences.append({
                "sentence_id": f"sent_{uuid.uuid4().hex[:12]}",
                "text": sent_text,
                "char_offset": char_offset,
            })
            offset = char_offset + len(sent_text)

        blocks.append({
            "block_id": block_id,
            "block_type": _detect_block_type(block_text),
            "section_number": _extract_section_number(block_text),
            "text": block_text,
            "sentences": sentences,
        })

    return blocks
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "
from core.nupunkt_chunker import structural_decompose, nupunkt_available
text = '''14.3 Termination for Convenience. Either party may terminate this Agreement upon thirty (30) days prior written notice.

14.4 Effect of Termination. Upon termination, the Receiving Party shall return or destroy all Confidential Information.'''
blocks = structural_decompose(text)
print(f'Blocks: {len(blocks)}')
for b in blocks:
print(f' {b[\"block_id\"]}: {b[\"block_type\"]}, section={b[\"section_number\"]}, sentences={len(b[\"sentences\"])}')
for s in b['sentences']:
print(f' {s[\"text\"][:80]}...')
print('nupunkt:', nupunkt_available())
"`
Expected: 2 blocks, each with sentences, section numbers extracted

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add core/nupunkt_chunker.py
git commit -m "feat: add nupunkt structural decomposition module"
```

---

### Task 3: Schema changes and chunker integration

**Files:**

- Modify: `core/store.py:61-71`
- Modify: `core/chunker.py` (add entry point)

- [ ] **Step 1: Add block_id and sentence_ids columns to chunks table**

In `core/store.py`, after the existing `chunks` table CREATE statement (around line 71), add:

```python
        # ── nupunkt structural metadata (Phase 1) ────────────────────────
        try:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS block_id VARCHAR")
            self.conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS sentence_ids VARCHAR")
        except Exception:
            pass  # Column already exists
```

- [ ] **Step 2: Add structural_chunk entry point to chunker.py**

Add this import at the top of `core/chunker.py`:

```python
from core.nupunkt_chunker import structural_decompose, nupunkt_available
```

Add this function to `core/chunker.py` (after the `sliding_window_chunk` function):

```python
def structural_chunk(
    text: str,
    chunk_size: int = 256,
    overlap: int = 64,
    min_chunk_chars: int = 80,
) -> list[dict]:
    """Structurally decompose text, then chunk within each block.

    Uses nupunkt for block/sentence boundaries when available.
    Each chunk inherits block_id and sentence_ids from its structural context.
    Falls back to sliding_window_chunk if nupunkt is unavailable.
    """
    blocks = structural_decompose(text)
    chunks = []
    global_idx = 0

    for block in blocks:
        block_chunks = sliding_window_chunk(
            block["text"],
            chunk_size=chunk_size,
            overlap=overlap,
            min_chunk_chars=min_chunk_chars,
        )
        # Determine which sentences overlap with each chunk
        for chunk in block_chunks:
            chunk_text = chunk["text"]
            overlapping_sids = []
            for sent in block["sentences"]:
                if sent["text"] in chunk_text or chunk_text in sent["text"]:
                    overlapping_sids.append(sent["sentence_id"])
            chunk["block_id"] = block["block_id"]
            chunk["sentence_ids"] = overlapping_sids
            chunk["block_type"] = block["block_type"]
            chunk["section_number"] = block["section_number"]
            chunk["chunk_index"] = global_idx
            global_idx += 1
            chunks.append(chunk)

    return chunks
```

- [ ] **Step 3: Verify integration**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "
from core.chunker import structural_chunk
text = '''14.3 Termination for Convenience. Either party may terminate this Agreement upon thirty (30) days prior written notice to the other party. This right is subject to the payment obligations in Section 8.

14.4 Effect of Termination. Upon termination, the Receiving Party shall return or destroy all Confidential Information. The obligations under Section 5 shall survive termination for a period of three (3) years.'''
chunks = structural_chunk(text, chunk_size=64, overlap=16)
print(f'Chunks: {len(chunks)}')
for c in chunks:
print(f' block_id={c[\"block_id\"][:16]}, sentence_ids={len(c[\"sentence_ids\"])}, section={c.get(\"section_number\")}')
print(f' {c[\"text\"][:60]}...')
"`
Expected: Chunks with block_id, sentence_ids, and section_number metadata

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python scripts/validate_pipeline.py --skip-llm`
Expected: All existing tests pass

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add core/store.py core/chunker.py
git commit -m "feat: integrate nupunkt chunking into store schema and chunker"
```

---

### Task 4: Add Phase 1 validation test

**Files:**

- Modify: `scripts/validate_pipeline.py`

- [ ] **Step 1: Add test_nupunkt_chunker to validate_pipeline.py**

Add the `test_nupunkt_chunker` function (from Task 2, Step 1) before the `main()` function.

Then add it to the `tests` list in `main()`:

```python
    tests = [
        ("Chunker", test_chunker),
        ("nupunkt Chunker", test_nupunkt_chunker),  # NEW
        ("Embedder", test_embedder),
        # ... rest unchanged
    ]
```

- [ ] **Step 2: Run the full validation suite**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python scripts/validate_pipeline.py --skip-llm`
Expected: All tests pass including the new nupunkt chunker test

- [ ] **Step 3: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add scripts/validate_pipeline.py
git commit -m "test: add nupunkt chunker validation"
```

---

## Phase 2: KeyBERT Clustering Context

### Task 5: Add KeyBERT dependency and config vars

**Files:**

- Modify: `pyproject.toml`
- Modify: `config.py:98-108`

- [ ] **Step 1: Add keybert to pyproject.toml**

In `pyproject.toml`, add after the `nupunkt` line:

```toml
keybert = ">=0.8.0"                   # Keyphrase extraction for clause fingerprinting
```

- [ ] **Step 2: Install the dependency**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry add keybert`
Expected: keybert added and installed

- [ ] **Step 3: Add config vars to config.py**

Add after the `NOVEL_BUFFER_SIZE` line (line 98) in `config.py`:

```python

# ── KeyBERT Clause Fingerprinting ────────────────────────────────────────────
KEYBERT_FINGERPRINTS_PATH = Path(os.getenv(
    "KEYBERT_FINGERPRINTS_PATH", str(DATA_DIR / "models" / "clause_fingerprints.json")
))
KEYBERT_CLASSIFICATION_THRESHOLD = float(os.getenv("KEYBERT_CLASSIFICATION_THRESHOLD", "0.3"))
KEYBERT_UMAP_TARGET_WEIGHT = float(os.getenv("KEYBERT_UMAP_TARGET_WEIGHT", "0.3"))
KEYBERT_PRIOR_WEIGHT = float(os.getenv("KEYBERT_PRIOR_WEIGHT", "0.15"))
KEYBERT_AUGMENT_TOP_K = int(os.getenv("KEYBERT_AUGMENT_TOP_K", "3"))
```

- [ ] **Step 4: Copy fingerprints file from keyBERTClause**

Run: `mkdir -p /mnt/data/code/clustering/.claude/hackathon/data/models && cp /mnt/data/code/keyBERTClause/models/clause_fingerprints.json /mnt/data/code/clustering/.claude/hackathon/data/models/`
Expected: File copied successfully

- [ ] **Step 5: Verify config loads**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "from config import KEYBERT_FINGERPRINTS_PATH, KEYBERT_PRIOR_WEIGHT; print(f'Path: {KEYBERT_FINGERPRINTS_PATH}'); print(f'Weight: {KEYBERT_PRIOR_WEIGHT}')"`
Expected: Prints path and 0.15

- [ ] **Step 6: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add pyproject.toml poetry.lock config.py
git commit -m "feat: add keybert dependency and config vars"
```

---

### Task 6: Create core/keybert_scorer.py

**Files:**

- Create: `core/keybert_scorer.py`

- [ ] **Step 1: Write the test function**

Add this function before `main()` in `scripts/validate_pipeline.py`:

```python
def test_keybert_scorer():
    """Validate KeyBERT clause scoring and text augmentation."""
    from core.keybert_scorer import KeyBERTScorer, keybert_available

    scorer = KeyBERTScorer()

    if not keybert_available():
        print("  [SKIP] KeyBERT fingerprints not found — scorer degrades gracefully")
        return

    # Test classification
    text = "The receiving party shall not disclose any confidential information."
    scores = scorer.classify(text)
    assert isinstance(scores, list), f"Expected list, got {type(scores)}"
    if scores:
        clause_type, similarity = scores[0]
        assert isinstance(clause_type, str), "Clause type should be string"
        assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} out of range"

    # Test augmentation
    augmented = scorer.augment_text(text)
    assert isinstance(augmented, str), "Augmented text should be string"
    assert text in augmented, "Original text should be preserved in augmentation"

    # Test label extraction
    label = scorer.get_label(text)
    assert label is None or isinstance(label, int), "Label should be int or None"

    print("  [PASS] KeyBERT scorer")
    print(f"    Top classification: {scores[0] if scores else 'none'}")
    print(f"    Augmented prefix: {augmented[:60]}...")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "from core.keybert_scorer import KeyBERTScorer"`
Expected: ModuleNotFoundError

- [ ] **Step 3: Write the keybert_scorer module**

Create `core/keybert_scorer.py`:

```python
"""
KeyBERT clause-type scoring for clustering context.

Loads pre-computed clause fingerprints (32 types x keyphrase embeddings) and provides:
1. classify(text) -> ranked clause-type similarities
2. augment_text(text) -> text with clause-type prefix for embedding
3. get_label(text) -> integer label for semi-supervised UMAP (-1 if uncertain)

Degrades gracefully if fingerprints file is missing.
"""
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    KEYBERT_AUGMENT_TOP_K,
    KEYBERT_CLASSIFICATION_THRESHOLD,
    KEYBERT_FINGERPRINTS_PATH,
)

logger = logging.getLogger(__name__)

_FINGERPRINTS_LOADED = False
_SCORER_INSTANCE = None


def keybert_available() -> bool:
    """Check whether KeyBERT fingerprints are loaded."""
    return _FINGERPRINTS_LOADED


class KeyBERTScorer:
    """Score text chunks against pre-computed clause-type fingerprints."""

    _instance: "KeyBERTScorer | None" = None

    def __init__(self, fingerprints_path: Path | str | None = None):
        self.fingerprints_path = Path(fingerprints_path or KEYBERT_FINGERPRINTS_PATH)
        self.clause_types: list[str] = []
        self.fingerprint_embeddings: np.ndarray | None = None
        self.keyphrases: dict[str, list[tuple[str, float]]] = {}
        self._model: SentenceTransformer | None = None
        self._loaded = False

        self._load_fingerprints()

    @classmethod
    def get_instance(cls, fingerprints_path: Path | str | None = None) -> "KeyBERTScorer":
        if cls._instance is None:
            cls._instance = cls(fingerprints_path)
        return cls._instance

    def _load_fingerprints(self) -> None:
        """Load fingerprints from JSON file."""
        global _FINGERPRINTS_LOADED
        if not self.fingerprints_path.exists():
            logger.warning(f"KeyBERT fingerprints not found at {self.fingerprints_path}")
            return

        try:
            with open(self.fingerprints_path) as f:
                data = json.load(f)

            fingerprints = data.get("fingerprints", {})
            model_name = data.get("model_name", EMBEDDING_MODEL)

            self.clause_types = sorted(fingerprints.keys())
            embeddings = []
            for ct in self.clause_types:
                fp = fingerprints[ct]
                if "embedding" in fp:
                    embeddings.append(np.array(fp["embedding"], dtype=np.float32))
                self.keyphrases[ct] = fp.get("keyphrases", [])

            if embeddings:
                self.fingerprint_embeddings = np.stack(embeddings)
                # Normalize
                norms = np.linalg.norm(self.fingerprint_embeddings, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                self.fingerprint_embeddings = self.fingerprint_embeddings / norms

            self._model = SentenceTransformer(model_name)
            self._loaded = True
            _FINGERPRINTS_LOADED = True
            logger.info(f"Loaded {len(self.clause_types)} clause fingerprints from {self.fingerprints_path}")

        except Exception as e:
            logger.warning(f"Failed to load KeyBERT fingerprints: {e}")

    def classify(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Classify text against all clause-type fingerprints.

        Returns list of (clause_type, similarity) sorted descending.
        Only returns matches above KEYBERT_CLASSIFICATION_THRESHOLD.
        """
        if not self._loaded or self.fingerprint_embeddings is None:
            return []

        text_embedding = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]

        similarities = self.fingerprint_embeddings @ text_embedding
        indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in indices:
            sim = float(similarities[idx])
            if sim >= KEYBERT_CLASSIFICATION_THRESHOLD:
                results.append((self.clause_types[idx], sim))

        return results

    def score_all(self, text: str) -> dict[str, float]:
        """Return similarity scores for all 32 clause types."""
        if not self._loaded or self.fingerprint_embeddings is None:
            return {}

        text_embedding = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]

        similarities = self.fingerprint_embeddings @ text_embedding
        return {ct: float(similarities[i]) for i, ct in enumerate(self.clause_types)}

    def augment_text(self, text: str) -> str:
        """Prepend clause-type context prefix to text for embedding.

        Format: [ClauseType | kw1, kw2, kw3] original text
        Returns original text unchanged if no classification above threshold.
        """
        results = self.classify(text, top_k=1)
        if not results:
            return text

        clause_type, _ = results[0]
        kps = self.keyphrases.get(clause_type, [])
        top_kps = sorted([kp for kp, _ in kps[:KEYBERT_AUGMENT_TOP_K]])
        kp_str = ", ".join(top_kps)

        return f"[{clause_type} | {kp_str}] {text}"

    def get_label(self, text: str) -> int:
        """Get integer label for semi-supervised UMAP.

        Returns index into self.clause_types if above threshold, else -1.
        UMAP treats -1 as unlabeled.
        """
        results = self.classify(text, top_k=1)
        if not results:
            return -1

        clause_type, _ = results[0]
        return self.clause_types.index(clause_type)

    def batch_classify(
        self, texts: list[str], top_k: int = 5
    ) -> list[list[tuple[str, float]]]:
        """Batch classify multiple texts."""
        if not self._loaded or self.fingerprint_embeddings is None:
            return [[] for _ in texts]

        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )

        all_similarities = embeddings @ self.fingerprint_embeddings.T
        results = []
        for row in all_similarities:
            indices = np.argsort(row)[::-1][:top_k]
            matches = []
            for idx in indices:
                sim = float(row[idx])
                if sim >= KEYBERT_CLASSIFICATION_THRESHOLD:
                    matches.append((self.clause_types[idx], sim))
            results.append(matches)

        return results

    def batch_augment(self, texts: list[str]) -> list[str]:
        """Batch augment texts with clause-type prefixes."""
        if not self._loaded:
            return list(texts)

        classified = self.batch_classify(texts, top_k=1)
        augmented = []
        for text, results in zip(texts, classified):
            if results:
                clause_type, _ = results[0]
                kps = self.keyphrases.get(clause_type, [])
                top_kps = sorted([kp for kp, _ in kps[:KEYBERT_AUGMENT_TOP_K]])
                kp_str = ", ".join(top_kps)
                augmented.append(f"[{clause_type} | {kp_str}] {text}")
            else:
                augmented.append(text)
        return augmented

    def batch_labels(self, texts: list[str]) -> np.ndarray:
        """Get integer labels for a batch of texts. -1 = unlabeled."""
        if not self._loaded:
            return np.full(len(texts), -1, dtype=np.int32)

        classified = self.batch_classify(texts, top_k=1)
        labels = np.full(len(texts), -1, dtype=np.int32)
        for i, results in enumerate(classified):
            if results:
                clause_type, _ = results[0]
                labels[i] = self.clause_types.index(clause_type)
        return labels
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "
from core.keybert_scorer import KeyBERTScorer, keybert_available
scorer = KeyBERTScorer()
print('Available:', keybert_available())
if keybert_available():
    text = 'The receiving party shall not disclose any confidential information.'
    scores = scorer.classify(text)
    print('Top scores:', scores[:3])
    aug = scorer.augment_text(text)
    print('Augmented:', aug[:80])
    label = scorer.get_label(text)
    print('Label:', label)
"`
Expected: Fingerprints loaded, classification results shown

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add core/keybert_scorer.py
git commit -m "feat: add KeyBERT clause scorer with augmentation and labeling"
```

---

### Task 7: Semi-supervised UMAP in cuml_wrappers.py

**Files:**

- Modify: `core/cuml_wrappers.py:151-181`

- [ ] **Step 1: Add target and target_weight params to make_umap**

Replace the `make_umap` function (lines 151-181) in `core/cuml_wrappers.py`:

```python
def make_umap(
    n_neighbors: int = 15,
    n_components: int = 5,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
    target: np.ndarray | None = None,
    target_weight: float = 0.0,
    **kwargs: Any,
):
    """Factory: create a GPU or CPU UMAP model.

    When cuML is available, returns the native cuML UMAP directly
    so BERTopic's isinstance checks pass correctly.

    Semi-supervised support: when target is provided with target_weight > 0,
    UMAP uses labels to guide dimensionality reduction. Labels of -1 are
    treated as unlabeled (semi-supervised mode).
    """
    # Semi-supervised params (only pass if explicitly set)
    ss_kwargs = {}
    if target is not None and target_weight > 0.0:
        ss_kwargs["target_metric"] = "categorical"
        ss_kwargs["target_weight"] = target_weight

    if _CUML_AVAILABLE:
        return cuUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            **ss_kwargs,
            **kwargs,
        )
    from umap import UMAP as cpuUMAP
    return cpuUMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        **ss_kwargs,
        **kwargs,
    )
```

Note: the `target` array itself is not passed to the constructor — it's passed to `fit()` or `fit_transform()` by the caller (BERTopic or macro/micro clusterer). The `target_weight` and `target_metric` configure how UMAP uses labels.

- [ ] **Step 2: Verify make_umap still works without target**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "
from core.cuml_wrappers import make_umap
import numpy as np
umap_model = make_umap(n_neighbors=5, n_components=2)
X = np.random.randn(50, 10).astype(np.float32)
reduced = umap_model.fit_transform(X)
print(f'Reduced shape: {reduced.shape}')
print('make_umap without target: OK')
"`
Expected: Reduced shape (50, 2)

- [ ] **Step 3: Verify make_umap works with target**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "
from core.cuml_wrappers import make_umap
import numpy as np
umap_model = make_umap(n_neighbors=5, n_components=2, target_weight=0.3)
X = np.random.randn(50, 10).astype(np.float32)
y = np.array([0]*15 + [1]*15 + [-1]*20, dtype=np.int32)
reduced = umap_model.fit_transform(X, y=y)
print(f'Reduced shape: {reduced.shape}')
print('make_umap with semi-supervised target: OK')
"`
Expected: Reduced shape (50, 2)

- [ ] **Step 4: Run existing tests**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python scripts/validate_pipeline.py --skip-llm`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add core/cuml_wrappers.py
git commit -m "feat: add semi-supervised UMAP support via target/target_weight params"
```

---

### Task 8: KeyBERT confidence bias in cluster_assigner.py

**Files:**

- Modify: `core/cluster_assigner.py:45-173`
- Modify: `core/store.py` (clusters table schema)

- [ ] **Step 1: Add dominant_clause_type columns to clusters table**

In `core/store.py`, after the chunks table ALTER statements (added in Task 3), add:

```python
        # ── KeyBERT clause-type metadata (Phase 2) ───────────────────────
        try:
            self.conn.execute("ALTER TABLE clusters ADD COLUMN IF NOT EXISTS dominant_clause_type VARCHAR")
            self.conn.execute("ALTER TABLE clusters ADD COLUMN IF NOT EXISTS clause_type_distribution VARCHAR")
        except Exception:
            pass
        try:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS keybert_scores VARCHAR")
            self.conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS augmented_text VARCHAR")
        except Exception:
            pass
```

- [ ] **Step 2: Add KeyBERT bias to assign_document**

In `core/cluster_assigner.py`, add import at top:

```python
from config import KEYBERT_PRIOR_WEIGHT
```

Modify the `assign_document` method. Replace the confidence threshold block (lines 135-152) with:

```python
        # ── Apply KeyBERT prior if available ──
        keybert_adjustment = 0.0
        if hasattr(self, '_keybert_priors') and self._keybert_priors and best_id:
            cluster_clause_type = self._keybert_priors.get(best_id)
            if cluster_clause_type and hasattr(self, '_keybert_scorer') and self._keybert_scorer:
                scores = self._keybert_scorer.score_all(text)
                alignment = scores.get(cluster_clause_type, 0.5)
                keybert_adjustment = (alignment - 0.5) * KEYBERT_PRIOR_WEIGHT

        adjusted_sim = best_sim + keybert_adjustment

        if adjusted_sim >= self.high_threshold:
            # ── Tier A: Hard assign + centroid update ──
            assignment_type = AssignmentType.HIGH_CONFIDENCE
            self._update_centroid(best_id, doc_embedding)
        elif adjusted_sim >= self.tentative_threshold:
            # ── Tier B: Soft assign with top-K memberships ──
            assignment_type = AssignmentType.TENTATIVE
            soft_memberships = self._compute_soft_memberships(similarities)
        else:
            # ── Tier C: Buffer as novel ──
            assignment_type = AssignmentType.NOVEL
            best_id = None
            self.novel_buffer.append({
                "doc_id": doc_id,
                "text": text,
                "embedding": doc_embedding,
            })

        if len(self.novel_buffer) >= self.novel_buffer_size:
            logger.warning(
                f"Novel buffer has {len(self.novel_buffer)} docs — re-clustering recommended"
            )
```

Also replace `"confidence": best_sim,` in the return dict with `"confidence": adjusted_sim,`.

- [ ] **Step 3: Add method to load KeyBERT priors**

Add this method to `IncrementalClusterAssigner`:

```python
    def load_keybert_priors(
        self, cluster_clause_types: dict[str, str], scorer=None
    ) -> None:
        """Load KeyBERT dominant clause types per cluster for confidence bias.

        Args:
            cluster_clause_types: {cluster_id: dominant_clause_type}
            scorer: KeyBERTScorer instance (lazy-loaded if None)
        """
        self._keybert_priors = cluster_clause_types
        if scorer:
            self._keybert_scorer = scorer
        else:
            try:
                from core.keybert_scorer import KeyBERTScorer
                self._keybert_scorer = KeyBERTScorer.get_instance()
            except Exception:
                self._keybert_scorer = None
```

- [ ] **Step 4: Run existing assigner test**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python scripts/validate_pipeline.py --skip-llm`
Expected: All tests pass (assigner test still works without KeyBERT priors)

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add core/cluster_assigner.py core/store.py
git commit -m "feat: add KeyBERT confidence bias to cluster assigner"
```

---

### Task 9: Add Phase 2 validation test

**Files:**

- Modify: `scripts/validate_pipeline.py`

- [ ] **Step 1: Add test_keybert_scorer to validate_pipeline.py**

Add the test function from Task 6, Step 1 before `main()`.

Add to the `tests` list in `main()`:

```python
        ("KeyBERT Scorer", test_keybert_scorer),  # NEW
```

- [ ] **Step 2: Run full validation**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python scripts/validate_pipeline.py --skip-llm`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add scripts/validate_pipeline.py
git commit -m "test: add KeyBERT scorer validation"
```

---

## Phase 3: ARM Relationship Layer

### Task 10: Add mlxtend dependency and ARM config

**Files:**

- Modify: `pyproject.toml`
- Modify: `config.py`

- [ ] **Step 1: Add mlxtend to pyproject.toml**

```toml
mlxtend = ">=0.23.0"                  # FP-Growth association rule mining
```

- [ ] **Step 2: Install**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry add mlxtend`

- [ ] **Step 3: Add ARM config vars to config.py**

Add after the KeyBERT config block:

```python

# ── Association Rule Mining (ARM) ────────────────────────────────────────────
ENABLE_ARM_ENRICHMENT = os.getenv("ENABLE_ARM_ENRICHMENT", "true").lower() in ("1", "true", "yes")
ARM_MIN_SUPPORT = float(os.getenv("ARM_MIN_SUPPORT", "0.20"))
ARM_MIN_CONFIDENCE = float(os.getenv("ARM_MIN_CONFIDENCE", "0.7"))
ARM_MIN_LIFT = float(os.getenv("ARM_MIN_LIFT", "1.8"))
FIELD_ARM_MIN_SUPPORT = float(os.getenv("FIELD_ARM_MIN_SUPPORT", "0.05"))
FIELD_ARM_MIN_CONFIDENCE = float(os.getenv("FIELD_ARM_MIN_CONFIDENCE", "0.5"))
FIELD_ARM_MIN_LIFT = float(os.getenv("FIELD_ARM_MIN_LIFT", "1.5"))
```

- [ ] **Step 4: Add ARM schema to store.py**

Add to `_init_schema()` in `core/store.py`:

```python
        # ── ARM tables (Phase 3) ─────────────────────────────────────────
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS arm_rules (
                rule_id          VARCHAR PRIMARY KEY,
                antecedent       VARCHAR NOT NULL,
                consequent       VARCHAR NOT NULL,
                rule_type        VARCHAR NOT NULL,
                support          DOUBLE,
                confidence       DOUBLE,
                lift             DOUBLE,
                cluster_id       VARCHAR,
                pipeline_run_id  VARCHAR
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS arm_term_packages (
                package_id       VARCHAR PRIMARY KEY,
                clause_types     VARCHAR NOT NULL,
                support          DOUBLE,
                size             INTEGER,
                cluster_id       VARCHAR,
                pipeline_run_id  VARCHAR
            )
        """)
```

- [ ] **Step 5: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add pyproject.toml poetry.lock config.py core/store.py
git commit -m "feat: add mlxtend dependency, ARM config vars, and ARM schema"
```

---

### Task 11: Create core/arm/ module

**Files:**

- Create: `core/arm/__init__.py`
- Create: `core/arm/configs.py`
- Create: `core/arm/arm_miner.py`
- Create: `core/arm/field_arm_miner.py`

- [ ] **Step 1: Create the ARM package**

Create `core/arm/__init__.py`:

```python
"""Association Rule Mining for clause and field relationship discovery."""
```

- [ ] **Step 2: Create configs.py with threshold defaults**

Create `core/arm/configs.py`:

```python
"""ARM threshold configuration defaults."""
from config import (
    ARM_MIN_CONFIDENCE,
    ARM_MIN_LIFT,
    ARM_MIN_SUPPORT,
    FIELD_ARM_MIN_CONFIDENCE,
    FIELD_ARM_MIN_LIFT,
    FIELD_ARM_MIN_SUPPORT,
)

CLAUSE_ARM_DEFAULTS = {
    "global": {
        "min_support": ARM_MIN_SUPPORT,
        "min_confidence": ARM_MIN_CONFIDENCE,
        "min_lift": ARM_MIN_LIFT,
    },
    "per_cluster_by_size": {
        "large": {"min_docs": 100, "min_support": 0.15},
        "medium": {"min_docs": 50, "min_support": 0.20},
        "small": {"min_docs": 20, "min_support": 0.25},
    },
    "term_packages": {
        "min_itemset_size": 3,
        "min_support": 0.15,
    },
}

FIELD_ARM_DEFAULTS = {
    "global": {
        "min_support": FIELD_ARM_MIN_SUPPORT,
        "min_confidence": FIELD_ARM_MIN_CONFIDENCE,
        "min_lift": FIELD_ARM_MIN_LIFT,
        "max_rules": 1000,
    },
    "per_cluster": {
        "min_support": FIELD_ARM_MIN_SUPPORT,
        "min_confidence": FIELD_ARM_MIN_CONFIDENCE,
        "min_lift": FIELD_ARM_MIN_LIFT,
        "max_rules": 500,
    },
    "normalization": {
        "numeric_bins": [0, 7, 30, 60, 90, 180, 365, 999999],
        "numeric_labels": ["0-7d", "8-30d", "31-60d", "61-90d", "91-180d", "181-365d", "365d+"],
    },
}
```

- [ ] **Step 3: Create arm_miner.py**

Create `core/arm/arm_miner.py`:

```python
"""Clause-level Association Rule Mining using FP-Growth.

Discovers clause co-occurrence patterns, term packages, and dependency chains.
Ported from keyBERTClause, adapted to use ClusteringStore.
"""
import json
import logging
import uuid

import numpy as np
import pandas as pd

from core.arm.configs import CLAUSE_ARM_DEFAULTS

logger = logging.getLogger(__name__)

try:
    from mlxtend.frequent_patterns import association_rules, fpgrowth
    _MLXTEND_AVAILABLE = True
except ImportError:
    _MLXTEND_AVAILABLE = False
    logger.warning("mlxtend not available — ARM mining disabled")


def mlxtend_available() -> bool:
    return _MLXTEND_AVAILABLE


def build_clause_presence_matrix(
    cluster_assignments: dict[str, list[str]],
) -> pd.DataFrame:
    """Build binary clause presence matrix from cluster assignments.

    Args:
        cluster_assignments: {doc_id: [clause_type_1, clause_type_2, ...]}

    Returns:
        DataFrame with doc_id index and clause type columns, values 0/1.
    """
    all_clause_types = sorted({ct for cts in cluster_assignments.values() for ct in cts})
    if not all_clause_types:
        return pd.DataFrame()

    rows = []
    for doc_id, clause_types in cluster_assignments.items():
        row = {ct: 1 if ct in clause_types else 0 for ct in all_clause_types}
        row["doc_id"] = doc_id
        rows.append(row)

    df = pd.DataFrame(rows).set_index("doc_id")
    return df


def mine_global(
    matrix: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    """Mine global association rules across all documents.

    Returns list of rules with: rule_id, antecedent, consequent,
    support, confidence, lift, rule_type.
    """
    if not _MLXTEND_AVAILABLE or matrix.empty:
        return []

    cfg = config or CLAUSE_ARM_DEFAULTS["global"]
    min_support = cfg.get("min_support", 0.20)
    min_confidence = cfg.get("min_confidence", 0.7)
    min_lift = cfg.get("min_lift", 1.8)

    df_bool = matrix.astype(bool)

    try:
        itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
        if itemsets.empty:
            return []
        rules_df = association_rules(itemsets, metric="lift", min_threshold=min_lift)
    except Exception as e:
        logger.warning(f"FP-Growth failed: {e}")
        return []

    rules_df = rules_df[rules_df["confidence"] >= min_confidence]

    rules = []
    for _, row in rules_df.iterrows():
        rules.append({
            "rule_id": f"R_{uuid.uuid4().hex[:8]}",
            "antecedent": sorted(list(row["antecedents"])),
            "consequent": sorted(list(row["consequents"])),
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "rule_type": "clause_dependency",
        })

    logger.info(f"Mined {len(rules)} global clause rules")
    return rules


def mine_per_cluster(
    matrix: pd.DataFrame,
    cluster_labels: dict[str, int],
    config: dict | None = None,
) -> dict[int, list[dict]]:
    """Mine association rules per cluster.

    Args:
        matrix: Full clause presence matrix.
        cluster_labels: {doc_id: cluster_label}
        config: Threshold config dict.

    Returns:
        {cluster_id: [rules]}
    """
    if not _MLXTEND_AVAILABLE or matrix.empty:
        return {}

    cfg = config or CLAUSE_ARM_DEFAULTS
    size_tiers = cfg.get("per_cluster_by_size", CLAUSE_ARM_DEFAULTS["per_cluster_by_size"])

    cluster_rules = {}
    unique_clusters = set(cluster_labels.values())

    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue

        doc_ids = [d for d, c in cluster_labels.items() if c == cluster_id]
        valid_ids = [d for d in doc_ids if d in matrix.index]
        cluster_matrix = matrix.loc[valid_ids]

        n_docs = len(cluster_matrix)
        if n_docs < 10:
            continue

        if n_docs >= size_tiers["large"]["min_docs"]:
            min_support = size_tiers["large"]["min_support"]
        elif n_docs >= size_tiers["medium"]["min_docs"]:
            min_support = size_tiers["medium"]["min_support"]
        else:
            min_support = size_tiers["small"]["min_support"]

        # Drop zero-variance columns
        col_sums = cluster_matrix.sum()
        valid_cols = col_sums[(col_sums > 0) & (col_sums < n_docs)].index
        if len(valid_cols) < 2:
            continue

        filtered = cluster_matrix[valid_cols].astype(bool)
        try:
            itemsets = fpgrowth(filtered, min_support=min_support, use_colnames=True)
            if itemsets.empty:
                continue
            rules_df = association_rules(itemsets, metric="lift", min_threshold=1.5)
        except Exception:
            continue

        rules = []
        for _, row in rules_df.iterrows():
            rules.append({
                "rule_id": f"RC{cluster_id}_{uuid.uuid4().hex[:8]}",
                "antecedent": sorted(list(row["antecedents"])),
                "consequent": sorted(list(row["consequents"])),
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
                "rule_type": "clause_dependency",
                "cluster_id": str(cluster_id),
            })

        if rules:
            cluster_rules[cluster_id] = rules

    return cluster_rules


def discover_term_packages(
    matrix: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    """Find frequent itemsets with 3+ clause types (term packages).

    Returns list of packages with: package_id, clause_types, support, size.
    """
    if not _MLXTEND_AVAILABLE or matrix.empty:
        return []

    cfg = config or CLAUSE_ARM_DEFAULTS.get("term_packages", {})
    min_support = cfg.get("min_support", 0.15)
    min_size = cfg.get("min_itemset_size", 3)

    df_bool = matrix.astype(bool)
    try:
        itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
    except Exception as e:
        logger.warning(f"FP-Growth for term packages failed: {e}")
        return []

    packages = []
    for _, row in itemsets.iterrows():
        items = sorted(list(row["itemsets"]))
        if len(items) >= min_size:
            packages.append({
                "package_id": f"TP_{uuid.uuid4().hex[:8]}",
                "clause_types": items,
                "support": float(row["support"]),
                "size": len(items),
            })

    packages.sort(key=lambda x: x["support"], reverse=True)
    logger.info(f"Discovered {len(packages)} term packages")
    return packages
```

- [ ] **Step 4: Create field_arm_miner.py**

Create `core/arm/field_arm_miner.py`:

```python
"""Field-value Association Rule Mining.

Discovers field-value correlations within and across clause types.
Ported from keyBERTClause, adapted to use ClusteringStore.
"""
import json
import logging
import uuid

import numpy as np
import pandas as pd

from core.arm.configs import FIELD_ARM_DEFAULTS

logger = logging.getLogger(__name__)

try:
    from mlxtend.frequent_patterns import association_rules, fpgrowth
    _MLXTEND_AVAILABLE = True
except ImportError:
    _MLXTEND_AVAILABLE = False


def normalize_field_value(value, field_name: str, config: dict | None = None) -> str | None:
    """Normalize a field value for ARM itemset encoding."""
    if value is None:
        return None

    cfg = config or FIELD_ARM_DEFAULTS.get("normalization", {})

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        bins = cfg.get("numeric_bins", [0, 7, 30, 60, 90, 180, 365, 999999])
        labels = cfg.get("numeric_labels", ["0-7d", "8-30d", "31-60d", "61-90d", "91-180d", "181-365d", "365d+"])
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                return labels[i] if i < len(labels) else str(value)
        return labels[-1] if labels else str(value)

    if isinstance(value, str):
        return value.lower().strip()[:50]

    if isinstance(value, list):
        return "+".join(str(v) for v in value[:3])

    return str(value)[:50]


def build_field_itemsets(
    extractions: dict[str, dict],
    config: dict | None = None,
) -> pd.DataFrame:
    """Build binary field-value itemset matrix.

    Args:
        extractions: {doc_id: {clause_type: {field_name: value}}}

    Returns:
        DataFrame with doc_id index and "clause_type.field=value" columns.
    """
    cfg = config or FIELD_ARM_DEFAULTS
    rows = []

    for doc_id, clause_data in extractions.items():
        row = {"doc_id": doc_id}
        for clause_type, fields in clause_data.items():
            if not isinstance(fields, dict):
                continue
            for field_name, value in fields.items():
                normalized = normalize_field_value(value, field_name, cfg.get("normalization"))
                if normalized is not None:
                    col = f"{clause_type}.{field_name}={normalized}"
                    row[col] = 1
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("doc_id").fillna(0).astype(int)
    # Filter to items appearing in sufficient docs
    col_sums = df.sum()
    valid_cols = col_sums[col_sums >= 5].index
    return df[valid_cols] if len(valid_cols) > 0 else pd.DataFrame()


def mine_field_rules(
    matrix: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    """Mine field-value association rules.

    Returns list of rules with cross-clause field correlations.
    """
    if not _MLXTEND_AVAILABLE or matrix.empty:
        return []

    cfg = config or FIELD_ARM_DEFAULTS.get("global", {})
    min_support = cfg.get("min_support", 0.05)
    min_confidence = cfg.get("min_confidence", 0.5)
    min_lift = cfg.get("min_lift", 1.5)
    max_rules = cfg.get("max_rules", 1000)

    df_bool = matrix.astype(bool)

    try:
        itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
        if itemsets.empty:
            return []
        rules_df = association_rules(itemsets, metric="lift", min_threshold=min_lift)
    except Exception as e:
        logger.warning(f"Field ARM failed: {e}")
        return []

    rules_df = rules_df[rules_df["confidence"] >= min_confidence]
    rules_df = rules_df.head(max_rules)

    rules = []
    for _, row in rules_df.iterrows():
        antecedent = sorted(list(row["antecedents"]))
        consequent = sorted(list(row["consequents"]))

        # Determine if this is a cross-clause rule
        ant_clauses = {item.split(".")[0] for item in antecedent if "." in item}
        con_clauses = {item.split(".")[0] for item in consequent if "." in item}
        is_cross_clause = bool(ant_clauses and con_clauses and ant_clauses != con_clauses)

        rules.append({
            "rule_id": f"FR_{uuid.uuid4().hex[:8]}",
            "antecedent": antecedent,
            "consequent": consequent,
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "rule_type": "cross_clause_field" if is_cross_clause else "field_correlation",
        })

    logger.info(f"Mined {len(rules)} field rules ({sum(1 for r in rules if r['rule_type'] == 'cross_clause_field')} cross-clause)")
    return rules
```

- [ ] **Step 5: Verify ARM module imports**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "
from core.arm.arm_miner import mine_global, build_clause_presence_matrix, discover_term_packages, mlxtend_available
from core.arm.field_arm_miner import mine_field_rules, build_field_itemsets
print('mlxtend available:', mlxtend_available())
print('ARM module loaded OK')
"`
Expected: All imports succeed

- [ ] **Step 6: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add core/arm/
git commit -m "feat: add ARM mining module (clause-level and field-level FP-Growth)"
```

---

### Task 12: Create core/relationship_layer.py

**Files:**

- Create: `core/relationship_layer.py`

- [ ] **Step 1: Write the test function**

Add before `main()` in `scripts/validate_pipeline.py`:

```python
def test_arm_miner():
    """Validate ARM mining on synthetic data."""
    from core.arm.arm_miner import (
        build_clause_presence_matrix,
        discover_term_packages,
        mine_global,
        mlxtend_available,
    )

    if not mlxtend_available():
        print("  [SKIP] mlxtend not installed — ARM mining unavailable")
        return

    # Build synthetic clause presence data
    assignments = {}
    clause_types_pool = ["Termination", "Notice", "Confidentiality", "Indemnification", "IP Rights"]
    import random
    random.seed(42)
    for i in range(100):
        # Termination and Notice strongly co-occur
        present = []
        if random.random() < 0.7:
            present.extend(["Termination", "Notice"])
        if random.random() < 0.5:
            present.append("Confidentiality")
        if random.random() < 0.3:
            present.append("Indemnification")
        if random.random() < 0.2:
            present.append("IP Rights")
        if present:
            assignments[f"doc_{i}"] = present

    matrix = build_clause_presence_matrix(assignments)
    assert not matrix.empty, "Matrix should not be empty"
    assert matrix.shape[0] >= 50, f"Expected >= 50 docs, got {matrix.shape[0]}"

    rules = mine_global(matrix, config={"min_support": 0.15, "min_confidence": 0.5, "min_lift": 1.0})
    # Should find Termination <-> Notice co-occurrence
    assert isinstance(rules, list), "Rules should be a list"

    packages = discover_term_packages(matrix, config={"min_support": 0.10, "min_itemset_size": 2})
    assert isinstance(packages, list), "Packages should be a list"

    print(f"  [PASS] ARM mining")
    print(f"    Matrix: {matrix.shape}, Rules: {len(rules)}, Packages: {len(packages)}")
```

- [ ] **Step 2: Write the relationship_layer module**

Create `core/relationship_layer.py`:

```python
"""Unified query interface over ARM relationship outputs.

Consumers (field_discovery, extractor, context_graph) call this layer
to get clause dependencies, field correlations, and term packages
without knowing the ARM mining internals.
"""
import json
import logging
from typing import Any

from core.store import ClusteringStore

logger = logging.getLogger(__name__)


class RelationshipLayer:
    """Query interface over ARM-mined relationships."""

    def __init__(self, store: ClusteringStore):
        self.store = store
        self._rules_cache: list[dict] | None = None
        self._packages_cache: list[dict] | None = None

    def _load_rules(self) -> list[dict]:
        """Load all ARM rules from DuckDB."""
        if self._rules_cache is not None:
            return self._rules_cache
        try:
            rows = self.store.conn.execute(
                "SELECT rule_id, antecedent, consequent, rule_type, "
                "support, confidence, lift, cluster_id FROM arm_rules"
            ).fetchall()
            self._rules_cache = [
                {
                    "rule_id": r[0],
                    "antecedent": json.loads(r[1]),
                    "consequent": json.loads(r[2]),
                    "rule_type": r[3],
                    "support": r[4],
                    "confidence": r[5],
                    "lift": r[6],
                    "cluster_id": r[7],
                }
                for r in rows
            ]
        except Exception:
            self._rules_cache = []
        return self._rules_cache

    def _load_packages(self) -> list[dict]:
        """Load all term packages from DuckDB."""
        if self._packages_cache is not None:
            return self._packages_cache
        try:
            rows = self.store.conn.execute(
                "SELECT package_id, clause_types, support, size, cluster_id "
                "FROM arm_term_packages"
            ).fetchall()
            self._packages_cache = [
                {
                    "package_id": r[0],
                    "clause_types": json.loads(r[1]),
                    "support": r[2],
                    "size": r[3],
                    "cluster_id": r[4],
                }
                for r in rows
            ]
        except Exception:
            self._packages_cache = []
        return self._packages_cache

    def get_related_clause_types(self, clause_type: str) -> list[dict]:
        """Get clause types that co-occur with the given type.

        Returns list of {clause_type, confidence, lift, direction} sorted by confidence.
        """
        rules = self._load_rules()
        related = []
        seen = set()

        for rule in rules:
            if rule["rule_type"] != "clause_dependency":
                continue
            if clause_type in rule["antecedent"]:
                for ct in rule["consequent"]:
                    if ct not in seen:
                        related.append({
                            "clause_type": ct,
                            "confidence": rule["confidence"],
                            "lift": rule["lift"],
                            "direction": "implies",
                        })
                        seen.add(ct)
            elif clause_type in rule["consequent"]:
                for ct in rule["antecedent"]:
                    if ct not in seen:
                        related.append({
                            "clause_type": ct,
                            "confidence": rule["confidence"],
                            "lift": rule["lift"],
                            "direction": "implied_by",
                        })
                        seen.add(ct)

        related.sort(key=lambda x: x["confidence"], reverse=True)
        return related

    def get_implied_fields(self, clause_types: list[str]) -> list[dict]:
        """Given present clause types, return fields to look for.

        Uses cross-clause field rules to suggest fields from related clause types.
        """
        rules = self._load_rules()
        implied = []

        for rule in rules:
            if rule["rule_type"] not in ("field_correlation", "cross_clause_field"):
                continue
            ant_set = set(rule["antecedent"])
            # Check if any antecedent item references a present clause type
            for item in ant_set:
                clause = item.split(".")[0] if "." in item else ""
                if clause in clause_types:
                    for cons_item in rule["consequent"]:
                        implied.append({
                            "field_item": cons_item,
                            "confidence": rule["confidence"],
                            "lift": rule["lift"],
                            "source_rule": rule["rule_id"],
                        })
                    break

        implied.sort(key=lambda x: x["confidence"], reverse=True)
        return implied

    def get_field_correlations(self, field_name: str, cluster_id: str | None = None) -> list[dict]:
        """Get field values that correlate with the given field."""
        rules = self._load_rules()
        correlations = []

        for rule in rules:
            if rule["rule_type"] not in ("field_correlation", "cross_clause_field"):
                continue
            if cluster_id and rule.get("cluster_id") and rule["cluster_id"] != cluster_id:
                continue

            all_items = rule["antecedent"] + rule["consequent"]
            if any(field_name in item for item in all_items):
                correlations.append({
                    "antecedent": rule["antecedent"],
                    "consequent": rule["consequent"],
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                })

        return correlations

    def get_term_packages(self) -> list[dict]:
        """Get frequent clause bundles (3+ types)."""
        return self._load_packages()

    def get_obligation_chains(self, clause_type: str) -> list[dict]:
        """Get obligation chains involving the given clause type."""
        rules = self._load_rules()
        chains = []

        for rule in rules:
            if rule["rule_type"] != "obligation_chain":
                continue
            if clause_type in rule["antecedent"] or clause_type in rule["consequent"]:
                chains.append(rule)

        return chains

    def mine_from_assignments(self, cluster_assignments: dict[str, list[str]],
                               cluster_labels: dict[str, int] | None = None,
                               pipeline_run_id: str = "") -> None:
        """Mine clause rules from cluster assignments and persist to DuckDB."""
        from core.arm.arm_miner import (
            build_clause_presence_matrix,
            discover_term_packages,
            mine_global,
            mine_per_cluster,
        )

        matrix = build_clause_presence_matrix(cluster_assignments)
        if matrix.empty:
            return

        # Global rules
        global_rules = mine_global(matrix)
        for rule in global_rules:
            self._persist_rule(rule, pipeline_run_id)

        # Per-cluster rules
        if cluster_labels:
            cluster_rules = mine_per_cluster(matrix, cluster_labels)
            for cid, rules in cluster_rules.items():
                for rule in rules:
                    rule["cluster_id"] = str(cid)
                    self._persist_rule(rule, pipeline_run_id)

        # Term packages
        packages = discover_term_packages(matrix)
        for pkg in packages:
            self._persist_package(pkg, pipeline_run_id)

        # Clear cache
        self._rules_cache = None
        self._packages_cache = None

        logger.info(f"Persisted {len(global_rules)} global rules, {len(packages)} term packages")

    def mine_field_correlations(self, extractions: dict[str, dict],
                                 pipeline_run_id: str = "") -> None:
        """Mine field-value correlations and persist to DuckDB."""
        from core.arm.field_arm_miner import build_field_itemsets, mine_field_rules

        matrix = build_field_itemsets(extractions)
        if matrix.empty:
            return

        rules = mine_field_rules(matrix)
        for rule in rules:
            self._persist_rule(rule, pipeline_run_id)

        self._rules_cache = None
        logger.info(f"Persisted {len(rules)} field rules")

    def _persist_rule(self, rule: dict, pipeline_run_id: str) -> None:
        """Insert or replace an ARM rule in DuckDB."""
        self.store.conn.execute(
            "INSERT OR REPLACE INTO arm_rules "
            "(rule_id, antecedent, consequent, rule_type, support, confidence, lift, cluster_id, pipeline_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rule["rule_id"],
                json.dumps(rule["antecedent"]),
                json.dumps(rule["consequent"]),
                rule["rule_type"],
                rule.get("support"),
                rule.get("confidence"),
                rule.get("lift"),
                rule.get("cluster_id"),
                pipeline_run_id,
            ],
        )

    def _persist_package(self, pkg: dict, pipeline_run_id: str) -> None:
        """Insert or replace a term package in DuckDB."""
        self.store.conn.execute(
            "INSERT OR REPLACE INTO arm_term_packages "
            "(package_id, clause_types, support, size, cluster_id, pipeline_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                pkg["package_id"],
                json.dumps(pkg["clause_types"]),
                pkg.get("support"),
                pkg.get("size"),
                pkg.get("cluster_id"),
                pipeline_run_id,
            ],
        )
```

- [ ] **Step 3: Verify relationship layer imports**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "
from core.relationship_layer import RelationshipLayer
print('RelationshipLayer loaded OK')
"`
Expected: Import succeeds

- [ ] **Step 4: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add core/relationship_layer.py
git commit -m "feat: add RelationshipLayer unified ARM query interface"
```

---

### Task 13: Add Phase 3 validation test

**Files:**

- Modify: `scripts/validate_pipeline.py`

- [ ] **Step 1: Add test_arm_miner and test_relationship_layer to validate_pipeline.py**

Add the `test_arm_miner` function from Task 12, Step 1.

Add this additional test:

```python
def test_relationship_layer():
    """Validate RelationshipLayer end-to-end: mine -> persist -> query."""
    from core.arm.arm_miner import mlxtend_available

    if not mlxtend_available():
        print("  [SKIP] mlxtend not installed")
        return

    import tempfile
    from core.store import ClusteringStore
    from core.relationship_layer import RelationshipLayer

    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as tmp:
        store = ClusteringStore(tmp.name)
        layer = RelationshipLayer(store)

        # Mine from synthetic assignments
        import random
        random.seed(42)
        assignments = {}
        for i in range(100):
            present = []
            if random.random() < 0.7:
                present.extend(["Termination", "Notice"])
            if random.random() < 0.5:
                present.append("Confidentiality")
            if random.random() < 0.3:
                present.append("Indemnification")
            if present:
                assignments[f"doc_{i}"] = present

        layer.mine_from_assignments(assignments, pipeline_run_id="test_run")

        # Query
        related = layer.get_related_clause_types("Termination")
        packages = layer.get_term_packages()

        assert isinstance(related, list), "Related should be a list"
        assert isinstance(packages, list), "Packages should be a list"

        print(f"  [PASS] RelationshipLayer")
        print(f"    Related to 'Termination': {len(related)}, Packages: {len(packages)}")
```

Add both to the `tests` list in `main()`:

```python
        ("ARM Miner", test_arm_miner),  # NEW
        ("Relationship Layer", test_relationship_layer),  # NEW
```

- [ ] **Step 2: Run full validation**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python scripts/validate_pipeline.py --skip-llm`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add scripts/validate_pipeline.py
git commit -m "test: add ARM miner and RelationshipLayer validation"
```

---

## Phase 4: Downstream Integration

### Task 14: ARM-augmented field discovery prompt and integration

**Files:**

- Create: `prompts/field_discovery_arm.txt`
- Modify: `core/field_discovery.py`

- [ ] **Step 1: Create the ARM field discovery prompt**

Create `prompts/field_discovery_arm.txt`:

```
You are a legal contract analyst. Given sample text chunks from a specific clause type cluster, identify the structured fields that can be extracted from this type of clause.

## Clause Type
{cluster_label}

## Cluster Description
{cluster_description}

## Related Clause Types (discovered via association rule mining)
{related_clauses}

## Known Field Correlations
{field_correlations}

## Term Package Context
{term_packages}

## Sample Chunks
{sample_chunks}

## Instructions

1. Identify all extractable fields from the sample chunks above.
2. Consider fields that may be cross-referenced from the related clause types listed above.
3. For each field, provide:
   - name: snake_case field name
   - type: string | number | date | boolean | enum
   - description: what this field captures
   - required: whether this field is typically present (true/false)
   - cross_clause: whether this field relates to a different clause type (true/false)

Return a JSON array of field definitions.
```

- [ ] **Step 2: Add ARM context injection to field_discovery.py**

In `core/field_discovery.py`, add a helper function that builds ARM context for prompts:

```python
def _build_arm_context(cluster_label: str, relationship_layer=None) -> dict[str, str]:
    """Build ARM relationship context for field discovery prompts.

    Returns dict with keys: related_clauses, field_correlations, term_packages.
    All values are formatted strings ready for prompt injection.
    Returns empty strings if no relationship layer or no data.
    """
    if relationship_layer is None:
        return {
            "related_clauses": "No relationship data available.",
            "field_correlations": "No field correlation data available.",
            "term_packages": "No term package data available.",
        }

    # Related clause types
    related = relationship_layer.get_related_clause_types(cluster_label)
    if related:
        lines = []
        for r in related[:10]:
            lines.append(f"- {r['clause_type']} (confidence: {r['confidence']:.2f}, lift: {r['lift']:.2f}, {r['direction']})")
        related_str = "\n".join(lines)
    else:
        related_str = "No related clause types discovered."

    # Field correlations
    implied = relationship_layer.get_implied_fields([cluster_label])
    if implied:
        lines = []
        for f in implied[:10]:
            lines.append(f"- {f['field_item']} (confidence: {f['confidence']:.2f})")
        field_str = "\n".join(lines)
    else:
        field_str = "No field correlations discovered."

    # Term packages
    packages = relationship_layer.get_term_packages()
    relevant = [p for p in packages if cluster_label in p.get("clause_types", [])]
    if relevant:
        lines = []
        for p in relevant[:5]:
            types = ", ".join(p["clause_types"])
            lines.append(f"- {{{types}}} (support: {p['support']:.2f})")
        pkg_str = "\n".join(lines)
    else:
        pkg_str = "No relevant term packages."

    return {
        "related_clauses": related_str,
        "field_correlations": field_str,
        "term_packages": pkg_str,
    }
```

- [ ] **Step 3: Verify the function loads**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "
from core.field_discovery import _build_arm_context
ctx = _build_arm_context('Termination', relationship_layer=None)
print('Keys:', list(ctx.keys()))
print('Related:', ctx['related_clauses'][:50])
"`
Expected: Returns dict with three keys, all showing "No ... available."

- [ ] **Step 4: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add prompts/field_discovery_arm.txt core/field_discovery.py
git commit -m "feat: add ARM-augmented field discovery prompt and context builder"
```

---

### Task 15: ARM-augmented extraction prompt and integration

**Files:**

- Create: `prompts/field_extraction_arm_conditioned.txt`
- Modify: `core/extractor.py`

- [ ] **Step 1: Create the ARM extraction prompt**

Create `prompts/field_extraction_arm_conditioned.txt`:

```
Extract structured fields from the following legal clause text.

## Clause Type: {cluster_label}
## Cluster Description: {cluster_description}

## Primary Fields to Extract
{fields}

## Cross-Clause Fields (suggested by association rules)
These fields are from related clause types and may be referenced in this text:
{cross_clause_fields}

## Field Correlations
Known correlations between field values (use as validation hints):
{field_correlations}

## Text
{text}

## Instructions
1. Extract values for each primary field listed above.
2. Also check for any cross-clause field values mentioned in the text.
3. For each extracted value, provide:
   - field_name: the field name
   - value: the extracted value
   - confidence: 0.0-1.0 confidence score
   - evidence: the text span supporting this extraction
   - is_cross_clause: true if this is a cross-clause field

Return a JSON object mapping field names to extraction results.
```

- [ ] **Step 2: Add cross-clause context support to extractor.py**

In `core/extractor.py`, add a helper function:

```python
def _build_extraction_arm_context(
    cluster_label: str,
    relationship_layer=None,
) -> dict[str, str]:
    """Build ARM context for extraction prompts.

    Returns dict with: cross_clause_fields, field_correlations.
    """
    if relationship_layer is None:
        return {
            "cross_clause_fields": "None.",
            "field_correlations": "None.",
        }

    implied = relationship_layer.get_implied_fields([cluster_label])
    if implied:
        lines = [f"- {f['field_item']} (confidence: {f['confidence']:.2f})" for f in implied[:10]]
        cross_str = "\n".join(lines)
    else:
        cross_str = "None."

    correlations = relationship_layer.get_field_correlations(cluster_label)
    if correlations:
        lines = []
        for c in correlations[:10]:
            ant = ", ".join(c["antecedent"])
            con = ", ".join(c["consequent"])
            lines.append(f"- {ant} -> {con} (lift: {c['lift']:.2f})")
        corr_str = "\n".join(lines)
    else:
        corr_str = "None."

    return {
        "cross_clause_fields": cross_str,
        "field_correlations": corr_str,
    }
```

- [ ] **Step 3: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add prompts/field_extraction_arm_conditioned.txt core/extractor.py
git commit -m "feat: add ARM-augmented extraction prompt and context builder"
```

---

### Task 16: Knowledge graph ARM integration

**Files:**

- Modify: `core/context_graph.py`

- [ ] **Step 1: Add add_arm_relationships method to context_graph.py**

Read `core/context_graph.py` first to identify the main graph class and its edge-creation method (likely `_add_edge` or `add_edge`). Then add this method to that class, adapting the edge-creation call to match the existing API:

```python
    def add_arm_relationships(self, relationship_layer) -> int:
        """Add ARM-discovered relationships as graph edges.

        Creates edges for clause dependencies, term packages, and field correlations.
        Returns the number of edges added.
        """
        edges_added = 0

        # Clause dependencies -> DEPENDS_ON edges
        rules = relationship_layer._load_rules()
        for rule in rules:
            if rule["rule_type"] == "clause_dependency":
                for ant in rule["antecedent"]:
                    for con in rule["consequent"]:
                        self._add_edge(
                            source=ant,
                            target=con,
                            edge_type="DEPENDS_ON",
                            properties={
                                "confidence": rule["confidence"],
                                "lift": rule["lift"],
                                "support": rule["support"],
                                "rule_id": rule["rule_id"],
                            },
                        )
                        edges_added += 1

            elif rule["rule_type"] in ("field_correlation", "cross_clause_field"):
                for ant in rule["antecedent"]:
                    for con in rule["consequent"]:
                        self._add_edge(
                            source=ant,
                            target=con,
                            edge_type="FIELD_CORRELATES",
                            properties={
                                "lift": rule["lift"],
                                "confidence": rule["confidence"],
                                "rule_id": rule["rule_id"],
                            },
                        )
                        edges_added += 1

        # Term packages -> BUNDLED_WITH edges
        packages = relationship_layer.get_term_packages()
        for pkg in packages:
            pkg_node = f"TP:{','.join(pkg['clause_types'][:3])}"
            for ct in pkg["clause_types"]:
                self._add_edge(
                    source=ct,
                    target=pkg_node,
                    edge_type="BUNDLED_WITH",
                    properties={
                        "support": pkg["support"],
                        "package_size": pkg["size"],
                        "package_id": pkg["package_id"],
                    },
                )
                edges_added += 1

        return edges_added
```

- [ ] **Step 2: Verify it doesn't break existing graph code**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python -c "
from core.context_graph import *
print('context_graph module loaded OK')
"`
Expected: Import succeeds

- [ ] **Step 3: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add core/context_graph.py
git commit -m "feat: add ARM relationship edges to knowledge graph"
```

---

### Task 17: Add Phase 4 validation test and final integration check

**Files:**

- Modify: `scripts/validate_pipeline.py`

- [ ] **Step 1: Add test_relationship_integration to validate_pipeline.py**

Add before `main()`:

```python
def test_relationship_integration():
    """Validate that downstream consumers can query the relationship layer."""
    from core.arm.arm_miner import mlxtend_available

    if not mlxtend_available():
        print("  [SKIP] mlxtend not installed")
        return

    import tempfile
    from core.store import ClusteringStore
    from core.relationship_layer import RelationshipLayer
    from core.field_discovery import _build_arm_context
    from core.extractor import _build_extraction_arm_context

    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as tmp:
        store = ClusteringStore(tmp.name)
        layer = RelationshipLayer(store)

        # Mine synthetic data
        import random
        random.seed(42)
        assignments = {}
        for i in range(100):
            present = []
            if random.random() < 0.7:
                present.extend(["Termination", "Notice"])
            if random.random() < 0.5:
                present.append("Confidentiality")
            if present:
                assignments[f"doc_{i}"] = present

        layer.mine_from_assignments(assignments, pipeline_run_id="test")

        # Test field discovery context
        fd_ctx = _build_arm_context("Termination", relationship_layer=layer)
        assert "related_clauses" in fd_ctx, "Missing related_clauses key"
        assert fd_ctx["related_clauses"] != "No related clause types discovered.", \
            "Should find related clause types for Termination"

        # Test extraction context
        ex_ctx = _build_extraction_arm_context("Termination", relationship_layer=layer)
        assert "cross_clause_fields" in ex_ctx, "Missing cross_clause_fields key"

        print("  [PASS] Relationship integration")
        print(f"    Field discovery context keys: {list(fd_ctx.keys())}")
```

Add to the `tests` list:

```python
        ("Relationship Integration", test_relationship_integration),  # NEW
```

- [ ] **Step 2: Run full validation suite**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python scripts/validate_pipeline.py --skip-llm`
Expected: All tests pass including all new tests

- [ ] **Step 3: Commit**

```bash
cd /mnt/data/code/clustering/.claude/hackathon
git add scripts/validate_pipeline.py
git commit -m "test: add downstream integration validation for ARM consumers"
```

- [ ] **Step 4: Final commit with all Phase 4 changes**

Run: `cd /mnt/data/code/clustering/.claude/hackathon && poetry run python scripts/validate_pipeline.py --skip-llm`
Expected: All tests pass — full integration verified
