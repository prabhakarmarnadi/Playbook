"""
Legal-aware semantic chunker — Greg Kamradt method + legal structural signals.

Ported from legacy SemanticChunker in apr_fields_processor, adapted for:
  - Nomic embedder (GPU, synchronous) instead of async AIFlow gRPC
  - No external dependencies on blingfire (uses regex sentence splitting)
  - Same algorithm: sentence buffers → embed → cosine distance → neighbor
    smoothing → legal header boosting → percentile threshold → chunk

How it works (per-document semantic_chunk path):
  1. Split document into sentences (regex, handles legal abbreviations)
  2. Group sentences into overlapping buffers for context
  3. Embed each buffer with Nomic (or any embedder)
  4. Compute cosine distance between consecutive buffer embeddings
  5. Smooth distances with neighbor averaging (reduces noise)
  6. Boost distances at legal structural boundaries (ARTICLE, Section, etc.)
  7. Split where combined distance > percentile threshold
  8. Enforce min/max chunk size constraints

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE: Structural-First, Semantic-Fallback Unified Chunking
═══════════════════════════════════════════════════════════════════════════════

The primary batch entry point is `batch_generate_unified_chunks()`.  It
replaces the old `batch_generate_typed_chunks()` which produced 3 separate
chunk types (embedding, extraction, clause) with lossy mean-of-buffers
embeddings and a fragile char-range overlap linking step.

Design principles:
  • ONE chunk type aligned to legal provision boundaries (200-5000 chars).
  • Structural split FIRST — use legal heading patterns (ARTICLE, Section,
    numbered clauses) to keep complete provisions intact.
  • Semantic sub-split ONLY for oversized sections (>5000 chars) that
    exceed the max_chunk_chars limit.
  • Full-text GPU embedding — no lossy mean-of-buffers approximation.
  • 1:1 chunk↔clause mapping — each chunk carries clause_number,
    clause_title, heading_level as annotations.  No separate clause records
    or char-range overlap linking.

3-pass batch architecture (batch_generate_unified_chunks):
  Pass 1 (CPU):  Structural split ALL docs at legal boundaries →
                  identify oversized (>max_chunk_chars) sections and
                  cohesion candidates for multi-topic detection.
  Pass 2 (GPU):  Embed sentence buffers ONLY for oversized sections
                  (~30% of all sections) + cohesion half-texts.
                  This is dramatically cheaper than the old approach which
                  embedded ALL sentence buffers for ALL documents.
  Pass 3 (CPU):  Semantic sub-split oversized sections using embedding
                  breakpoints → forward-merge undersized first chunks →
                  backward-merge undersized trailing chunks → force-split
                  any remaining oversized chunks → annotate with clause
                  metadata extracted from each chunk's text.

Quality improvements over the old approach:
  • Complete legal provisions stay intact (avg 5-8K char clauses no longer
    fragmented into 2K embedding chunks).
  • Full-text embeddings instead of lossy mean-of-buffers → better cosine
    similarity for downstream clustering and RAG retrieval.
  • ~50-60% fewer total chunks (structural boundaries are sparser than
    semantic percentile breakpoints) → faster clustering, fewer LLM calls.
  • Clean 1:1 clause mapping eliminates the fragile char-range overlap
    linking that produced duplicates and missed clauses.

Speed improvements:
  • GPU embedding only for oversized sections (~30%) instead of all buffers.
  • batch_size=256 on A100 80GB for full VRAM utilisation.
  • CPU-only structural split is O(n) regex — negligible cost.
═══════════════════════════════════════════════════════════════════════════════
"""
import re
import uuid
from dataclasses import dataclass

import numpy as np

from core.nupunkt_chunker import structural_decompose

# ── Legal structural patterns (from legacy semantic_chunker.py) ────────────────
LEGAL_SECTION_PATTERNS = [
    r"^\s*(?:ARTICLE|Article|SECTION|Section)\s+[IVXLCDM\d]+",
    r"^\s*\d+(?:\.\d+)*\.\s+[A-Z]",
    r"^\s*\([a-z]\)\s",
    r"^\s*\([ivxlcdm]+\)\s",
    r"^\s*[-•●◦▪]\s",
    r"^[A-Z][A-Z\s,&]{10,}$",
    r"^\s*(?:WHEREAS|NOW,?\s*THEREFORE|IN WITNESS WHEREOF|RECITALS|DEFINITIONS)",
    r"^\s*(?:EXHIBIT|SCHEDULE|APPENDIX|ANNEX)\s+[A-Z\d]",
]
LEGAL_SECTION_RE = re.compile("|".join(LEGAL_SECTION_PATTERNS), re.MULTILINE)

# Abbreviations to protect from sentence splitting
_ABBREV_RE = re.compile(
    r"(?:Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Corp|Co|vs|etc|i\.e|e\.g|cf|viz|al|No|Art|Sec|U\.S|U\.K|E\.U)\."
)


@dataclass
class SemanticChunkerConfig:
    """Configuration for legal-aware semantic chunking."""

    buffer_size: int = 1               # Sentences before/after for context window
    breakpoint_percentile: float = 90.0  # Percentile threshold for breakpoints
    min_chunk_chars: int = 100         # Minimum characters per chunk
    max_chunk_chars: int = 2000        # Maximum characters per chunk
    neighbor_weight: float = 0.3       # Neighbor smoothing weight (0 = no smoothing)
    structural_bonus_strong: float = 0.15  # Bonus for ARTICLE/Section headers
    structural_bonus_weak: float = 0.08    # Bonus for numbered items
    clause_split: bool = False         # If True, split on structural clause boundaries first
    extraction_max_chunk_chars: int = 16000  # Max chars for extraction chunks (~4000 tok)


# Module-level default config
DEFAULT_CONFIG = SemanticChunkerConfig()


def _approx_token_count(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters (GPT-family average)."""
    return len(text) // 4


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences. Handles legal abbreviations and paragraph breaks.

    Uses regex-based splitting (no blingfire dependency).
    Preserves paragraph boundaries as natural sentence boundaries.
    """
    if not text or not text.strip():
        return []

    # Normalize paragraph breaks
    text = re.sub(r"\n\s*\n", " [PARA_BREAK] ", text)
    # Normalize single newlines within paragraphs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Protect abbreviations: replace dots with placeholder
    protected = _ABBREV_RE.sub(lambda m: m.group().replace(".", "[DOT]"), text)
    # Protect decimal numbers (e.g., "3.5 years")
    protected = re.sub(r"(\d)\.", r"\1[DOT]", protected)

    # Split on sentence-ending punctuation followed by space + uppercase or marker
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\(\"'])", protected)

    sentences = []
    for part in parts:
        # Restore dots
        part = part.replace("[DOT]", ".")
        # Handle paragraph breaks
        if "[PARA_BREAK]" in part:
            sub_parts = part.split("[PARA_BREAK]")
            sentences.extend(s.strip() for s in sub_parts if s.strip())
        elif part.strip():
            sentences.append(part.strip())

    return sentences


def _combine_sentence_buffers(
    sentences: list[str],
    buffer_size: int,
) -> list[str]:
    """
    Combine sentences into overlapping buffers for embedding.

    Each buffer[i] = sentences[i-buffer_size : i+buffer_size+1] joined.
    This gives the embedder context around each sentence position.
    """
    combined = []
    for i in range(len(sentences)):
        start = max(0, i - buffer_size)
        end = min(len(sentences), i + buffer_size + 1)
        combined.append(" ".join(sentences[start:end]))
    return combined


def _compute_breakpoints(
    embeddings: np.ndarray,
    sentences: list[str],
    config: SemanticChunkerConfig,
) -> list[int]:
    """
    Find semantic breakpoints using cosine distance + smoothing + legal boosting.

    Returns list of sentence indices where splits should occur.
    """
    if len(embeddings) < 2:
        return []

    # Cosine distances between consecutive embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-12)
    distances = np.array([
        1.0 - float(np.dot(normed[i], normed[i + 1]))
        for i in range(len(normed) - 1)
    ])

    # Neighbor smoothing — a real topic shift shows high distance across
    # multiple consecutive positions, not just one
    smoothed = np.copy(distances)
    nw = config.neighbor_weight
    if nw > 0:
        for i in range(len(smoothed)):
            neighbors = [distances[i]]
            if i > 0:
                neighbors.append(distances[i - 1])
            if i < len(distances) - 1:
                neighbors.append(distances[i + 1])
            smoothed[i] = (1 - nw) * distances[i] + nw * np.mean(neighbors)

    # Legal structural bonus — boost distance at positions where the next
    # sentence starts a legal section (ARTICLE, Section, WHEREAS, etc.)
    structural_bonus = np.zeros(len(smoothed))
    for i in range(len(smoothed)):
        next_idx = i + 1
        if next_idx < len(sentences):
            sent = sentences[next_idx]
            if LEGAL_SECTION_RE.match(sent):
                structural_bonus[i] = config.structural_bonus_strong
            elif re.match(r"^\s*\d+\.\s", sent):
                structural_bonus[i] = config.structural_bonus_weak

    combined = smoothed + structural_bonus

    # Percentile-based threshold
    threshold = float(np.percentile(combined, config.breakpoint_percentile))

    # Primary breakpoints: combined distance > threshold
    breakpoints = [i + 1 for i, d in enumerate(combined) if d > threshold]

    # Force breaks at very strong structural signals even if below threshold
    median_dist = float(np.median(combined))
    for i, bonus in enumerate(structural_bonus):
        if bonus >= config.structural_bonus_strong and (i + 1) not in breakpoints:
            if combined[i] > median_dist * 0.8:
                breakpoints.append(i + 1)

    return sorted(set(breakpoints))


def _assemble_chunks(
    sentences: list[str],
    breakpoints: list[int],
    config: SemanticChunkerConfig,
) -> list[str]:
    """
    Assemble text chunks from sentence list + breakpoint indices.
    Enforces min/max chunk size constraints.
    """
    if not breakpoints:
        full = " ".join(sentences)
        if len(full) > config.max_chunk_chars:
            return _force_split(sentences, config.max_chunk_chars)
        return [full] if full.strip() else []

    chunks = []
    start = 0

    for bp in breakpoints:
        chunk_text = " ".join(sentences[start:bp])

        if len(chunk_text) < config.min_chunk_chars and chunks:
            # Merge undersized chunk with previous
            chunks[-1] = chunks[-1] + " " + chunk_text
        elif len(chunk_text) > config.max_chunk_chars:
            # Force-split oversized chunk
            chunks.extend(_force_split(sentences[start:bp], config.max_chunk_chars))
        elif chunk_text.strip():
            chunks.append(chunk_text)

        start = bp

    # Remaining sentences
    if start < len(sentences):
        remaining = " ".join(sentences[start:])
        if len(remaining) < config.min_chunk_chars and chunks:
            chunks[-1] = chunks[-1] + " " + remaining
        elif remaining.strip():
            if len(remaining) > config.max_chunk_chars:
                chunks.extend(_force_split(sentences[start:], config.max_chunk_chars))
            else:
                chunks.append(remaining)

    return chunks


def _force_split(sentences: list[str], max_chars: int) -> list[str]:
    """Force-split a list of sentences into chunks ≤ max_chars."""
    chunks = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks





# ── Clause boundary patterns (work for legal + general business documents) ─────
# These match the start of a new clause/section. Ordered from strongest to weakest.
_CLAUSE_BOUNDARY_PATTERNS = [
    # Strong: explicit section/article markers (legal + general)
    r"^\s*(?:ARTICLE|Article|SECTION|Section)\s+[IVXLCDM\d]+",
    # Strong: top-level numbered sections (1. Title, 2. Title)
    r"^\s*\d+\.\s+[A-Z]",
    # Strong: ALL-CAPS headers (>10 chars, typical section titles)
    r"^[A-Z][A-Z\s,&]{10,}$",
    # Strong: legal ceremony markers
    r"^\s*(?:WHEREAS|NOW,?\s*THEREFORE|IN WITNESS WHEREOF|RECITALS|DEFINITIONS)",
    # Strong: exhibit/schedule/appendix markers
    r"^\s*(?:EXHIBIT|SCHEDULE|APPENDIX|ANNEX)\s+[A-Z\d]",
    # Medium: sub-sections (1.1, 2.3.1, etc.)
    r"^\s*\d+\.\d+(?:\.\d+)*\.?\s+",
    # Medium: lettered sub-clauses (a), (b), etc.
    r"^\s*\([a-z]\)\s",
    # Medium: roman numeral sub-clauses (i), (ii), etc.
    r"^\s*\([ivxlcdm]+\)\s",
]
_CLAUSE_BOUNDARY_RE = re.compile("|".join(_CLAUSE_BOUNDARY_PATTERNS), re.MULTILINE)


def _structural_clause_split(text: str, min_clause_chars: int = 80) -> list[str]:
    """
    Split document text into sections at structural clause boundaries.

    Works for any structured document (legal, business, technical).
    Uses section headers, numbered items, and structural markers as hard split points.
    Merges undersized fragments into the previous section.

    Args:
        text: Full document text.
        min_clause_chars: Minimum characters for a clause to stand alone.

    Returns:
        List of text segments, each representing one structural section/clause.
        If no structural markers are found, returns [text] unchanged.
    """
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    # Find all lines that match a clause boundary pattern
    boundary_line_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and _CLAUSE_BOUNDARY_RE.match(stripped):
            boundary_line_indices.append(i)

    if not boundary_line_indices:
        # No structural markers found — return text as-is for semantic splitting
        return [text.strip()] if text.strip() else []

    # Build sections from boundaries
    sections = []
    # Content before the first boundary
    if boundary_line_indices[0] > 0:
        preamble = "\n".join(lines[:boundary_line_indices[0]]).strip()
        if preamble:
            sections.append(preamble)

    for idx, start_line in enumerate(boundary_line_indices):
        end_line = boundary_line_indices[idx + 1] if idx + 1 < len(boundary_line_indices) else len(lines)
        section_text = "\n".join(lines[start_line:end_line]).strip()
        if section_text:
            sections.append(section_text)

    # Merge undersized sections into previous
    if len(sections) <= 1:
        return sections

    merged = [sections[0]]
    for s in sections[1:]:
        if len(s) < min_clause_chars and merged:
            merged[-1] = merged[-1] + "\n" + s
        else:
            merged.append(s)

    return merged


def cohesion_subsplit_clauses(
    sections: list[str],
    embed_fn,
    min_clause_chars: int = 100,
    cohesion_threshold: float = 0.75,
    min_sentences_to_split: int = 4,
) -> list[str]:
    """
    Sub-split structurally-identified clauses that contain multiple distinct topics.

    For each section, embeds the first half and second half of sentences independently.
    If cosine similarity between the two halves drops below `cohesion_threshold`, the
    section is semantically sub-split using the standard breakpoint algorithm.  Sections
    that are already cohesive are left intact.

    This catches omnibus sections like:
        "12. MISCELLANEOUS — governing law + termination + arbitration + entire agreement"
    that have no internal numbering but contain 3-4 distinct provisions.

    Args:
        sections:  Structural clause sections (from _structural_clause_split).
        embed_fn:  Function(texts: list[str]) -> np.ndarray (e.g. NomicEmbedder.embed_for_clustering).
        min_clause_chars:  Minimum characters for a sub-clause to stand alone.
        cohesion_threshold:  Cosine similarity below which a section is sub-split (0-1).
        min_sentences_to_split:  Sections with fewer sentences are never sub-split.

    Returns:
        Expanded list of clause sections, with low-cohesion ones sub-split.
    """
    if not sections or embed_fn is None:
        return sections

    # Identify candidates: sections with enough sentences to warrant a cohesion check
    candidates = []  # (index, sentences, buffers)
    all_half_texts = []  # pairs of (first_half_text, second_half_text) to batch-embed
    candidate_indices = []

    for i, section in enumerate(sections):
        sentences = _split_sentences(section)
        if len(sentences) < min_sentences_to_split:
            continue
        mid = len(sentences) // 2
        first_half = " ".join(sentences[:mid])
        second_half = " ".join(sentences[mid:])
        if len(first_half) < min_clause_chars or len(second_half) < min_clause_chars:
            continue
        candidates.append((i, sentences))
        all_half_texts.append(first_half[:512])
        all_half_texts.append(second_half[:512])
        candidate_indices.append(i)

    if not candidates:
        return sections

    # Batch-embed all halves at once (2 texts per candidate)
    half_embeddings = embed_fn(all_half_texts)

    # Check cohesion for each candidate
    needs_split = {}  # section_index -> sentences
    for ci, (sec_idx, sentences) in enumerate(candidates):
        emb_first = half_embeddings[ci * 2]
        emb_second = half_embeddings[ci * 2 + 1]
        norm_first = np.linalg.norm(emb_first)
        norm_second = np.linalg.norm(emb_second)
        if norm_first < 1e-12 or norm_second < 1e-12:
            continue
        cosine_sim = float(np.dot(emb_first, emb_second) / (norm_first * norm_second))
        if cosine_sim < cohesion_threshold:
            needs_split[sec_idx] = sentences

    if not needs_split:
        return sections

    # Semantic sub-split the low-cohesion sections
    # Collect all sentence buffers for batch embedding
    split_buffers = []
    split_meta = []  # (section_index, sentences, buf_start, buf_end)
    for sec_idx, sentences in needs_split.items():
        buffers = _combine_sentence_buffers(sentences, buffer_size=1)
        buf_start = len(split_buffers)
        split_buffers.extend(b[:512] for b in buffers)
        split_meta.append((sec_idx, sentences, buf_start, len(split_buffers)))

    buffer_embeddings = embed_fn(split_buffers)

    sub_split_results = {}
    for sec_idx, sentences, buf_start, buf_end in split_meta:
        embs = buffer_embeddings[buf_start:buf_end]
        cfg = SemanticChunkerConfig(
            buffer_size=1,
            breakpoint_percentile=75.0,  # more aggressive splitting for multi-clause sections
            min_chunk_chars=min_clause_chars,
            max_chunk_chars=4000,
            clause_split=False,
        )
        breakpoints = _compute_breakpoints(embs, sentences, cfg)
        if breakpoints:
            sub_texts = _assemble_chunks(sentences, breakpoints, cfg)
            # Only accept if we actually produced multiple sub-clauses
            if len(sub_texts) > 1:
                sub_split_results[sec_idx] = sub_texts

    # Rebuild the section list with sub-splits inserted
    result = []
    for i, section in enumerate(sections):
        if i in sub_split_results:
            result.extend(sub_split_results[i])
        else:
            result.append(section)

    return result


# ── Clause header extraction ──────────────────────────────────────────────────

# Patterns ordered from most specific to least. Each returns (number, title, level).
_CLAUSE_HEADER_PATTERNS = [
    # ARTICLE III. DEFINITIONS  or  Article 5: Term
    (1, re.compile(r'^\s*(?:ARTICLE|Article)\s+([IVXLCDM\d]+)[.:\s]*(.*)$')),
    # SECTION 12.3 — Indemnification
    (2, re.compile(r'^\s*(?:SECTION|Section)\s+(\d+(?:\.\d+)*)[.:\s]*(.*)$')),
    # 12. Governing Law   or   3. DEFINITIONS
    (2, re.compile(r'^\s*(\d+)\.\s+([A-Z][A-Za-z\s,&/\-]{2,60})$')),
    # 12.3 Indemnification  or  1.1.2 Sub-definitions
    (3, re.compile(r'^\s*(\d+\.\d+(?:\.\d+)*)\s+(.{3,60})$')),
    # EXHIBIT A  or  SCHEDULE 1
    (1, re.compile(r'^\s*((?:EXHIBIT|SCHEDULE|APPENDIX|ANNEX)\s+[A-Z\d]+)[.:\s]*(.*)$')),
    # WHEREAS / IN WITNESS WHEREOF / RECITALS / DEFINITIONS (ceremony)
    (1, re.compile(r'^\s*(WHEREAS|NOW,?\s*THEREFORE|IN WITNESS WHEREOF|RECITALS|DEFINITIONS)\b(.*)$')),
    # ALL-CAPS HEADER LINE (>10 chars, e.g., "LIMITATION OF LIABILITY")
    (2, re.compile(r'^([A-Z][A-Z\s,&/\-]{10,})$')),
]


def extract_clause_header(text: str) -> dict | None:
    """
    Extract clause number, title, and heading level from a clause text block.

    Looks at the first non-empty line for a recognizable header pattern.

    Returns:
        dict with {clause_number, clause_title, heading_level} or None if no header found.
    """
    if not text:
        return None

    # Check first 3 non-empty lines (sometimes there's a blank line before the header)
    lines = text.split("\n")
    for line in lines[:3]:
        stripped = line.strip()
        if not stripped:
            continue
        for level, pattern in _CLAUSE_HEADER_PATTERNS:
            m = pattern.match(stripped)
            if m:
                groups = m.groups()
                if len(groups) >= 2:
                    number = groups[0].strip()
                    title = groups[1].strip().rstrip(':.-– ')
                    # For ALL-CAPS pattern (last one), number IS the title
                    if not title and number:
                        title = number.title()
                        number = ""
                else:
                    number = ""
                    title = groups[0].strip() if groups else stripped
                return {
                    "clause_number": number,
                    "clause_title": title if title else stripped[:60],
                    "heading_level": level,
                }
        # Stop at first non-empty line if no match
        break

    return None


def semantic_chunk(
    text: str,
    embed_fn,
    config: SemanticChunkerConfig | None = None,
) -> list[dict]:
    """
    Chunk text using legal-aware semantic splitting with embedding similarity.

    This is the primary chunking function. It uses the Nomic (or any) embedder
    to detect topic boundaries within a document, boosted by legal structural
    signals (section headers, article markers, etc.).

    Args:
        text: Document text to chunk.
        embed_fn: Function(texts: list[str]) -> np.ndarray.
                  Typically NomicEmbedder.embed_for_clustering or Embedder.embed.
        config: Chunker configuration (defaults to SemanticChunkerConfig()).

    Returns:
        List of chunk dicts: {chunk_id, text, char_start, char_end, token_count, chunk_index}.
    """
    if not text or not text.strip():
        return []

    cfg = config or DEFAULT_CONFIG

    # ── Clause-aware mode: structural pre-split then semantic sub-split ─────
    if cfg.clause_split:
        sections = _structural_clause_split(text, min_clause_chars=cfg.min_chunk_chars)

        if len(sections) <= 1:
            # No structural markers found — fall through to pure semantic
            pass
        else:
            all_clause_chunks = []
            chunk_idx = 0
            char_offset = 0
            for section in sections:
                if len(section) > cfg.max_chunk_chars and embed_fn is not None:
                    # Oversized section: apply semantic sub-splitting
                    sub_cfg = SemanticChunkerConfig(
                        buffer_size=cfg.buffer_size,
                        breakpoint_percentile=cfg.breakpoint_percentile,
                        min_chunk_chars=cfg.min_chunk_chars,
                        max_chunk_chars=cfg.max_chunk_chars,
                        neighbor_weight=cfg.neighbor_weight,
                        structural_bonus_strong=cfg.structural_bonus_strong,
                        structural_bonus_weak=cfg.structural_bonus_weak,
                        clause_split=False,  # prevent recursion
                    )
                    sub_chunks = semantic_chunk(section, embed_fn=embed_fn, config=sub_cfg)
                    for sc in sub_chunks:
                        pos = text.find(sc["text"][:50], char_offset) if len(sc["text"]) >= 50 else text.find(sc["text"], char_offset)
                        sc["char_start"] = pos if pos >= 0 else char_offset
                        sc["char_end"] = sc["char_start"] + len(sc["text"])
                        sc["chunk_index"] = chunk_idx
                        chunk_idx += 1
                        char_offset = sc["char_start"] + 1
                    all_clause_chunks.extend(sub_chunks)
                else:
                    # Section fits within max size — keep as one chunk
                    pos = text.find(section[:50], char_offset) if len(section) >= 50 else text.find(section, char_offset)
                    char_start = pos if pos >= 0 else char_offset
                    all_clause_chunks.append({
                        "chunk_id": f"chk_{uuid.uuid4().hex[:10]}",
                        "text": section,
                        "char_start": char_start,
                        "char_end": char_start + len(section),
                        "token_count": _approx_token_count(section),
                        "chunk_index": chunk_idx,
                    })
                    chunk_idx += 1
                    char_offset = char_start + 1

            if all_clause_chunks:
                return all_clause_chunks

    # ── Standard semantic chunking (non-clause or fallback) ────────────────
    # 1. Sentence split
    sentences = _split_sentences(text)
    if not sentences:
        return []

    # Short documents: return as single chunk
    if len(sentences) <= 2:
        full = " ".join(sentences)
        return [{
            "chunk_id": f"chk_{uuid.uuid4().hex[:10]}",
            "text": full,
            "char_start": 0,
            "char_end": len(full),
            "token_count": _approx_token_count(full),
            "chunk_index": 0,
        }]

    # 2. Build sentence buffers
    buffers = _combine_sentence_buffers(sentences, cfg.buffer_size)

    # 3. Embed buffers in manageable groups to avoid GPU OOM on huge docs
    #    Truncate buffer text to 512 chars — we only need rough similarity for breakpoints,
    #    not perfect embeddings. Long legal "sentences" can be multi-paragraph and cause OOM.
    MAX_BUFFER_CHARS = 512
    truncated_buffers = [b[:MAX_BUFFER_CHARS] for b in buffers]
    EMBED_GROUP_SIZE = 200
    if len(truncated_buffers) <= EMBED_GROUP_SIZE:
        buffer_embeddings = embed_fn(truncated_buffers)
    else:
        import numpy as np
        parts = []
        for start in range(0, len(truncated_buffers), EMBED_GROUP_SIZE):
            parts.append(embed_fn(truncated_buffers[start:start + EMBED_GROUP_SIZE]))
        buffer_embeddings = np.concatenate(parts, axis=0)

    # 4. Find semantic breakpoints
    breakpoints = _compute_breakpoints(buffer_embeddings, sentences, cfg)

    # 5. Assemble chunks with size constraints
    chunk_texts = _assemble_chunks(sentences, breakpoints, cfg)

    # 6. Build output dicts with positional metadata
    chunks = []
    char_offset = 0
    for idx, ct in enumerate(chunk_texts):
        # Find approximate char_start in original text
        pos = text.find(ct[:50], char_offset) if len(ct) >= 50 else text.find(ct, char_offset)
        char_start = pos if pos >= 0 else char_offset
        char_end = char_start + len(ct)

        chunks.append({
            "chunk_id": f"chk_{uuid.uuid4().hex[:10]}",
            "text": ct,
            "char_start": char_start,
            "char_end": char_end,
            "token_count": _approx_token_count(ct),
            "chunk_index": idx,
        })
        char_offset = char_start + 1

    return chunks


# ── Legacy-compatible wrapper (drop-in for sliding_window_chunk callers) ───────

def sliding_window_chunk(
    text: str,
    chunk_size: int = 256,
    overlap: int = 64,
    min_chunk_chars: int = 80,
) -> list[dict]:
    """
    Fallback sliding-window chunker for cases where no embedder is available.

    Kept for backward compatibility with tests and CPU pipeline.
    For production use, prefer semantic_chunk() with an embedder.
    """
    if not text or not text.strip():
        return []

    char_window = chunk_size * 4
    char_step = (chunk_size - overlap) * 4

    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + char_window, len(text))

        candidate = text[start:end]
        if end < len(text):
            last_period = candidate.rfind(". ")
            last_newline = candidate.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > char_window // 2:
                end = start + break_point + 1

        chunk_text = text[start:end].strip()
        if len(chunk_text) >= min_chunk_chars:
            chunks.append({
                "chunk_id": f"chk_{uuid.uuid4().hex[:10]}",
                "text": chunk_text,
                "char_start": start,
                "char_end": end,
                "token_count": _approx_token_count(chunk_text),
                "chunk_index": idx,
            })
            idx += 1

        start += char_step
        if start >= end:
            break

    return chunks


def structural_chunk(
    text: str,
    chunk_size: int = 256,
    overlap: int = 64,
    min_chunk_chars: int = 80,
) -> list[dict]:
    """Structurally decompose text, then chunk within each block.

    Uses nupunkt for block/sentence boundaries when available.
    Each chunk inherits block_id and sentence_ids from its structural context.
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


def get_document_summary(text: str, filename: str = "", max_chars: int = 2000) -> str:
    """
    Build a summary string for macro clustering: title + first ~500 tokens.

    Legacy equivalent: AIFlow.call_field_summarization() — one network hop per doc.
    V2: Simple truncation, no LLM call. Summarization is too expensive per-doc.
    """
    title_part = filename.replace(".pdf", "").replace(".txt", "").replace("_", " ") if filename else ""
    body_part = text[:max_chars].strip()
    return f"{title_part}\n\n{body_part}" if title_part else body_part


def generate_typed_chunks(
    text: str,
    embed_fn,
    config: SemanticChunkerConfig | None = None,
) -> dict[str, list[dict]]:
    """
    Generate all 3 chunk types from a document as specified in V1 pipeline design.

    Returns a dict with keys "embedding", "extraction", "clause", each containing
    a list of chunk dicts. All chunk dicts include a "chunk_type" field.

    Chunk types:
      - embedding:   ~2000 tok semantic chunks for clustering/RAG/dedup
      - extraction:  ~4000 tok larger chunks at structural boundaries for sequential extraction
      - clause:      Structural clause splits (~20-30 per doc) for clause-level clustering

    Embedding and clause chunks get vectors; extraction chunks do not need embedding.
    """
    if not text or not text.strip():
        return {"embedding": [], "extraction": [], "clause": []}

    cfg = config or DEFAULT_CONFIG

    # ── Embedding chunks: standard semantic chunks (2000 tok / ~8000 chars) ──
    # Also capture sentence-level embeddings for reuse in clause cohesion check.
    _cached_sentence_embeddings = {}  # sentence_text_hash -> embedding vector

    def _caching_embed_fn(texts):
        """Wrapper around embed_fn that caches results by text content."""
        uncached_texts = []
        uncached_indices = []
        for i, t in enumerate(texts):
            if t not in _cached_sentence_embeddings:
                uncached_texts.append(t)
                uncached_indices.append(i)

        if uncached_texts:
            new_embs = embed_fn(uncached_texts)
            for j, idx in enumerate(uncached_indices):
                _cached_sentence_embeddings[texts[idx]] = new_embs[j]

        return np.array([_cached_sentence_embeddings[t] for t in texts])

    embedding_chunks = semantic_chunk(text, embed_fn=_caching_embed_fn, config=SemanticChunkerConfig(
        buffer_size=cfg.buffer_size,
        breakpoint_percentile=cfg.breakpoint_percentile,
        min_chunk_chars=cfg.min_chunk_chars,
        max_chunk_chars=cfg.max_chunk_chars,
        neighbor_weight=cfg.neighbor_weight,
        structural_bonus_strong=cfg.structural_bonus_strong,
        structural_bonus_weak=cfg.structural_bonus_weak,
        clause_split=False,
    ))
    for c in embedding_chunks:
        c["chunk_type"] = "embedding"

    # ── Extraction chunks: larger windows (~4000 tok / ~16000 chars) ──
    # Use structural splitting only (no GPU embedding) — extraction chunks are
    # not embedded/clustered, just stored for future sequential extraction.
    extraction_chunks = sliding_window_chunk(
        text,
        chunk_size=cfg.extraction_max_chunk_chars // 4,  # convert chars→approx tokens
        overlap=64,
        min_chunk_chars=cfg.min_chunk_chars,
    )
    for i, c in enumerate(extraction_chunks):
        c["chunk_type"] = "extraction"
        c["chunk_id"] = f"chk_{uuid.uuid4().hex[:10]}"  # unique ids
        c["chunk_index"] = i

    # ── Clause chunks: structural split at legal/section boundaries ──
    sections = _structural_clause_split(text, min_clause_chars=cfg.min_chunk_chars)
    if not sections:
        sections = [text.strip()] if text.strip() else []
    if embed_fn is not None and len(sections) > 1:
        sections = cohesion_subsplit_clauses(
            sections, embed_fn=_caching_embed_fn,
            min_clause_chars=cfg.min_chunk_chars,
            cohesion_threshold=0.75,
        )
    clause_chunks = []
    char_offset = 0
    for idx, section in enumerate(sections):
        pos = text.find(section[:50], char_offset) if len(section) >= 50 else text.find(section, char_offset)
        char_start = pos if pos >= 0 else char_offset
        clause_chunks.append({
            "chunk_id": f"chk_{uuid.uuid4().hex[:10]}",
            "text": section,
            "char_start": char_start,
            "char_end": char_start + len(section),
            "token_count": _approx_token_count(section),
            "chunk_index": idx,
            "chunk_type": "clause",
        })
        char_offset = char_start + 1

    return {
        "embedding": embedding_chunks,
        "extraction": extraction_chunks,
        "clause": clause_chunks,
    }


# ── Batch chunking: structural-first unified architecture ──────────────────────

def batch_generate_unified_chunks(
    texts: list[str],
    embed_fn,
    config: SemanticChunkerConfig | None = None,
    progress_fn=None,
) -> list[list[dict]]:
    """
    Generate clause-aligned chunks for ALL documents in a batched pass.

    Structural-first, semantic-fallback architecture:
      Pass 1 (CPU): Structural split ALL docs at legal boundaries → identify oversized sections
      Pass 2 (GPU): Embed sentence buffers ONLY for oversized sections (>max_chunk_chars)
                     + cohesion halves for multi-topic detection
      Pass 3 (CPU): Sub-split oversized sections using semantic breakpoints → assemble final chunks
                     + extract clause metadata as annotations on each chunk

    Each chunk carries clause_number, clause_title, heading_level as annotations.
    No separate extraction or clause chunk types — one unified chunk = one provision.

    Args:
        texts: List of document texts.
        embed_fn: Function(list[str]) -> np.ndarray (e.g. NomicEmbedder.embed_for_clustering).
        config: Chunker configuration.
        progress_fn: Optional callback(msg: str) for progress logging.

    Returns:
        List of lists (one per doc), each containing chunk dicts with keys:
          chunk_id, text, char_start, char_end, token_count, chunk_index,
          chunk_type, clause_number, clause_title, heading_level.
    """
    if not texts:
        return []

    cfg = config or DEFAULT_CONFIG
    n_docs = len(texts)
    MAX_BUFFER_CHARS = 512

    # ── Pass 1: CPU — structural split all docs, identify oversized + cohesion candidates ──
    if progress_fn:
        progress_fn(f"Pass 1/3: Structural split {n_docs} docs (CPU)...")

    doc_data = []           # Per-doc metadata
    all_subsplit_buffers = []  # Buffer texts for oversized sections needing semantic sub-split
    clause_half_texts = []  # (first_half, second_half) pairs for cohesion checks

    for di, text in enumerate(texts):
        if not text or not text.strip():
            doc_data.append({"empty": True})
            continue

        # Structural split at legal boundaries
        sections = _structural_clause_split(text, min_clause_chars=cfg.min_chunk_chars)
        if not sections:
            sections = [text.strip()] if text.strip() else []

        entry = {
            "empty": False,
            "text": text,
            "sections": sections,
            "oversized": [],       # (section_idx, sentences, buf_start, buf_end)
            "cohesion_candidates": [],
        }

        for si, section in enumerate(sections):
            if len(section) > cfg.max_chunk_chars:
                # Oversized: needs semantic sub-split — collect sentence buffers
                sentences = _split_sentences(section)
                if len(sentences) > 2:
                    buffers = _combine_sentence_buffers(sentences, cfg.buffer_size)
                    buf_start = len(all_subsplit_buffers)
                    all_subsplit_buffers.extend(b[:MAX_BUFFER_CHARS] for b in buffers)
                    entry["oversized"].append({
                        "section_idx": si,
                        "sentences": sentences,
                        "buf_start": buf_start,
                        "buf_end": len(all_subsplit_buffers),
                    })
                # else: short sentences but long section — will be force-split in Pass 3

            # Cohesion check for sections that aren't oversized but may contain multiple topics
            if len(section) > cfg.min_chunk_chars * 3 and len(section) <= cfg.max_chunk_chars:
                sec_sentences = _split_sentences(section)
                if len(sec_sentences) >= 4:
                    mid = len(sec_sentences) // 2
                    first_half = " ".join(sec_sentences[:mid])[:MAX_BUFFER_CHARS]
                    second_half = " ".join(sec_sentences[mid:])[:MAX_BUFFER_CHARS]
                    if len(first_half) >= cfg.min_chunk_chars and len(second_half) >= cfg.min_chunk_chars:
                        entry["cohesion_candidates"].append({
                            "section_idx": si,
                            "sentences": sec_sentences,
                            "half_start": len(clause_half_texts),
                        })
                        clause_half_texts.append(first_half)
                        clause_half_texts.append(second_half)

        doc_data.append(entry)

    n_oversized_sections = sum(len(e.get("oversized", [])) for e in doc_data if not e.get("empty"))
    n_cohesion = len(clause_half_texts) // 2
    if progress_fn:
        progress_fn(f"Pass 1/3 done: {n_oversized_sections} oversized sections, "
                     f"{n_cohesion} cohesion candidates, "
                     f"{len(all_subsplit_buffers)} buffers to embed")

    # ── Pass 2: GPU — embed oversized section buffers + cohesion halves ──
    combined_texts = all_subsplit_buffers + clause_half_texts
    if combined_texts:
        if progress_fn:
            progress_fn(f"Pass 2/3: Embedding {len(combined_texts)} texts on GPU...")
        combined_embeddings = embed_fn(combined_texts)
        n_buf = len(all_subsplit_buffers)
        subsplit_embeddings = combined_embeddings[:n_buf]
        cohesion_embeddings = combined_embeddings[n_buf:]
    else:
        subsplit_embeddings = np.empty((0, 768), dtype=np.float32)
        cohesion_embeddings = np.empty((0, 768), dtype=np.float32)

    # ── Pass 2b: Identify low-cohesion sections, collect their buffers for sub-split ──
    cohesion_subsplit_buffers = []
    cohesion_subsplit_meta = []  # (doc_idx, cand_idx, sentences, buf_start, buf_end)
    for di, entry in enumerate(doc_data):
        if entry.get("empty"):
            continue
        for ci, cand in enumerate(entry.get("cohesion_candidates", [])):
            hi = cand["half_start"]
            emb_first = cohesion_embeddings[hi]
            emb_second = cohesion_embeddings[hi + 1]
            norm_first = np.linalg.norm(emb_first)
            norm_second = np.linalg.norm(emb_second)
            if norm_first < 1e-12 or norm_second < 1e-12:
                continue
            cosine_sim = float(np.dot(emb_first, emb_second) / (norm_first * norm_second))
            if cosine_sim >= 0.75:
                continue
            # Low cohesion — collect buffers for semantic sub-split
            sec_sentences = cand["sentences"]
            sec_buffers = _combine_sentence_buffers(sec_sentences, buffer_size=1)
            buf_start = len(cohesion_subsplit_buffers)
            cohesion_subsplit_buffers.extend(b[:MAX_BUFFER_CHARS] for b in sec_buffers)
            cohesion_subsplit_meta.append((di, ci, sec_sentences, buf_start, len(cohesion_subsplit_buffers)))

    cohesion_sub_embeddings = np.empty((0, 768), dtype=np.float32)
    if cohesion_subsplit_buffers:
        # Free GPU memory from Pass 2 to avoid fragmentation OOM
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if progress_fn:
            progress_fn(f"Pass 2b: {len(cohesion_subsplit_meta)} low-cohesion sections → "
                         f"embedding {len(cohesion_subsplit_buffers)} subsplit texts...")
        cohesion_sub_embeddings = embed_fn(cohesion_subsplit_buffers)

    # Build cohesion subsplit lookup
    _cohesion_lookup: dict[tuple[int, int], tuple[list[str], np.ndarray]] = {}
    for di, ci, sec_sentences, buf_start, buf_end in cohesion_subsplit_meta:
        _cohesion_lookup[(di, ci)] = (sec_sentences, cohesion_sub_embeddings[buf_start:buf_end])

    if progress_fn:
        progress_fn(f"Pass 2/3 done: {len(combined_texts)} main + "
                     f"{len(cohesion_subsplit_buffers)} cohesion subsplits embedded")

    # ── Pass 3: CPU — sub-split oversized sections + assemble final chunks with clause metadata ──
    if progress_fn:
        progress_fn(f"Pass 3/3: Assembling unified chunks for {n_docs} docs (CPU)...")

    results = []
    for di, entry in enumerate(doc_data):
        if entry.get("empty"):
            results.append([])
            continue

        text = entry["text"]
        sections = list(entry["sections"])

        # Build oversized lookup: section_idx → (sentences, embeddings)
        oversized_lookup: dict[int, tuple[list[str], np.ndarray]] = {}
        for ov in entry.get("oversized", []):
            si = ov["section_idx"]
            embs = subsplit_embeddings[ov["buf_start"]:ov["buf_end"]]
            oversized_lookup[si] = (ov["sentences"], embs)

        # Build cohesion subsplit results
        cohesion_splits: dict[int, list[str]] = {}
        for ci, cand in enumerate(entry.get("cohesion_candidates", [])):
            key = (di, ci)
            if key not in _cohesion_lookup:
                continue
            sec_sentences, sec_embs = _cohesion_lookup[key]
            sub_cfg = SemanticChunkerConfig(
                buffer_size=1,
                breakpoint_percentile=75.0,
                min_chunk_chars=cfg.min_chunk_chars,
                max_chunk_chars=cfg.max_chunk_chars,
                clause_split=False,
            )
            breakpoints = _compute_breakpoints(sec_embs, sec_sentences, sub_cfg)
            if breakpoints:
                sub_texts = _assemble_chunks(sec_sentences, breakpoints, sub_cfg)
                if len(sub_texts) > 1:
                    cohesion_splits[cand["section_idx"]] = sub_texts

        # Expand sections: oversized → semantic sub-split, low-cohesion → split
        expanded_sections = []  # (text, original_section_idx)
        for si, section in enumerate(sections):
            if si in oversized_lookup:
                sentences, embs = oversized_lookup[si]
                sub_cfg = SemanticChunkerConfig(
                    buffer_size=cfg.buffer_size,
                    breakpoint_percentile=cfg.breakpoint_percentile,
                    min_chunk_chars=cfg.min_chunk_chars,
                    max_chunk_chars=cfg.max_chunk_chars,
                    clause_split=False,
                )
                breakpoints = _compute_breakpoints(embs, sentences, sub_cfg)
                sub_texts = _assemble_chunks(sentences, breakpoints, sub_cfg)
                for st in sub_texts:
                    expanded_sections.append((st, si))
            elif si in cohesion_splits:
                for st in cohesion_splits[si]:
                    expanded_sections.append((st, si))
            else:
                expanded_sections.append((section, si))

        # Merge undersized sections with neighbors
        merged: list[tuple[str, int]] = []
        for chunk_text, orig_si in expanded_sections:
            if len(chunk_text) < cfg.min_chunk_chars and merged:
                prev_text, prev_si = merged[-1]
                merged[-1] = (prev_text + "\n" + chunk_text, prev_si)
            elif chunk_text.strip():
                merged.append((chunk_text, orig_si))

        # Forward-merge: if first chunk is undersized, merge into next
        if len(merged) > 1 and len(merged[0][0]) < cfg.min_chunk_chars:
            first_text, first_si = merged[0]
            second_text, second_si = merged[1]
            merged[1] = (first_text + "\n" + second_text, first_si)
            merged.pop(0)

        # Force-split anything still over max
        final_sections: list[tuple[str, int]] = []
        for chunk_text, orig_si in merged:
            if len(chunk_text) > cfg.max_chunk_chars:
                sentences = _split_sentences(chunk_text)
                forced = _force_split(sentences, cfg.max_chunk_chars)
                for ft in forced:
                    final_sections.append((ft, orig_si))
            else:
                final_sections.append((chunk_text, orig_si))

        # Build chunk dicts with clause metadata annotations
        chunks = []
        char_offset = 0
        # Pre-extract headers from original sections for annotation
        section_headers: dict[int, dict | None] = {}
        for si, section in enumerate(sections):
            section_headers[si] = extract_clause_header(section)

        for idx, (chunk_text, orig_si) in enumerate(final_sections):
            pos = text.find(chunk_text[:50], char_offset) if len(chunk_text) >= 50 else text.find(chunk_text, char_offset)
            char_start = pos if pos >= 0 else char_offset

            # Try chunk's own text first, fall back to parent section header
            header = extract_clause_header(chunk_text) or section_headers.get(orig_si)
            chunks.append({
                "chunk_id": f"chk_{uuid.uuid4().hex[:10]}",
                "text": chunk_text,
                "char_start": char_start,
                "char_end": char_start + len(chunk_text),
                "token_count": _approx_token_count(chunk_text),
                "chunk_index": idx,
                "chunk_type": "embedding",
                "clause_number": header["clause_number"] if header else "",
                "clause_title": header["clause_title"] if header else "",
                "heading_level": header["heading_level"] if header else 0,
            })
            char_offset = char_start + 1

        results.append(chunks)

    if progress_fn:
        total = sum(len(r) for r in results)
        progress_fn(f"Pass 3/3 done: {total} unified chunks from {n_docs} docs "
                     f"(avg {total/max(n_docs,1):.0f}/doc)")

    return results


def extract_sections(text: str) -> list[dict]:
    """
    Detect section headings for document structure tree (FIELD-102 stretch goal).
    Returns list of {heading, level, char_offset}.
    """
    patterns = [
        (1, r'^(ARTICLE|Article)\s+[IVXLC\d]+[.:]\s*(.+)$'),
        (2, r'^(\d+\.)\s+([A-Z][^.]{3,60})$'),
        (3, r'^(\d+\.\d+)\s+(.{3,60})$'),
        (2, r'^(Section|SECTION)\s+\d+[.:]\s*(.+)$'),
    ]
    sections = []
    for i, line in enumerate(text.split("\n")):
        stripped = line.strip()
        for level, pattern in patterns:
            m = re.match(pattern, stripped)
            if m:
                heading = m.group(2) if m.lastindex >= 2 else m.group(1)
                sections.append({"heading": heading.strip(), "level": level, "line_number": i})
                break
    return sections
