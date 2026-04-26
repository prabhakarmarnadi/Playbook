"""
Legal-aware semantic chunker — Greg Kamradt method + legal structural signals.

Ported from legacy SemanticChunker in apr_fields_processor, adapted for:
  - Nomic embedder (GPU, synchronous) instead of async AIFlow gRPC
  - No external dependencies on blingfire (uses regex sentence splitting)
  - Same algorithm: sentence buffers → embed → cosine distance → neighbor
    smoothing → legal header boosting → percentile threshold → chunk

How it works:
  1. Split document into sentences (regex, handles legal abbreviations)
  2. Group sentences into overlapping buffers for context
  3. Embed each buffer with Nomic (or any embedder)
  4. Compute cosine distance between consecutive buffer embeddings
  5. Smooth distances with neighbor averaging (reduces noise)
  6. Boost distances at legal structural boundaries (ARTICLE, Section, etc.)
  7. Split where combined distance > percentile threshold
  8. Enforce min/max chunk size constraints
"""
import re
import uuid
from dataclasses import dataclass

import numpy as np

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
    embedding_chunks = semantic_chunk(text, embed_fn=embed_fn, config=SemanticChunkerConfig(
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
    extraction_cfg = SemanticChunkerConfig(
        buffer_size=cfg.buffer_size,
        breakpoint_percentile=95.0,  # fewer breakpoints → larger chunks
        min_chunk_chars=cfg.min_chunk_chars,
        max_chunk_chars=cfg.extraction_max_chunk_chars,
        neighbor_weight=cfg.neighbor_weight,
        structural_bonus_strong=cfg.structural_bonus_strong,
        structural_bonus_weak=cfg.structural_bonus_weak,
        clause_split=False,
    )
    extraction_chunks = semantic_chunk(text, embed_fn=embed_fn, config=extraction_cfg)
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
            sections, embed_fn=embed_fn,
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
