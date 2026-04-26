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

_NUPUNKT_AVAILABLE = False
try:
    import nupunkt
    _NUPUNKT_AVAILABLE = True
    logger.info("nupunkt detected — using legal-aware sentence tokenization")
except ImportError:
    logger.warning("nupunkt not available — falling back to regex sentence splitting")


def nupunkt_available() -> bool:
    return _NUPUNKT_AVAILABLE


_ABBREV_RE = re.compile(
    r"(?:Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Corp|Co|vs|etc|i\.e|e\.g|cf|viz|al|No|Art|Sec|U\.S|U\.K|E\.U)\."
)


def _regex_split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    protected = _ABBREV_RE.sub(lambda m: m.group().replace(".", "[DOT]"), text)
    protected = re.sub(r"(\d)\.", r"\1[DOT]", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\(\"'])", protected)
    sentences = [p.replace("[DOT]", ".").strip() for p in parts if p.strip()]
    return sentences


def _nupunkt_split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return nupunkt.sent_tokenize(text)


def _split_into_blocks(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def _detect_block_type(block_text: str) -> str:
    stripped = block_text.strip()
    if re.match(r"^(?:ARTICLE|SECTION|Part)\s", stripped, re.IGNORECASE):
        return "heading"
    if re.match(r'^"[^"]+"\s+(?:means|shall mean|refers to)', stripped):
        return "definition"
    if re.match(r"^(?:WHEREAS|RECITAL|PREAMBLE)", stripped, re.IGNORECASE):
        return "recital"
    return "clause_body"


def _extract_section_number(block_text: str) -> str | None:
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
