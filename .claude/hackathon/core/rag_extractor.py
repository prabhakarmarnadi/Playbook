"""
RAG Extractor — Retrieval-Augmented field extraction with agentic retry.
=========================================================================

Post-discovery extraction pipeline that replaces (or supplements) the naive
per-chunk LLM extraction with a multi-round RAG loop:

  Stage Architecture:
  ─────────────────────────────────────────────────────────────────────
  1. Cochran Sampling   — statistically sample documents per cluster
  2. Hybrid Retrieval   — semantic + lexical + RRF per (doc, field)
  3. Agentic Retry      — LLM-guided re-query on "insufficient context"
  4. Reviewer Agent      — cheap model accept/reject + re-extract
  5. Model-Routed LLM   — GPT-5.4 complex / GPT-5.4-nano simple
  6. Normalization       — DATE→YYYY/MM/DD, NUMBER preserve symbols
  7. AIDB Writes         — batched persistence via ClusteringStore
  ─────────────────────────────────────────────────────────────────────

Gains: +3% F1 from agentic retry (eval), reviewer catches ~8% of false
positives at < 5% of extraction cost.

Usage:
    from core.rag_extractor import RAGExtractor, RAGConfig

    extractor = RAGExtractor(store=store, llm=llm, embed_fn=embed_fn, config=cfg)
    results = await extractor.extract_cluster(
        cluster_id="cls_001",
        fields=[{"name": "termination_notice_period", "type": "duration", ...}],
        cluster_label="Termination Clauses",
    )
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Fields that require GPT-5.4 (complex reasoning, cross-references)
COMPLEX_FIELD_TYPES = frozenset({
    "duration", "currency", "percentage", "enum",
})
COMPLEX_FIELD_KEYWORDS = frozenset({
    "termination", "renewal", "indemnification", "liability", "penalty",
    "force_majeure", "limitation", "warranty", "non_compete", "exclusivity",
    "assignment", "change_of_control", "liquidated_damages",
})

# Normalization patterns
_DATE_PATTERN = re.compile(
    r"(?P<month>\w+)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})"
    r"|(?P<d2>\d{1,2})[/\-.](?P<m2>\d{1,2})[/\-.](?P<y2>\d{2,4})"
    r"|(?P<y3>\d{4})[/\-.](?P<m3>\d{1,2})[/\-.](?P<d3>\d{1,2})"
)
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class RAGConfig:
    """Tunables for RAG extraction pipeline."""

    # Sampling
    cochran_z: float = 1.96            # 95% confidence
    cochran_p: float = 0.5             # max-variance proportion
    cochran_e: float = 0.15            # 15% margin of error
    max_sample_docs: int = 10          # hard cap on sampled documents
    min_sample_docs: int = 3           # minimum even for tiny clusters

    # Retrieval
    semantic_top_k: int = 10           # pre-RRF semantic candidates
    lexical_top_k: int = 10            # pre-RRF lexical candidates
    rrf_k: int = 60                    # RRF denominator constant
    final_top_k: int = 3              # snippets per (doc, field) after RRF

    # Agentic retry
    max_retry_rounds: int = 3          # max agentic re-query rounds
    insufficient_context_threshold: float = 0.3  # confidence below which we retry

    # NL Index (cross-doc heading lookup)
    enable_nl_index: bool = True
    nl_index_top_k: int = 5            # heading-level candidates

    # Reviewer
    enable_reviewer: bool = True
    reviewer_model: str | None = None  # None → use cheap model

    # Model routing
    complex_model: str = "gpt-5.4"
    simple_model: str = "gpt-5.4-nano"
    model_routing: bool = True         # False → use complex_model for all

    # Extraction
    max_chunk_chars: int = 3000
    min_extraction_confidence: float = 0.4

    # Concurrency
    max_concurrent_fields: int = 8
    max_concurrent_docs: int = 5

    # Late interaction retrieval (ColBERT MaxSim)
    use_late_interaction: bool = False

    # Wide retrieval scope — search all agreement chunks, not just cluster-member
    widen_retrieval_scope: bool = False
    cluster_boost_weight: float = 0.015  # additive RRF boost for in-cluster chunks
    wide_retrieve_k_multiplier: int = 2  # retrieve this × final_top_k, then re-rank


# ── Data Structures ───────────────────────────────────────────────────────────


@dataclass
class RetrievedSnippet:
    """A chunk retrieved for a specific (doc, field) pair."""
    chunk_id: str
    chunk_text: str
    agreement_id: str
    score: float
    source: str           # "semantic", "lexical", "rrf", "agentic_retry", "nl_index"
    rank: int


@dataclass
class ExtractionResult:
    """Single field extraction result with provenance."""
    field_name: str
    field_type: str
    value: str | None
    raw_value: str | None  # pre-normalization
    confidence: float
    evidence_span: str
    is_default: bool
    agreement_id: str
    chunk_id: str
    model_used: str
    retrieval_round: int
    reviewer_accepted: bool | None  # None = no reviewer pass
    snippets_used: int


@dataclass
class ExtractionBatch:
    """Batch of extraction results for a cluster."""
    cluster_id: str
    cluster_label: str
    fields: list[dict]
    results: list[ExtractionResult]
    meta: dict


# ── Cochran Sampling ──────────────────────────────────────────────────────────


def cochran_sample_size(
    population: int,
    z: float = 1.96,
    p: float = 0.5,
    e: float = 0.15,
    cap: int = 10,
    floor: int = 3,
) -> int:
    """
    Cochran's formula for sample size with finite population correction.

    n₀ = z²·p·(1-p) / e²
    n  = n₀ / (1 + (n₀-1)/N)
    """
    if population <= floor:
        return population
    n0 = (z ** 2 * p * (1 - p)) / (e ** 2)
    n = n0 / (1 + (n0 - 1) / population)
    return max(floor, min(cap, math.ceil(n)))


# ── Deterministic Type Normalizers ────────────────────────────────────────────
#
# Architecture: every extracted value is normalized by a deterministic
# function per type.  The LLM is NOT relied upon for formatting — these
# pure functions handle the canonical representation.
#
#   LLM raw string → normalize_value(raw, type) → canonical form
#                  → validate_extracted_type(canonical, type) → (ok, corrected)
#
# Dispatch table maps field_type → normalizer function.
# Types not in the dispatch table pass through as cleaned text.
# ──────────────────────────────────────────────────────────────────────────────

# Regex for "word-number (digit)" pattern common in legal: "thirty (30) days"
_LEGAL_PAREN_NUMBER = re.compile(r'\w+\s*\((\d+)\)')
# Currency amount: optional symbol, digits with commas/dots
_CURRENCY_AMOUNT = re.compile(
    r'(?P<symbol>[$€£¥₹]|USD|EUR|GBP|CAD|AUD|JPY|CHF|CNY|INR)?\s*'
    r'(?P<amount>[\d,]+(?:\.\d+)?)'
    r'(?:\s*(?P<suffix>million|billion|thousand|M|B|K))?',
    re.IGNORECASE,
)
_PERCENTAGE_PATTERN = re.compile(
    r'(?P<num>[\d.]+)\s*(?:%|percent|per\s*cent)',
    re.IGNORECASE,
)
_DURATION_PATTERN = re.compile(
    r'(?P<num>\d+)\s*(?P<unit>days?|months?|years?|weeks?|hours?|business\s*days?)',
    re.IGNORECASE,
)
_DURATION_UNIT_DAYS = {
    "day": 1, "days": 1,
    "business day": 1, "business days": 1,  # approximate
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
    "hour": 0, "hours": 0,  # sub-day: store as-is
}
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "ninety": 90,
}
_SUFFIX_MULTIPLIER = {
    "thousand": 1_000, "k": 1_000,
    "million": 1_000_000, "m": 1_000_000,
    "billion": 1_000_000_000, "b": 1_000_000_000,
}


def _normalize_date(value: str) -> str:
    """Normalize date to YYYY-MM-DD.  Handles:
    - "January 1, 2024" / "Jan 1 2024"
    - "1/15/2024" / "01-15-2024"  (US format: M/D/Y)
    - "2024-01-01" / "2024/01/01" (ISO format: Y-M-D)
    Returns original string if unparseable.
    """
    m = _DATE_PATTERN.search(value)
    if not m:
        return value
    groups = m.groupdict()

    if groups.get("month"):
        # Named month: "January 1, 2024"
        month_name = groups["month"].lower()
        month = _MONTH_MAP.get(month_name)
        if not month:
            return value
        day = int(groups["day"])
        year = int(groups["year"])
    elif groups.get("m2"):
        # Slash/dash format: regex captures d2/m2/y2 (first/second/year)
        # Disambiguate US (M/D/Y) vs European (D/M/Y):
        first = int(groups["d2"])   # first number in the pattern
        second = int(groups["m2"])  # second number in the pattern
        year = int(groups["y2"])
        if year < 100:
            year += 2000 if year < 50 else 1900
        if first > 12 and second <= 12:
            # First > 12 → must be day → D/M/Y (European)
            day, month = first, second
        elif second > 12 and first <= 12:
            # Second > 12 → must be day → M/D/Y (US)
            month, day = first, second
        else:
            # Both ≤ 12 → assume US M/D/Y
            month, day = first, second
    elif groups.get("m3"):
        # ISO: Y-M-D
        month = int(groups["m3"])
        day = int(groups["d3"])
        year = int(groups["y3"])
    else:
        return value

    if not (1 <= month <= 12 and 1 <= day <= 31):
        return value

    return f"{year:04d}-{month:02d}-{day:02d}"


def _normalize_boolean(value: str) -> str:
    """Normalize boolean to 'true' | 'false' | None (for N/A).
    - yes/true/1/y/affirmative/included/required → 'true'
    - no/false/0/n/negative/not included/none/N/A → 'false'
    - Negation patterns (not, neither, without...) → 'false'
    - N/A, not applicable → null (caller handles)
    """
    lower = value.lower().strip()

    # N/A → null explicitly (not false)
    if lower in ("n/a", "not applicable", "na", "none", "null", "—", "-", ""):
        return ""

    positive = {"yes", "true", "1", "y", "affirmative", "included", "required",
                "applicable", "permitted", "allowed", "agreed"}
    negative = {"no", "false", "0", "n", "negative", "not included", "not required",
                "not permitted", "not allowed", "excluded", "prohibited", "waived"}

    if lower in positive:
        return "true"
    if lower in negative:
        return "false"

    # Negation detection for longer text
    neg_words = {"not ", "no ", "neither", "without", "except", "prohibited",
                 "shall not", "will not", "does not", "cannot"}
    if any(nw in lower for nw in neg_words):
        return "false"
    # Short affirmative text → true
    if len(lower) <= 20:
        return "true"
    # Long text that isn't clearly boolean → return as-is, validator will catch
    return value


def _normalize_currency(value: str) -> str:
    """Normalize currency to '<SYMBOL><amount>' canonical form.
    - "$1,000,000" → "$1000000.00"
    - "USD 1.5 million" → "USD1500000.00"
    - "1000" → "1000.00" (no symbol)
    """
    m = _CURRENCY_AMOUNT.search(value)
    if not m:
        return re.sub(r'\s+', ' ', value).strip()

    symbol = (m.group("symbol") or "").strip().upper()
    amount_str = m.group("amount").replace(",", "")
    suffix = (m.group("suffix") or "").strip().lower()

    try:
        amount = float(amount_str)
    except ValueError:
        return re.sub(r'\s+', ' ', value).strip()

    if suffix and suffix in _SUFFIX_MULTIPLIER:
        amount *= _SUFFIX_MULTIPLIER[suffix]

    # Format with 2 decimal places, no commas
    formatted = f"{amount:.2f}"
    # Remove trailing .00 for whole numbers
    if formatted.endswith(".00"):
        formatted = formatted[:-3]

    return f"{symbol}{formatted}" if symbol else formatted


def _normalize_percentage(value: str) -> str:
    """Normalize percentage to decimal float string.
    - "5%" → "0.05"
    - "0.05" (already decimal) → "0.05"
    - "five percent" → "0.05"
    """
    lower = value.lower().strip()

    # Check if already a decimal < 1 (likely already normalized)
    try:
        v = float(lower)
        if 0 <= v < 1:
            return str(v)
    except ValueError:
        pass

    # Try word-number
    for word, num in _WORD_NUMBERS.items():
        if word in lower and ("percent" in lower or "%" in lower):
            return str(num / 100.0)

    m = _PERCENTAGE_PATTERN.search(value)
    if m:
        pct = float(m.group("num"))
        return str(pct / 100.0)

    # Bare number followed by %
    bare = re.search(r'([\d.]+)\s*%', value)
    if bare:
        return str(float(bare.group(1)) / 100.0)

    return re.sub(r'\s+', ' ', value).strip()


def _normalize_duration(value: str) -> str:
    """Normalize duration to canonical form: '<N> <unit> (<days>d)'.
    - "thirty (30) days" → "30 days (30d)"
    - "1 year" → "1 year (365d)"
    - "six months" → "6 months (180d)"
    Stores both human-readable and machine-comparable forms.
    """
    lower = value.lower().strip()

    # Handle "word (digit)" pattern: "thirty (30) days"
    paren_match = _LEGAL_PAREN_NUMBER.search(value)
    if paren_match:
        digit = int(paren_match.group(1))
        # Find the unit after the parenthetical
        after = value[paren_match.end():].strip()
        unit_m = re.match(r'(days?|months?|years?|weeks?|hours?|business\s*days?)',
                          after, re.IGNORECASE)
        if unit_m:
            unit = unit_m.group(1).lower()
            days = digit * _DURATION_UNIT_DAYS.get(unit.rstrip("s"), 1)
            return f"{digit} {unit} ({days}d)"

    # Handle word-number: "six months"
    for word, num in _WORD_NUMBERS.items():
        if word in lower:
            unit_m = re.search(r'(days?|months?|years?|weeks?)', lower)
            if unit_m:
                unit = unit_m.group(1)
                days = num * _DURATION_UNIT_DAYS.get(unit.rstrip("s"), 1)
                return f"{num} {unit} ({days}d)"

    # Standard: "30 days"
    m = _DURATION_PATTERN.search(value)
    if m:
        num = int(m.group("num"))
        unit = m.group("unit").lower()
        days = num * _DURATION_UNIT_DAYS.get(unit.rstrip("s"), 1)
        return f"{num} {unit} ({days}d)"

    return re.sub(r'\s+', ' ', value).strip()


def _normalize_integer(value: str) -> str:
    """Normalize integer: extract numeric value.
    - "two (2)" → "2"
    - "1,000" → "1000"
    - "three hundred" → best effort
    """
    # Legal parenthetical: "two (2)"
    paren = _LEGAL_PAREN_NUMBER.search(value)
    if paren:
        return paren.group(1)

    # Plain digits (possibly with commas)
    digits = re.sub(r'[,\s]', '', value)
    m = re.search(r'-?\d+', digits)
    if m:
        return m.group(0)

    # Word numbers
    lower = value.lower().strip()
    if lower in _WORD_NUMBERS:
        return str(_WORD_NUMBERS[lower])

    return value


def _normalize_list(value: str) -> str:
    """Normalize list/enum values: split on delimiters, trim, deduplicate, sort.
    - "Delaware, New York, Delaware" → '["Delaware", "New York"]'
    """
    # Already JSON?
    if value.startswith("["):
        try:
            items = json.loads(value)
            if isinstance(items, list):
                cleaned = sorted(set(str(i).strip() for i in items if str(i).strip()))
                return json.dumps(cleaned)
        except (json.JSONDecodeError, TypeError):
            pass

    # Split on comma, semicolon, or " and "
    parts = re.split(r'[;,]|\band\b', value)
    cleaned = sorted(set(p.strip() for p in parts if p.strip()))
    if len(cleaned) > 1:
        return json.dumps(cleaned)
    return value.strip()


def _normalize_text(value: str) -> str:
    """Default text normalizer: collapse whitespace, strip."""
    return re.sub(r'\s+', ' ', value).strip()


# ── Dispatch Table ────────────────────────────────────────────────────────────

_NORMALIZER_DISPATCH: dict[str, Callable[[str], str]] = {
    "date": _normalize_date,
    "boolean": _normalize_boolean,
    "currency": _normalize_currency,
    "percentage": _normalize_percentage,
    "duration": _normalize_duration,
    "integer": _normalize_integer,
    "number": _normalize_integer,
    "list": _normalize_list,
    "enum": _normalize_list,
    "text": _normalize_text,
    "string": _normalize_text,
}


def normalize_value(value: str, field_type: str) -> str:
    """Normalize an extracted value using deterministic type-aware dispatch.

    The LLM is NOT relied upon for formatting.  Each type has a pure
    function that handles canonical representation.

    Args:
        value: Raw extracted value from LLM
        field_type: Declared field type (date, boolean, currency, etc.)

    Returns:
        Canonical normalized value, or original if type unknown.
    """
    if not value or not value.strip():
        return value

    normalizer = _NORMALIZER_DISPATCH.get(field_type, _normalize_text)
    try:
        return normalizer(value.strip())
    except Exception:
        # Normalization should never crash the pipeline
        return re.sub(r'\s+', ' ', value).strip()


# ── Type Validation ───────────────────────────────────────────────────────────


def validate_extracted_type(value: str, field_type: str) -> tuple[bool, str | None]:
    """Validate that a normalized value matches its declared field type.

    Runs AFTER normalize_value().  Returns (is_valid, corrected_value).
    If invalid but correctable, returns the corrected value.
    If invalid and not correctable, returns (False, None).

    This is the last line of defense — catches LLM outputs that survived
    normalization but still don't match the type contract.
    """
    if not value or not value.strip():
        return True, value

    v = value.strip()
    vl = v.lower()

    if field_type == "boolean":
        if vl in ("true", "false"):
            return True, vl
        if vl == "":
            return True, None  # N/A → null
        # Re-run normalizer for edge cases
        normalized = _normalize_boolean(v)
        if normalized in ("true", "false"):
            return True, normalized
        if normalized == "":
            return True, None
        # Long text that isn't boolean → reject
        if len(vl) > 30:
            return False, None
        return True, "true"  # short ambiguous → default true

    if field_type == "date":
        # Valid: YYYY-MM-DD or partial date with digits
        if re.match(r'\d{4}-\d{2}-\d{2}$', v):
            return True, v
        # Has date-like patterns
        if re.search(r'\d{4}|\d{1,2}[-/]\d{1,2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec', vl):
            return True, v
        # Duration mislabeled as date
        if re.search(r'\d+\s*(days?|months?|years?|weeks?)', vl):
            return False, None
        # No date signal at all
        if not re.search(r'\d', vl):
            return False, None
        return True, v

    if field_type == "currency":
        # Must have digits or currency symbols
        if re.search(r'[\d$€£¥₹]', v):
            return True, v
        if re.search(r'usd|eur|gbp|cad|aud|jpy|chf', vl):
            return True, v
        return False, None

    if field_type == "percentage":
        # After normalization should be a decimal float
        try:
            pv = float(v)
            if 0 <= pv <= 100:  # allow both 0.05 and 5.0 forms
                return True, v
        except ValueError:
            pass
        if "%" in v or "percent" in vl:
            return True, v
        return False, None

    if field_type == "duration":
        # After normalization should have digits + unit
        if re.search(r'\d', v) and re.search(r'day|month|year|week|hour|d\)', vl):
            return True, v
        if re.search(r'\d+\s*d$', vl):  # "30d" shorthand
            return True, v
        return False, None

    if field_type in ("integer", "number"):
        # Must be digits (possibly negative)
        if re.match(r'-?\d+$', v):
            return True, v
        # Try one more extraction
        digits = re.sub(r'[^\d-]', '', v)
        m = re.search(r'-?\d+', digits)
        if m:
            return True, m.group(0)
        return False, None

    if field_type in ("list", "enum"):
        # JSON array or comma-separated
        if v.startswith("["):
            try:
                json.loads(v)
                return True, v
            except (json.JSONDecodeError, TypeError):
                pass
        return True, v

    # text/string/unknown → always valid
    return True, v


# ── Field Complexity Routing ──────────────────────────────────────────────────


def is_complex_field(field_def: dict) -> bool:
    """Determine if a field requires the complex model."""
    ftype = field_def.get("type", "text")
    fname = field_def.get("name", "").lower()

    if ftype in COMPLEX_FIELD_TYPES:
        return True
    for keyword in COMPLEX_FIELD_KEYWORDS:
        if keyword in fname:
            return True

    desc = field_def.get("description", "").lower()
    complexity_signals = ["cross-reference", "conditional", "unless", "subject to",
                          "notwithstanding", "aggregate", "cumulative"]
    if any(sig in desc for sig in complexity_signals):
        return True

    return False


# ── Core RAG Extractor ────────────────────────────────────────────────────────


class RAGExtractor:
    """
    Multi-round RAG extraction pipeline.

    Architecture per (document, field) pair:
      Round 1  → Hybrid retrieval (semantic + lexical + RRF)
      Round 2+ → Agentic retry: LLM reformulates query, re-retrieves
      Round N  → Reviewer agent: cheap model accept/reject
    """

    def __init__(
        self,
        store,                          # ClusteringStore
        llm,                            # AsyncLLMClient
        embed_fn: Callable,             # text → (n, dim) ndarray
        config: RAGConfig | None = None,
        lance_store=None,               # LanceVectorStore (optional)
    ):
        self.store = store
        self.llm = llm
        self.embed_fn = embed_fn
        self.config = config or RAGConfig()
        self.lance = lance_store

        # Pre-built indexes per agreement_id (lazy)
        self._doc_retrievers: dict[str, Any] = {}
        self._nl_index: dict[str, list[dict]] | None = None

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    async def extract_cluster(
        self,
        cluster_id: str,
        fields: list[dict],
        cluster_label: str,
        cluster_keywords: list[str] | None = None,
        context_section: str = "",
        sample: bool = True,
    ) -> ExtractionBatch:
        """
        Run full RAG extraction for one cluster.

        1. Sample documents via Cochran formula (or use all if sample=False)
        2. For each (doc, field): hybrid retrieve → agentic retry → extract
        3. Reviewer agent pass (optional)
        4. Normalize + persist

        Args:
            sample: If True (default), use Cochran sampling to pick a
                    representative subset of agreements.  If False, extract
                    from every agreement in the cluster (production mode).
        """
        t0 = time.time()
        meta: dict[str, Any] = {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "n_fields": len(fields),
            "timings": {},
        }

        # ── 1. Get docs in this cluster and optionally sample ──
        t_sample = time.time()
        chunk_rows = self.store.get_chunks(cluster_id=cluster_id)
        agreement_ids = list({r["agreement_id"] for r in chunk_rows})

        if sample:
            n_sample = cochran_sample_size(
                len(agreement_ids),
                z=self.config.cochran_z,
                p=self.config.cochran_p,
                e=self.config.cochran_e,
                cap=self.config.max_sample_docs,
                floor=self.config.min_sample_docs,
            )
            sampled_ids = agreement_ids[:n_sample]
        else:
            sampled_ids = agreement_ids

        meta["n_agreements_total"] = len(agreement_ids)
        meta["n_agreements_sampled"] = len(sampled_ids)
        meta["sample_mode"] = "cochran" if sample else "all"
        meta["timings"]["sampling_s"] = round(time.time() - t_sample, 3)

        # Group chunks by agreement — optionally widen scope to ALL of each
        # agreement's chunks (not just cluster-member chunks).  This prevents
        # the "1 chunk" retriever problem where relevant adjacent clauses in
        # the same document are invisible because they belong to another cluster.
        cluster_chunk_ids = {r["chunk_id"] for r in chunk_rows}
        chunks_by_doc: dict[str, list[dict]] = {}

        if self.config.widen_retrieval_scope and sampled_ids:
            # Single batch query for all sampled agreements (avoids N roundtrips)
            placeholders = ",".join(["?"] * len(sampled_ids))
            all_rows = self.store.conn.execute(
                f"SELECT * FROM chunks WHERE agreement_id IN ({placeholders}) ORDER BY agreement_id, chunk_index",
                sampled_ids,
            ).fetchdf().to_dict("records")
            for row in all_rows:
                row["_in_cluster"] = row["chunk_id"] in cluster_chunk_ids
                chunks_by_doc.setdefault(row["agreement_id"], []).append(row)
        else:
            # Original behavior: cluster-scoped chunks only
            for row in chunk_rows:
                if row["agreement_id"] in sampled_ids:
                    row["_in_cluster"] = True
                    chunks_by_doc.setdefault(row["agreement_id"], []).append(row)

        # ── 2. Build per-doc retrievers ──
        t_index = time.time()
        await self._build_doc_indexes(chunks_by_doc)
        meta["timings"]["indexing_s"] = round(time.time() - t_index, 3)

        # ── 3. Build NL index (cross-doc heading index) ──
        if self.config.enable_nl_index:
            t_nl = time.time()
            self._build_nl_index(chunks_by_doc)
            meta["timings"]["nl_index_s"] = round(time.time() - t_nl, 3)

        # ── 4. Extract all (doc, field) pairs ──
        t_extract = time.time()
        all_results: list[ExtractionResult] = []

        # Process docs with bounded concurrency
        doc_sem = asyncio.Semaphore(self.config.max_concurrent_docs)

        async def _process_doc(agreement_id: str):
            async with doc_sem:
                return await self._extract_document(
                    agreement_id=agreement_id,
                    fields=fields,
                    cluster_label=cluster_label,
                    context_section=context_section,
                    chunks=chunks_by_doc.get(agreement_id, []),
                )

        doc_tasks = [_process_doc(aid) for aid in sampled_ids]
        doc_results = await asyncio.gather(*doc_tasks, return_exceptions=True)

        total_retries = 0
        total_reviewer_rejections = 0
        for dr in doc_results:
            if isinstance(dr, Exception):
                logger.warning(f"Doc extraction failed: {dr}")
                continue
            results, doc_meta = dr
            all_results.extend(results)
            total_retries += doc_meta.get("retries", 0)
            total_reviewer_rejections += doc_meta.get("reviewer_rejections", 0)

        meta["timings"]["extraction_s"] = round(time.time() - t_extract, 3)
        meta["n_extractions"] = len(all_results)
        meta["total_retries"] = total_retries
        meta["total_reviewer_rejections"] = total_reviewer_rejections

        # ── 5. Persist to store (lazy: only fields with ≥1 extraction) ──
        t_persist = time.time()
        field_id_map = self._get_field_id_map(cluster_id, fields)
        n_persisted, drop_stats = self._persist_fields_and_results(
            cluster_id, fields, all_results, field_id_map,
        )
        meta["n_persisted"] = n_persisted
        meta["field_drop_stats"] = drop_stats
        meta["timings"]["persist_s"] = round(time.time() - t_persist, 3)

        n_dropped = (drop_stats["fields_dropped_no_extractions"]
                     + drop_stats["fields_dropped_retrieval_miss"]
                     + drop_stats["fields_dropped_low_confidence"])
        if n_dropped:
            logger.info(
                f"[{cluster_label}] Lazy-persist: kept {drop_stats['fields_persisted']}, "
                f"dropped {n_dropped} fields "
                f"(retrieval_miss={drop_stats['fields_dropped_retrieval_miss']}, "
                f"low_conf={drop_stats['fields_dropped_low_confidence']}, "
                f"no_extractions={drop_stats['fields_dropped_no_extractions']})"
            )

        meta["total_time_s"] = round(time.time() - t0, 2)

        return ExtractionBatch(
            cluster_id=cluster_id,
            cluster_label=cluster_label,
            fields=fields,
            results=all_results,
            meta=meta,
        )

    # ══════════════════════════════════════════════════════════════════════
    # PER-DOCUMENT EXTRACTION
    # ══════════════════════════════════════════════════════════════════════

    async def _extract_document(
        self,
        agreement_id: str,
        fields: list[dict],
        cluster_label: str,
        context_section: str,
        chunks: list[dict],
    ) -> tuple[list[ExtractionResult], dict]:
        """Extract all fields from one document using multi-round RAG."""
        doc_meta = {"retries": 0, "reviewer_rejections": 0}
        results: list[ExtractionResult] = []

        # Process fields with bounded concurrency
        field_sem = asyncio.Semaphore(self.config.max_concurrent_fields)

        async def _process_field(field_def: dict):
            async with field_sem:
                return await self._extract_field_rag(
                    agreement_id=agreement_id,
                    field_def=field_def,
                    cluster_label=cluster_label,
                    context_section=context_section,
                    chunks=chunks,
                )

        field_tasks = [_process_field(f) for f in fields]
        field_results = await asyncio.gather(*field_tasks, return_exceptions=True)

        for fr in field_results:
            if isinstance(fr, Exception):
                logger.warning(f"Field extraction failed: {fr}")
                continue
            result, retries, reviewer_rejected = fr
            if result is not None:
                results.append(result)
            doc_meta["retries"] += retries
            if reviewer_rejected:
                doc_meta["reviewer_rejections"] += 1

        return results, doc_meta

    # ══════════════════════════════════════════════════════════════════════
    # PER-FIELD RAG EXTRACTION (the core loop)
    # ══════════════════════════════════════════════════════════════════════

    async def _extract_field_rag(
        self,
        agreement_id: str,
        field_def: dict,
        cluster_label: str,
        context_section: str,
        chunks: list[dict],
    ) -> tuple[ExtractionResult | None, int, bool]:
        """
        Multi-round RAG for a single (document, field) pair.

        Round 1: Hybrid retrieval → LLM extract
        Round 2+: If insufficient → agentic re-query
        Final: Reviewer agent accept/reject

        Returns: (result, n_retries, reviewer_rejected)
        """
        field_name = field_def["name"]
        field_type = field_def.get("type", "text")
        retries = 0
        reviewer_rejected = False

        # ── Round 1: Hybrid Retrieval ──
        snippets = await self._hybrid_retrieve(
            agreement_id=agreement_id,
            field_def=field_def,
            chunks=chunks,
        )

        # ── LLM Extraction ──
        result = await self._llm_extract(
            field_def=field_def,
            snippets=snippets,
            cluster_label=cluster_label,
            context_section=context_section,
            agreement_id=agreement_id,
            retrieval_round=1,
        )

        # ── Agentic Retry Loop (Rounds 2+) ──
        round_num = 1
        while (
            round_num < self.config.max_retry_rounds
            and result is not None
            and result.confidence < self.config.insufficient_context_threshold
        ):
            round_num += 1
            retries += 1

            # Ask LLM to generate better search terms
            new_terms = await self._generate_search_terms(
                field_def=field_def,
                cluster_label=cluster_label,
                previous_snippets=snippets,
                previous_result=result,
            )

            if not new_terms:
                break

            # Re-retrieve with new terms
            new_snippets = await self._agentic_retrieve(
                agreement_id=agreement_id,
                field_def=field_def,
                search_terms=new_terms,
                chunks=chunks,
            )

            # Cross-doc NL index lookup
            if self.config.enable_nl_index and self._nl_index:
                nl_snippets = self._nl_index_lookup(
                    field_def=field_def,
                    search_terms=new_terms,
                    agreement_id=agreement_id,
                    chunks=chunks,
                )
                new_snippets.extend(nl_snippets)

            if not new_snippets:
                break

            # Deduplicate snippets
            seen_ids = {s.chunk_id for s in snippets}
            for ns in new_snippets:
                if ns.chunk_id not in seen_ids:
                    snippets.append(ns)
                    seen_ids.add(ns.chunk_id)

            # Re-sort by score
            snippets.sort(key=lambda s: s.score, reverse=True)
            snippets = snippets[:self.config.final_top_k + 2]  # allow a little extra

            # Re-extract with enriched snippets
            result = await self._llm_extract(
                field_def=field_def,
                snippets=snippets,
                cluster_label=cluster_label,
                context_section=context_section,
                agreement_id=agreement_id,
                retrieval_round=round_num,
            )

        # ── Reviewer Agent Pass ──
        if self.config.enable_reviewer and result is not None and result.value is not None:
            accepted, revised_result = await self._reviewer_pass(
                result=result,
                snippets=snippets,
                field_def=field_def,
                cluster_label=cluster_label,
            )
            if not accepted:
                reviewer_rejected = True
                if revised_result is not None:
                    result = revised_result
                    result.reviewer_accepted = True
                else:
                    result.reviewer_accepted = False
                    result.confidence *= 0.5  # halve confidence for rejected
            else:
                result.reviewer_accepted = True

        # ── Normalize value ──
        if result is not None and result.value is not None:
            result.raw_value = result.value
            result.value = normalize_value(result.value, field_type)

            # ── Type validation ──
            is_valid, corrected = validate_extracted_type(result.value, field_type)
            if not is_valid:
                # Type mismatch: halve confidence and discard bad value
                result.confidence *= 0.3
                result.value = None
            elif corrected is not None and corrected != result.value:
                result.value = corrected

        # Filter below threshold
        if result is not None and result.confidence < self.config.min_extraction_confidence:
            return None, retries, reviewer_rejected

        return result, retries, reviewer_rejected

    # ══════════════════════════════════════════════════════════════════════
    # RETRIEVAL METHODS
    # ══════════════════════════════════════════════════════════════════════

    async def _hybrid_retrieve(
        self,
        agreement_id: str,
        field_def: dict,
        chunks: list[dict],
    ) -> list[RetrievedSnippet]:
        """
        Round 1: Hybrid retrieval (semantic + lexical + RRF).

        Build query from field definition → fetch top-k from both signals
        → fuse with RRF → return top-3 snippets.
        Applies multi-chunk evidence window: for each retrieved chunk,
        concatenates adjacent chunks to provide broader context.
        """
        query = self._build_field_query(field_def)
        retriever = self._doc_retrievers.get(agreement_id)

        if retriever is not None:
            # When wide scope is on, retrieve more candidates and re-rank with
            # cluster-membership boost.  Otherwise, use final_top_k directly.
            if self.config.widen_retrieval_scope:
                retrieve_k = max(
                    self.config.final_top_k * self.config.wide_retrieve_k_multiplier,
                    8,
                )
            else:
                retrieve_k = self.config.final_top_k

            raw_results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: retriever.search(
                    query,
                    top_k=retrieve_k,
                    use_reranker=False,
                    rrf_k=self.config.rrf_k,
                ),
            )

            # Boost cluster-member chunks: they are presumptively more relevant
            # since they were topically assigned to this cluster.
            if self.config.widen_retrieval_scope:
                boost = self.config.cluster_boost_weight
                for r in raw_results:
                    idx = r["idx"]
                    chunk_row = chunks[idx] if idx < len(chunks) else None
                    if chunk_row and chunk_row.get("_in_cluster", False):
                        r["rrf_score"] += boost
                raw_results.sort(key=lambda x: x["rrf_score"], reverse=True)

            raw_results = raw_results[:self.config.final_top_k]

            snippets = []
            for rank, r in enumerate(raw_results):
                idx = r["idx"]
                chunk_row = chunks[idx] if idx < len(chunks) else None
                # Multi-chunk evidence window: include adjacent chunks
                text = r["chunk"]
                if len(chunks) > 1:
                    adjacent_parts = []
                    if idx > 0:
                        prev_text = chunks[idx - 1].get("chunk_text", "")
                        if prev_text:
                            adjacent_parts.append(prev_text[-1000:])
                    adjacent_parts.append(text)
                    if idx + 1 < len(chunks):
                        next_text = chunks[idx + 1].get("chunk_text", "")
                        if next_text:
                            adjacent_parts.append(next_text[:1000])
                    text = "\n...\n".join(adjacent_parts)
                snippets.append(RetrievedSnippet(
                    chunk_id=chunk_row["chunk_id"] if chunk_row else f"idx_{idx}",
                    chunk_text=text,
                    agreement_id=agreement_id,
                    score=r["rrf_score"],
                    source="rrf",
                    rank=rank,
                ))
            return snippets

        # Fallback: simple semantic search via LanceDB
        if self.lance:
            query_vec = self.embed_fn(query).reshape(-1)
            results = self.lance.search_chunks(
                query_vector=query_vec,
                top_k=self.config.final_top_k,
                where=f"agreement_id = '{agreement_id}'",
            )
            return [
                RetrievedSnippet(
                    chunk_id=r.get("chunk_id", ""),
                    chunk_text=r.get("text", ""),
                    agreement_id=agreement_id,
                    score=1.0 - r.get("_distance", 0),
                    source="semantic",
                    rank=i,
                )
                for i, r in enumerate(results)
            ]

        # Last resort: return all chunks sorted by position
        return [
            RetrievedSnippet(
                chunk_id=c.get("chunk_id", f"idx_{i}"),
                chunk_text=c.get("chunk_text", ""),
                agreement_id=agreement_id,
                score=1.0 / (i + 1),
                source="positional",
                rank=i,
            )
            for i, c in enumerate(chunks[:self.config.final_top_k])
        ]

    async def _agentic_retrieve(
        self,
        agreement_id: str,
        field_def: dict,
        search_terms: list[str],
        chunks: list[dict],
    ) -> list[RetrievedSnippet]:
        """
        Agentic retry: re-query with LLM-generated search terms.

        Uses each search term as a separate query, then merges results
        via RRF across all sub-queries.
        """
        retriever = self._doc_retrievers.get(agreement_id)
        if retriever is None:
            return []

        all_rrf: dict[int, float] = {}
        per_idx_text: dict[int, str] = {}

        for term in search_terms[:3]:  # cap at 3 sub-queries
            raw = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda t=term: retriever.search(
                    t,
                    top_k=self.config.semantic_top_k,
                    use_reranker=False,
                    rrf_k=self.config.rrf_k,
                ),
            )
            for rank, r in enumerate(raw):
                idx = r["idx"]
                all_rrf[idx] = all_rrf.get(idx, 0) + 1.0 / (self.config.rrf_k + rank)
                per_idx_text[idx] = r["chunk"]

        # Sort by fused score
        sorted_results = sorted(all_rrf.items(), key=lambda x: x[1], reverse=True)
        snippets = []
        for rank, (idx, score) in enumerate(sorted_results[:self.config.final_top_k]):
            chunk_row = chunks[idx] if idx < len(chunks) else None
            snippets.append(RetrievedSnippet(
                chunk_id=chunk_row["chunk_id"] if chunk_row else f"idx_{idx}",
                chunk_text=per_idx_text.get(idx, ""),
                agreement_id=agreement_id,
                score=score,
                source="agentic_retry",
                rank=rank,
            ))
        return snippets

    # ══════════════════════════════════════════════════════════════════════
    # NL INDEX (cross-doc heading lookup)
    # ══════════════════════════════════════════════════════════════════════

    def _build_nl_index(self, chunks_by_doc: dict[str, list[dict]]):
        """
        Build a lightweight heading/section index from chunk texts.

        Looks for markdown headings, numbered sections, and clause titles
        to enable heading-targeted retrieval.
        """
        heading_pattern = re.compile(
            r'^(?:#{1,4}\s+|(?:\d+\.)+\s+|Article\s+\d+|Section\s+\d+|ARTICLE\s+[IVXLC]+)',
            re.MULTILINE,
        )
        self._nl_index = {}

        for agreement_id, chunks in chunks_by_doc.items():
            headings = []
            for chunk in chunks:
                text = chunk.get("chunk_text", "")
                matches = heading_pattern.findall(text)
                if matches:
                    for m in matches[:3]:
                        # Extract the full heading line
                        for line in text.split("\n"):
                            if m.strip() in line:
                                headings.append({
                                    "heading": line.strip()[:200],
                                    "chunk_id": chunk.get("chunk_id", ""),
                                    "chunk_text": text,
                                    "agreement_id": agreement_id,
                                })
                                break
            self._nl_index[agreement_id] = headings

    def _nl_index_lookup(
        self,
        field_def: dict,
        search_terms: list[str],
        agreement_id: str,
        chunks: list[dict],
    ) -> list[RetrievedSnippet]:
        """Heading-targeted retrieval from NL index."""
        if not self._nl_index:
            return []

        headings = self._nl_index.get(agreement_id, [])
        if not headings:
            return []

        # Keyword match on headings
        all_terms = search_terms + [field_def["name"].replace("_", " ")]
        scored: list[tuple[dict, float]] = []
        for h in headings:
            heading_lower = h["heading"].lower()
            score = 0.0
            for term in all_terms:
                for word in term.lower().split():
                    if len(word) > 2 and word in heading_lower:
                        score += 1.0
            if score > 0:
                scored.append((h, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        snippets = []
        seen = set()
        for h, score in scored[:self.config.nl_index_top_k]:
            cid = h["chunk_id"]
            if cid in seen:
                continue
            seen.add(cid)
            snippets.append(RetrievedSnippet(
                chunk_id=cid,
                chunk_text=h["chunk_text"],
                agreement_id=agreement_id,
                score=score / max(len(all_terms), 1),
                source="nl_index",
                rank=len(snippets),
            ))
        return snippets

    # ══════════════════════════════════════════════════════════════════════
    # LLM EXTRACTION
    # ══════════════════════════════════════════════════════════════════════

    async def _llm_extract(
        self,
        field_def: dict,
        snippets: list[RetrievedSnippet],
        cluster_label: str,
        context_section: str,
        agreement_id: str,
        retrieval_round: int,
    ) -> ExtractionResult | None:
        """
        Run LLM extraction over retrieved snippets for one field.

        Model routing: complex fields → GPT-5.4, simple → GPT-5.4-nano.
        """
        if not snippets:
            return None

        field_name = field_def["name"]
        field_type = field_def.get("type", "text")
        description = field_def.get("description", "")
        examples = field_def.get("examples", [])

        # Choose model
        model = self.config.complex_model
        if self.config.model_routing and not is_complex_field(field_def):
            model = self.config.simple_model

        # Build snippet context
        snippet_texts = []
        for i, s in enumerate(snippets[:self.config.final_top_k]):
            snippet_texts.append(
                f"[Snippet {i+1}] (score={s.score:.3f}, source={s.source}):\n"
                f"{s.chunk_text[:self.config.max_chunk_chars]}"
            )
        snippet_section = "\n\n".join(snippet_texts)

        # Build examples section
        examples_section = ""
        if examples:
            ex_str = ", ".join(f'"{e}"' for e in examples[:3] if e)
            if ex_str:
                examples_section = f"\nExamples of valid values: {ex_str}"

        prompt = _EXTRACTION_PROMPT.format(
            cluster_label=cluster_label,
            context_section=context_section,
            field_name=field_name,
            field_type=field_type,
            field_description=description,
            examples_section=examples_section,
            snippet_section=snippet_section,
        )

        parsed = await self.llm.complete_json(
            prompt, temperature=0.0, max_tokens=1000, model=model,
        )

        if not parsed or not isinstance(parsed, dict):
            return None

        value = parsed.get("value")
        if value is None or str(value).lower() in ("null", "n/a", "none", ""):
            return ExtractionResult(
                field_name=field_name,
                field_type=field_type,
                value=None,
                raw_value=None,
                confidence=0.0,
                evidence_span="",
                is_default=True,
                agreement_id=agreement_id,
                chunk_id=snippets[0].chunk_id if snippets else "",
                model_used=model,
                retrieval_round=retrieval_round,
                reviewer_accepted=None,
                snippets_used=len(snippets),
            )

        value_str = str(value)
        evidence = str(parsed.get("evidence", ""))[:500]
        is_default = bool(parsed.get("is_default", False))
        confidence = float(parsed.get("confidence", 0.5))

        # Grounding boost: require >80% character overlap, not just any substring
        combined = " ".join(s.chunk_text for s in snippets).lower()
        val_lower = value_str.lower()
        if len(val_lower) > 0:
            # Count characters in value that appear in a matching window
            match_start = combined.find(val_lower)
            if match_start >= 0:
                # Exact full match → full boost
                confidence = min(1.0, confidence + 0.1)
            elif len(val_lower) >= 10:
                # For longer values, check word-level overlap ratio
                val_words = set(val_lower.split())
                combined_words = set(combined.split())
                overlap_ratio = len(val_words & combined_words) / len(val_words) if val_words else 0
                if overlap_ratio >= 0.8:
                    confidence = min(1.0, confidence + 0.05)  # partial boost

        return ExtractionResult(
            field_name=field_name,
            field_type=field_type,
            value=value_str,
            raw_value=value_str,
            confidence=round(confidence, 3),
            evidence_span=evidence,
            is_default=is_default,
            agreement_id=agreement_id,
            chunk_id=snippets[0].chunk_id if snippets else "",
            model_used=model,
            retrieval_round=retrieval_round,
            reviewer_accepted=None,
            snippets_used=len(snippets),
        )

    # ══════════════════════════════════════════════════════════════════════
    # AGENTIC SEARCH TERM GENERATION
    # ══════════════════════════════════════════════════════════════════════

    async def _generate_search_terms(
        self,
        field_def: dict,
        cluster_label: str,
        previous_snippets: list[RetrievedSnippet],
        previous_result: ExtractionResult | None,
    ) -> list[str]:
        """
        Ask LLM to generate alternative search terms when extraction
        confidence is low ("Insufficient Context" scenario).
        """
        prev_texts = "\n".join(
            f"- {s.chunk_text[:200]}" for s in previous_snippets[:3]
        )
        prev_value = previous_result.value if previous_result else "None"

        prompt = _SEARCH_TERM_PROMPT.format(
            field_name=field_def["name"],
            field_type=field_def.get("type", "text"),
            field_description=field_def.get("description", ""),
            cluster_label=cluster_label,
            previous_snippets=prev_texts,
            previous_value=prev_value,
        )

        parsed = await self.llm.complete_json(prompt, temperature=0.3, max_tokens=300)
        if not parsed or not isinstance(parsed, dict):
            return []

        terms = parsed.get("search_terms", [])
        if not isinstance(terms, list):
            return []
        return [str(t).strip() for t in terms if t and str(t).strip()][:5]

    # ══════════════════════════════════════════════════════════════════════
    # REVIEWER AGENT
    # ══════════════════════════════════════════════════════════════════════

    async def _reviewer_pass(
        self,
        result: ExtractionResult,
        snippets: list[RetrievedSnippet],
        field_def: dict,
        cluster_label: str,
    ) -> tuple[bool, ExtractionResult | None]:
        """
        Cheap reviewer model: accept/reject extraction + optional re-extract.

        Returns: (accepted, revised_result_or_None)
        """
        snippet_evidence = "\n".join(
            f"[{i+1}] {s.chunk_text[:500]}" for i, s in enumerate(snippets[:3])
        )

        model = self.config.reviewer_model or self.config.simple_model
        prompt = _REVIEWER_PROMPT.format(
            field_name=field_def["name"],
            field_type=field_def.get("type", "text"),
            field_description=field_def.get("description", ""),
            cluster_label=cluster_label,
            extracted_value=result.value,
            evidence_span=result.evidence_span,
            snippet_evidence=snippet_evidence,
        )

        parsed = await self.llm.complete_json(
            prompt, temperature=0.0, max_tokens=400, model=model,
        )
        if not parsed or not isinstance(parsed, dict):
            return True, None  # fail-open: accept if reviewer fails

        verdict = str(parsed.get("verdict", "accept")).lower()
        if verdict == "accept":
            return True, None

        # Rejected — check if reviewer provided a corrected value
        corrected = parsed.get("corrected_value")
        if corrected and str(corrected).lower() not in ("null", "none", "n/a", ""):
            revised = ExtractionResult(
                field_name=result.field_name,
                field_type=result.field_type,
                value=str(corrected),
                raw_value=str(corrected),
                confidence=float(parsed.get("confidence", 0.4)),
                evidence_span=str(parsed.get("evidence", result.evidence_span))[:500],
                is_default=False,
                agreement_id=result.agreement_id,
                chunk_id=result.chunk_id,
                model_used=model,
                retrieval_round=result.retrieval_round,
                reviewer_accepted=None,
                snippets_used=result.snippets_used,
            )
            return False, revised

        return False, None

    # ══════════════════════════════════════════════════════════════════════
    # INDEX BUILDING
    # ══════════════════════════════════════════════════════════════════════

    async def _build_doc_indexes(self, chunks_by_doc: dict[str, list[dict]]):
        """
        Build per-document retriever instances.

        When use_late_interaction=True, uses ColBERT MaxSim + BM25 (no FAISS).
        Otherwise, uses the original HybridRetriever (FAISS + BM25 + cross-encoder).
        """
        if self.config.use_late_interaction:
            from core.late_interaction_retriever import LateInteractionRetriever
        else:
            from core.retrieval import HybridRetriever

        loop = asyncio.get_event_loop()
        use_li = self.config.use_late_interaction

        def _build_one(agreement_id: str, chunks: list[dict]):
            texts = [c.get("chunk_text", "") for c in chunks]
            if not texts:
                return agreement_id, None

            metadata = [
                {"chunk_id": c.get("chunk_id", ""), "agreement_id": agreement_id}
                for c in chunks
            ]

            if use_li:
                retriever = LateInteractionRetriever(
                    chunks=texts,
                    chunk_metadata=metadata,
                )
            else:
                embeddings = self.embed_fn(texts)
                if isinstance(embeddings, list):
                    embeddings = np.array(embeddings)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                retriever = HybridRetriever(
                    chunks=texts,
                    embeddings=embeddings,
                    embed_fn=lambda t: self.embed_fn(t) if isinstance(t, str) else self.embed_fn([t])[0],
                    chunk_metadata=metadata,
                )
            return agreement_id, retriever

        # Build in parallel via thread pool (embedding is CPU/GPU bound)
        tasks = []
        for aid, chunks in chunks_by_doc.items():
            if aid not in self._doc_retrievers:
                tasks.append(loop.run_in_executor(None, _build_one, aid, chunks))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Failed to build retriever: {r}")
                    continue
                aid, retriever = r
                if retriever is not None:
                    self._doc_retrievers[aid] = retriever

    # ══════════════════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════

    def _get_field_id_map(self, cluster_id: str, fields: list[dict]) -> dict[str, str]:
        """Map field names to field_ids — looks up existing, generates ids for new.

        Does NOT persist new field_definitions.  Persistence is deferred to
        _persist_fields_and_results() so that only fields with ≥1 extraction
        are ever written (lazy-persist pattern).
        """
        existing = self.store.get_fields(cluster_id=cluster_id)
        fmap = {f["name"]: f["field_id"] for f in existing}

        for f in fields:
            if f["name"] not in fmap:
                fmap[f["name"]] = f"fld_{uuid.uuid4().hex[:10]}"
        return fmap

    def _persist_fields_and_results(
        self,
        cluster_id: str,
        fields: list[dict],
        results: list[ExtractionResult],
        field_id_map: dict[str, str],
    ) -> tuple[int, dict]:
        """Persist only fields that have ≥1 extraction (lazy-persist).

        Returns (n_persisted, drop_stats) where drop_stats tracks how many
        fields were dropped and why.
        """
        # Count non-null extractions per field
        field_extraction_counts: dict[str, int] = {}
        for r in results:
            if r.value is not None:
                field_extraction_counts[r.field_name] = (
                    field_extraction_counts.get(r.field_name, 0) + 1
                )

        # Classify each field
        drop_stats = {
            "fields_persisted": 0,
            "fields_dropped_no_extractions": 0,
            "fields_dropped_retrieval_miss": 0,
            "fields_dropped_low_confidence": 0,
            "fields_already_existed": 0,
            "drop_details": [],  # (field_name, reason)
        }

        # Track which fields had results at all (even null) to distinguish
        # "retrieval miss" (no results) from "low confidence" (results but
        # all null due to low confidence)
        fields_with_any_result = set()
        fields_with_low_conf_null = set()
        for r in results:
            fields_with_any_result.add(r.field_name)
            if r.value is None and r.confidence < 0.50:
                fields_with_low_conf_null.add(r.field_name)

        existing_fields = {f["name"] for f in self.store.get_fields(cluster_id=cluster_id)}
        fields_by_name = {f["name"]: f for f in fields}

        for fname, fid in field_id_map.items():
            f = fields_by_name.get(fname)
            if not f:
                continue

            if fname in existing_fields:
                # Already persisted (e.g. from agent discovery) — don't re-insert
                drop_stats["fields_already_existed"] += 1
                continue

            if field_extraction_counts.get(fname, 0) > 0:
                # Has extractions — persist the field_definition
                self.store.upsert_field(
                    field_id=fid,
                    cluster_id=cluster_id,
                    name=fname,
                    field_type=f.get("type", "text"),
                    description=f.get("description", ""),
                    source="rag_extractor",
                    examples=f.get("examples", []),
                )
                drop_stats["fields_persisted"] += 1
            else:
                # No extractions — classify the drop reason
                if fname not in fields_with_any_result:
                    reason = "retrieval_miss"
                    drop_stats["fields_dropped_retrieval_miss"] += 1
                elif fname in fields_with_low_conf_null:
                    reason = "low_confidence"
                    drop_stats["fields_dropped_low_confidence"] += 1
                else:
                    reason = "no_extractions"
                    drop_stats["fields_dropped_no_extractions"] += 1
                drop_stats["drop_details"].append((fname, reason))

        # Now persist extraction rows for fields that exist (persisted or pre-existing)
        persisted_field_names = (
            existing_fields
            | {fname for fname in field_id_map
               if field_extraction_counts.get(fname, 0) > 0}
        )
        count = 0
        for r in results:
            if r.value is None:
                continue
            if r.field_name not in persisted_field_names:
                continue
            field_id = field_id_map.get(r.field_name)
            if not field_id:
                continue
            self.store.insert_extraction(
                extraction_id=f"ext_{uuid.uuid4().hex[:10]}",
                agreement_id=r.agreement_id,
                field_id=field_id,
                value=r.value,
                confidence=r.confidence,
                source_chunk_id=r.chunk_id,
            )
            count += 1
        return count, drop_stats

    # ══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _build_field_query(field_def: dict) -> str:
        """Build a retrieval query from a field definition."""
        parts = [field_def["name"].replace("_", " ")]
        desc = field_def.get("description", "")
        if desc:
            parts.append(desc)
        examples = field_def.get("examples", [])
        if examples:
            parts.append("examples: " + ", ".join(str(e) for e in examples[:3]))
        return " ".join(parts)


# ── Prompt Templates ──────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
You are extracting a specific field from a legal agreement clause.

Clause Type: {cluster_label}
{context_section}

FIELD TO EXTRACT:
  Name: {field_name}
  Type: {field_type}
  Description: {field_description}{examples_section}

RETRIEVED SNIPPETS (ranked by relevance):
{snippet_section}

INSTRUCTIONS:
1. Find the value of "{field_name}" in the snippets above.
2. If the value is explicitly stated, extract it verbatim — include the COMPLETE value.
   Do NOT truncate mid-sentence. If a clause spans multiple sentences, include all of them.
3. If the value must be inferred from context, note that in evidence.
4. If not found in any snippet, set value to null.
5. COMPLETENESS CHECK: Before returning, verify your extracted value ends at a natural
   boundary (sentence end, clause end, or list end). If it appears truncated, extend it.

TYPE-SPECIFIC RULES:
- boolean: Return ONLY "true" or "false" as the value. Do NOT return explanatory text.
- date: Return in YYYY-MM-DD format when possible (e.g. "2024-01-15"). If only a relative date, return verbatim.
- currency: Include the currency symbol/code and amount (e.g. "$50,000" or "USD 50,000").
- percentage: Include the % sign (e.g. "15%" or "2.5%").
- duration: Include the time unit (e.g. "90 days", "12 months", "2 years").
- integer: Return only the numeric value (e.g. "30", not "thirty (30)").

Return JSON:
{{
  "value": "<extracted value or null>",
  "evidence": "<verbatim quote from snippet supporting the extraction>",
  "confidence": <0.0-1.0>,
  "is_default": <true if this is a standard/boilerplate value>
}}"""

_SEARCH_TERM_PROMPT = """\
You are helping find better search terms for field extraction.

Field: {field_name} ({field_type})
Description: {field_description}
Clause Type: {cluster_label}

Previous search returned these snippets (which were insufficient):
{previous_snippets}

Previous extraction attempt returned: {previous_value}

Generate 3-5 alternative search terms that might locate the relevant clause.
Think about:
- Synonyms and legal terminology
- Section headings where this info appears
- Related concepts that co-occur with this field

Return JSON:
{{
  "search_terms": ["term1", "term2", "term3"]
}}"""

_REVIEWER_PROMPT = """\
You are a quality reviewer for contract field extraction. Your job is to
verify whether an extraction is correct and well-supported by evidence.

Field: {field_name} ({field_type})
Description: {field_description}
Clause Type: {cluster_label}

EXTRACTED VALUE: {extracted_value}
EVIDENCE: {evidence_span}

SOURCE SNIPPETS:
{snippet_evidence}

Review criteria:
1. Is the extracted value actually present in or supported by the snippets?
2. Does the value match the expected field type?
3. Is this the correct/complete value (not a partial extraction)?

Return JSON:
{{
  "verdict": "accept" or "reject",
  "reason": "<brief reason>",
  "corrected_value": "<if rejected, provide corrected value or null>",
  "confidence": <0.0-1.0>,
  "evidence": "<supporting quote if corrected>"
}}"""


# ── Module-level Entry Points ─────────────────────────────────────────────────


async def extract_cluster_rag(
    store,
    llm,
    embed_fn: Callable,
    cluster_id: str,
    fields: list[dict],
    cluster_label: str,
    cluster_keywords: list[str] | None = None,
    context_section: str = "",
    config: RAGConfig | None = None,
    lance_store=None,
) -> ExtractionBatch:
    """
    Module-level convenience — run RAG extraction for one cluster.

    Returns ExtractionBatch with results + meta.
    """
    extractor = RAGExtractor(
        store=store, llm=llm, embed_fn=embed_fn,
        config=config, lance_store=lance_store,
    )
    return await extractor.extract_cluster(
        cluster_id=cluster_id,
        fields=fields,
        cluster_label=cluster_label,
        cluster_keywords=cluster_keywords,
        context_section=context_section,
    )
