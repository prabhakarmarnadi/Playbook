"""
Field Feature Extraction — ported from apr-fields-processor.

Loads 221 predefined field configs (with regex patterns, key terms, examples,
negative examples) and 500+ business fields from the legacy apr-fields-processor
data directory.  Provides multi-strategy extraction per chunk:

  1. Pattern-based (regex)  — highest signal
  2. Keyword proximity       — medium signal
  3. Type inference           — date / amount / number detection

Combined with embedding similarity scores from FieldRegistry, this gives
each chunk a rich feature vector of matched fields before LLM discovery.

Usage in Stage 2 of minimal_pipeline:
    ff = FieldFeatureExtractor.from_legacy_dir(legacy_dir)
    for chunk in chunks:
        features = ff.extract_features(chunk["text"])
        chunk["field_features"] = features
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Default legacy data path ──────────────────────────────────────────────────

_DEFAULT_LEGACY_DIR = Path(__file__).resolve().parent.parent.parent / "apr-fields-processor"
_PREDEFINED_DIR = "src/data/field_configs/predefined"
_BUSINESS_FILE = "src/data/field_configs/business/gliner_field_types.json"


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class RichFieldConfig:
    """Full field config mirroring apr-fields-processor FieldConfig."""
    name: str
    extraction_type: str          # text, date, number, amount, entity, boolean, duration
    taxonomy_category: str
    description: str
    key_terms: list[str] = field(default_factory=list)
    regex_patterns: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    context_window: int = 200
    source: str = "predefined"    # predefined | business
    extraction_strategy: str | None = None
    clause_type_description: str | None = None


@dataclass
class FieldMatch:
    """A single field feature match found in a chunk."""
    field_name: str
    extraction_type: str
    taxonomy_category: str
    method: str                    # pattern | keyword | type_inference
    value: str
    confidence: float
    position: int = 0
    context: str = ""


@dataclass
class ChunkFieldFeatures:
    """Aggregated field features for one chunk."""
    matches: list[FieldMatch] = field(default_factory=list)
    matched_field_names: set[str] = field(default_factory=set)
    matched_categories: set[str] = field(default_factory=set)
    top_confidence: float = 0.0

    def summary_line(self, max_fields: int = 5) -> str:
        """One-line summary for injection into embeddings / prompts."""
        if not self.matches:
            return ""
        top = sorted(self.matches, key=lambda m: m.confidence, reverse=True)[:max_fields]
        parts = [f"{m.field_name}({m.method[0]}:{m.confidence:.2f})" for m in top]
        return "Fields: " + ", ".join(parts)


# ── Loaders ───────────────────────────────────────────────────────────────────


def _parse_key_terms(raw: list[str]) -> list[str]:
    """Strip description lines (start with '-') from alternating term/desc format."""
    return [t.strip() for t in raw if t.strip() and not t.strip().startswith("-")]


def load_predefined_fields(config_dir: Path) -> list[RichFieldConfig]:
    """Load 221 predefined field configs from custom_field_config_*.json files."""
    configs: list[RichFieldConfig] = []
    if not config_dir.exists():
        logger.warning(f"Predefined fields dir not found: {config_dir}")
        return configs

    for json_file in sorted(config_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            cfg = data.get("config", {})
            fields_arr = data.get("fields", [])
            hints = data.get("extraction_hints", {})

            if not fields_arr:
                continue

            fd = fields_arr[0]
            name = fd.get("name", cfg.get("extraction_name", json_file.stem))
            configs.append(RichFieldConfig(
                name=name,
                extraction_type=cfg.get("extraction_type", "text"),
                taxonomy_category=cfg.get("taxonomy_category", "Unknown"),
                description=fd.get("description", "")[:500],
                key_terms=_parse_key_terms(hints.get("key_terms", [])),
                regex_patterns=hints.get("regex_patterns", []),
                examples=[ex[:200] for ex in fd.get("examples", [])[:5]],
                negative_examples=[ne[:200] for ne in fd.get("negative_examples", [])[:3]],
                context_window=hints.get("context_window", 200),
                source="predefined",
                extraction_strategy=hints.get("extraction_strategy"),
                clause_type_description=hints.get("clause_type_description"),
            ))
        except Exception as e:
            logger.debug(f"Skipping {json_file.name}: {e}")

    logger.info(f"Loaded {len(configs)} predefined field configs")
    return configs


def load_business_fields(json_path: Path) -> list[RichFieldConfig]:
    """Load 500+ business fields from gliner_field_types.json."""
    configs: list[RichFieldConfig] = []
    if not json_path.exists():
        logger.warning(f"Business fields file not found: {json_path}")
        return configs

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        _CAT_DESC = {
            "common_business": "Common business entity",
            "contract_legal": "Legal contract clause or term",
            "financial": "Financial or monetary information",
            "employment": "Employment-related information",
            "real_estate": "Real estate or property information",
            "healthcare": "Healthcare or medical information",
            "insurance": "Insurance policy information",
            "intellectual_property": "Intellectual property information",
            "software_it": "Technology or IT-related information",
            "shipping_logistics": "Supply chain or logistics information",
            "nda_confidentiality": "NDA or confidentiality information",
            "master_services_agreement": "Master services agreement terms",
            "software_license": "Software licensing terms",
            "academic": "Academic or educational information",
        }

        for category, field_names in data.items():
            if not isinstance(field_names, list):
                continue
            cat_desc = _CAT_DESC.get(category, "Business entity")
            for fname in field_names:
                readable = fname.replace("_", " ").title()
                # Generate key terms from the field name words
                terms = [w for w in fname.replace("_", " ").split() if len(w) > 2]
                configs.append(RichFieldConfig(
                    name=fname,
                    extraction_type="text",
                    taxonomy_category=category,
                    description=f"{cat_desc}: {readable}",
                    key_terms=terms + [fname.replace("_", " ")],
                    source="business",
                ))
    except Exception as e:
        logger.warning(f"Failed to load business fields: {e}")

    logger.info(f"Loaded {len(configs)} business field configs")
    return configs


# ── Extractors ────────────────────────────────────────────────────────────────


class PatternMatcher:
    """Regex-based value extraction from text (ported from PatternExtractor)."""

    @staticmethod
    def extract(text: str, patterns: list[str], context_window: int = 200,
                max_matches: int = 5) -> list[dict]:
        if not text or not patterns:
            return []
        matches = []
        for pat_str in patterns:
            try:
                pat = re.compile(pat_str, re.IGNORECASE | re.MULTILINE)
                for m in pat.finditer(text):
                    value = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
                    if value and value.strip():
                        start = m.start()
                        ctx_start = max(0, start - context_window)
                        ctx_end = min(len(text), m.end() + context_window)
                        matches.append({
                            "value": value.strip(),
                            "pattern": pat_str,
                            "start": start,
                            "context": text[ctx_start:ctx_end],
                            "confidence": 0.80,
                        })
                    if len(matches) >= max_matches:
                        break
            except re.error:
                continue
            if len(matches) >= max_matches:
                break
        return matches


class KeywordMatcher:
    """Keyword proximity-based extraction (ported from KeywordExtractor)."""

    @staticmethod
    def find_near_keywords(text: str, keywords: list[str],
                           window_after: int = 150) -> list[dict]:
        if not text or not keywords:
            return []
        matches = []
        text_lower = text.lower()
        for kw in keywords:
            if not kw or len(kw) < 3:
                continue
            kw_lower = kw.lower()
            start = 0
            while True:
                pos = text_lower.find(kw_lower, start)
                if pos == -1:
                    break
                # Word boundary check
                before_ok = pos == 0 or not text[pos - 1].isalnum()
                after_ok = (pos + len(kw) >= len(text) or
                            not text[pos + len(kw)].isalnum())
                if before_ok and after_ok:
                    end = min(len(text), pos + len(kw) + window_after)
                    # Extract value after keyword (look for separator then value)
                    after_text = text[pos + len(kw):end]
                    value = KeywordMatcher._extract_value(after_text)
                    if value:
                        distance = 0  # right at keyword
                        conf = max(0.5, 1.0 - (abs(distance) / window_after))
                        matches.append({
                            "value": value,
                            "keyword": kw,
                            "start": pos,
                            "keyword_distance": distance,
                            "confidence": conf,
                        })
                start = pos + 1
                if len(matches) >= 5:
                    break
            if len(matches) >= 5:
                break
        return matches

    @staticmethod
    def _extract_value(after_text: str) -> str:
        """Extract value after a keyword separator (:, is, =, etc.)."""
        # Look for a separator
        sep_match = re.match(r'\s*[:=\-–]\s*', after_text)
        if sep_match:
            rest = after_text[sep_match.end():]
        else:
            rest = after_text.lstrip()

        if not rest:
            return ""

        # Take up to end of line or sentence
        line = rest.split("\n")[0].strip()
        # Trim at sentence boundaries
        for end_pat in [". ", "; ", ", and ", ", or "]:
            idx = line.find(end_pat)
            if idx > 0:
                line = line[:idx]
                break
        return line[:200].strip()


class TypeDetector:
    """Date / amount / number detection (ported from TypeInferencer)."""

    DATE_PATTERNS = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{4}\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
        r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b",
    ]

    AMOUNT_PATTERNS = [
        r"\$\s*\d+(?:,\d{3})*(?:\.\d{2})?",
        r"USD\s*\d+(?:,\d{3})*(?:\.\d{2})?",
        r"\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD)",
        r"€\s*\d+(?:,\d{3})*(?:\.\d{2})?",
        r"£\s*\d+(?:,\d{3})*(?:\.\d{2})?",
    ]

    DURATION_PATTERNS = [
        r"\b\d+\s*(?:year|month|week|day|hour)s?\b",
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|thirty|sixty|ninety)\s*\(?(?:\d+)?\)?\s*(?:year|month|week|day)s?\b",
    ]

    PERCENTAGE_PATTERNS = [
        r"\b\d+(?:\.\d+)?\s*%",
        r"\b\d+(?:\.\d+)?\s*percent\b",
    ]

    @classmethod
    def detect_types(cls, text: str) -> list[dict]:
        """Find all typed values (date, amount, duration, percentage) in text."""
        results = []
        for type_name, patterns in [
            ("date", cls.DATE_PATTERNS),
            ("amount", cls.AMOUNT_PATTERNS),
            ("duration", cls.DURATION_PATTERNS),
            ("percentage", cls.PERCENTAGE_PATTERNS),
        ]:
            for pat in patterns:
                for m in re.finditer(pat, text, re.IGNORECASE):
                    results.append({
                        "value": m.group(0).strip(),
                        "type": type_name,
                        "start": m.start(),
                        "confidence": 0.85,
                    })
        return results

    @classmethod
    def validate_type(cls, value: str, expected_type: str) -> bool:
        """Check if a value matches the expected extraction type."""
        if not value:
            return False
        detected = cls.detect_types(value)
        type_map = {"date": "date", "number": "amount", "currency": "amount",
                     "amount": "amount", "duration": "duration",
                     "percentage": "percentage"}
        expected = type_map.get(expected_type, expected_type)
        return any(d["type"] == expected for d in detected)


# ── Confidence scoring ────────────────────────────────────────────────────────


def _score_extraction(
    method: str,
    raw_confidence: float,
    embedding_score: float | None = None,
    type_valid: bool = False,
    has_example_overlap: bool = False,
) -> float:
    """
    Weighted confidence score (ported from ConfidenceScorer).

    Weights: embedding 0.40, pattern 0.25, keyword 0.15, type 0.10, example 0.10
    """
    scores = {
        "embedding_similarity": embedding_score if embedding_score is not None else 0.5,
        "pattern_match": 0.9 if method == "pattern" else raw_confidence,
        "keyword_proximity": raw_confidence if method == "keyword" else 0.5,
        "type_validation": 1.0 if type_valid else 0.5,
        "example_match": 0.8 if has_example_overlap else 0.5,
    }
    weights = {
        "embedding_similarity": 0.40,
        "pattern_match": 0.25,
        "keyword_proximity": 0.15,
        "type_validation": 0.10,
        "example_match": 0.10,
    }
    return round(sum(scores[k] * weights[k] for k in weights), 3)


# ── Main Orchestrator ─────────────────────────────────────────────────────────


class FieldFeatureExtractor:
    """
    Multi-strategy field feature extraction from document chunks.

    Loads 221 predefined + 500+ business field configs and runs
    pattern → keyword → type-inference extraction on each chunk.
    """

    def __init__(self, configs: list[RichFieldConfig]):
        self.configs = configs
        self._predefined = [c for c in configs if c.source == "predefined"]
        self._business = [c for c in configs if c.source == "business"]
        self._by_name: dict[str, RichFieldConfig] = {c.name: c for c in configs}
        logger.info(f"FieldFeatureExtractor: {len(self._predefined)} predefined, "
                     f"{len(self._business)} business fields")

    @classmethod
    def from_legacy_dir(cls, legacy_dir: Path | str | None = None) -> "FieldFeatureExtractor":
        """Load configs from apr-fields-processor data directory."""
        root = Path(legacy_dir) if legacy_dir else _DEFAULT_LEGACY_DIR
        predefined = load_predefined_fields(root / _PREDEFINED_DIR)
        business = load_business_fields(root / _BUSINESS_FILE)
        return cls(predefined + business)

    @property
    def predefined_count(self) -> int:
        return len(self._predefined)

    @property
    def business_count(self) -> int:
        return len(self._business)

    @property
    def total_count(self) -> int:
        return len(self.configs)

    def get_field(self, name: str) -> RichFieldConfig | None:
        return self._by_name.get(name)

    def get_fields_by_category(self, category: str) -> list[RichFieldConfig]:
        return [c for c in self.configs if category.lower() in c.taxonomy_category.lower()]

    def extract_features(self, text: str, max_matches_per_field: int = 3) -> ChunkFieldFeatures:
        """
        Extract field features from a single chunk of text.

        Runs predefined field configs (pattern + keyword) against the text,
        then detects typed values (date, amount, duration, percentage).

        Returns ChunkFieldFeatures with all matches.
        """
        features = ChunkFieldFeatures()

        # Strategy 1: Pattern-based extraction (predefined fields with regex)
        for fc in self._predefined:
            if not fc.regex_patterns:
                continue
            pattern_matches = PatternMatcher.extract(
                text, fc.regex_patterns,
                context_window=fc.context_window,
                max_matches=max_matches_per_field,
            )
            for pm in pattern_matches:
                type_valid = TypeDetector.validate_type(pm["value"], fc.extraction_type)
                conf = _score_extraction("pattern", pm["confidence"],
                                         type_valid=type_valid)
                features.matches.append(FieldMatch(
                    field_name=fc.name,
                    extraction_type=fc.extraction_type,
                    taxonomy_category=fc.taxonomy_category,
                    method="pattern",
                    value=pm["value"][:200],
                    confidence=conf,
                    position=pm["start"],
                    context=pm.get("context", "")[:300],
                ))

        # Strategy 2: Keyword proximity (predefined fields with key_terms)
        for fc in self._predefined:
            if not fc.key_terms:
                continue
            # Skip if we already have pattern matches for this field
            if fc.name in {m.field_name for m in features.matches if m.method == "pattern"}:
                continue
            kw_matches = KeywordMatcher.find_near_keywords(
                text, fc.key_terms[:10],
                window_after=fc.context_window,
            )
            for km in kw_matches[:max_matches_per_field]:
                type_valid = TypeDetector.validate_type(km["value"], fc.extraction_type)
                conf = _score_extraction("keyword", km["confidence"],
                                         type_valid=type_valid)
                features.matches.append(FieldMatch(
                    field_name=fc.name,
                    extraction_type=fc.extraction_type,
                    taxonomy_category=fc.taxonomy_category,
                    method="keyword",
                    value=km["value"][:200],
                    confidence=conf,
                    position=km["start"],
                ))

        # Strategy 3: Type inference (detect dates, amounts, durations, percentages)
        typed_values = TypeDetector.detect_types(text)
        for tv in typed_values[:10]:
            features.matches.append(FieldMatch(
                field_name=f"_detected_{tv['type']}",
                extraction_type=tv["type"],
                taxonomy_category="type_inference",
                method="type_inference",
                value=tv["value"][:200],
                confidence=tv["confidence"],
                position=tv["start"],
            ))

        # Aggregate
        features.matched_field_names = {m.field_name for m in features.matches
                                         if not m.field_name.startswith("_detected_")}
        features.matched_categories = {m.taxonomy_category for m in features.matches
                                        if m.taxonomy_category != "type_inference"}
        features.top_confidence = max((m.confidence for m in features.matches), default=0.0)

        return features

    def batch_extract_features(
        self, texts: list[str], max_matches_per_field: int = 3,
    ) -> list[ChunkFieldFeatures]:
        """Extract features from multiple chunk texts."""
        return [self.extract_features(t, max_matches_per_field) for t in texts]

    def get_field_feature_summary(self, features_list: list[ChunkFieldFeatures]) -> dict:
        """Aggregate field feature stats across all chunks of a document."""
        all_fields: dict[str, int] = {}
        all_categories: dict[str, int] = {}
        total_matches = 0

        for feat in features_list:
            total_matches += len(feat.matches)
            for fn in feat.matched_field_names:
                all_fields[fn] = all_fields.get(fn, 0) + 1
            for cat in feat.matched_categories:
                all_categories[cat] = all_categories.get(cat, 0) + 1

        return {
            "total_matches": total_matches,
            "unique_fields": len(all_fields),
            "unique_categories": len(all_categories),
            "top_fields": sorted(all_fields.items(), key=lambda x: x[1], reverse=True)[:15],
            "category_distribution": dict(sorted(all_categories.items(),
                                                  key=lambda x: x[1], reverse=True)),
        }

    def enrich_summary_with_features(
        self,
        base_summary: str,
        features: ChunkFieldFeatures,
        max_feature_lines: int = 5,
    ) -> str:
        """
        Append field feature annotations to a base summary string.

        Used to enrich field-aware doc summaries with evidence from
        pattern/keyword matches before embedding for macro clustering.
        """
        if not features.matches:
            return base_summary

        # Deduplicate by field name, keep highest confidence
        best: dict[str, FieldMatch] = {}
        for m in features.matches:
            if m.field_name.startswith("_detected_"):
                continue
            existing = best.get(m.field_name)
            if not existing or m.confidence > existing.confidence:
                best[m.field_name] = m

        top = sorted(best.values(), key=lambda m: m.confidence, reverse=True)[:max_feature_lines]
        if not top:
            return base_summary

        feature_lines = [
            f"[{m.field_name}|{m.method}|{m.confidence:.2f}] {m.value[:100]}"
            for m in top
        ]
        return base_summary + "\n\nDetected fields:\n" + "\n".join(feature_lines)
