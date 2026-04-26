#!/usr/bin/env python3
"""
RLM Full Pipeline — DSPy RLM for Both Discovery AND Extraction
================================================================

Uses DSPy's RLM (Recursive Language Model with Python REPL) for both:
  1. Field Discovery — same as core/rlm_v2_dspy.py (reuses ClusterDiscoveryPipeline)
  2. Field Extraction — NEW: gives the RLM all chunks + discovered fields and lets
     it write Python code to systematically extract values from each chunk.

The extraction RLM has access to:
  - All chunk texts (as a Python list)
  - Discovered field definitions
  - llm_query() tool for calling the sub-LLM on individual chunks
  - Python's re, json, etc. for pattern matching

This combines the "sequential chunk extraction" pattern with the power of
programmatic exploration — the RLM can try multiple strategies per field
(regex first, then LLM fallback) and aggregate across chunks.

Self-contained: only needs core/rlm_v2_dspy.py for discovery.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import dspy
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DSPy SIGNATURE FOR RLM EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

class RLMExtractionSig(dspy.Signature):
    """You are a legal contract extraction agent with access to a Python REPL.
    Given a list of field definitions and clause text chunks, systematically
    extract the value of each field from the chunks.

    STRATEGY — follow these steps:
    1. Parse fields_json to get the list of fields to extract.
    2. Parse chunks_json to get the list of text chunks.
    3. For each field:
       a. First try REGEX: search all chunks for patterns matching the field type
          (dates, amounts, durations, names, etc.). Print any matches found.
       b. If regex finds nothing, use llm_query() to ask the sub-LLM to extract
          the specific field from the most relevant chunks.
       c. If multiple values found, pick the most specific/complete one.
    4. For values found, VERIFY they are verbatim in the source text.
       Only keep values where the exact text appears in a chunk.
    5. Build the final JSON result and SUBMIT.

    RULES:
    - Extract ONLY values explicitly stated in the text (verbatim preferred).
    - Return null for fields not found in any chunk.
    - Do NOT fabricate or infer values.
    - For each extracted value, include which chunk index it came from.
    - Aim for HIGH PRECISION over HIGH RECALL — skip uncertain values.

    OUTPUT FORMAT: JSON object mapping field names to:
    {{"field_name": {{"value": "extracted text", "chunk_index": <int>, "method": "regex|llm"}}}}
    Use null as value if not found."""

    fields_json: str = dspy.InputField(desc="JSON array of field definitions: [{name, type, description, examples}, ...]")
    chunks_json: str = dspy.InputField(desc="JSON array of chunk texts to extract from")
    cluster_label: str = dspy.InputField(desc="The clause type label (e.g. 'Franchise agreement')")
    extraction_hints: str = dspy.InputField(desc="Optional hints about what to look for (from discovery context)")

    extraction_json: str = dspy.OutputField(
        desc='JSON object mapping field names to {value, chunk_index, method} or null. '
             'Example: {"fee_amount": {"value": "$50,000", "chunk_index": 2, "method": "regex"}, "term_years": null}'
    )


# ══════════════════════════════════════════════════════════════════════════════
# RLM EXTRACTION MODULE
# ══════════════════════════════════════════════════════════════════════════════

class RLMExtractor(dspy.Module):
    """Uses DSPy RLM to programmatically extract field values from chunks.

    The RLM gets a Python REPL with llm_query() and can write code to:
    - Iterate over chunks searching for field values
    - Use regex patterns for structured data (dates, amounts)
    - Call the sub-LLM for complex extractions
    - Verify values are grounded in source text
    """

    def __init__(self, max_iterations: int = 30, max_llm_calls: int = 60,
                 sub_lm: "dspy.LM | None" = None):
        super().__init__()
        self.rlm = dspy.RLM(
            RLMExtractionSig,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            verbose=False,
            sub_lm=sub_lm,
        )

    def forward(self, fields_json: str, chunks_json: str,
                cluster_label: str, extraction_hints: str = "") -> dspy.Prediction:
        return self.rlm(
            fields_json=fields_json,
            chunks_json=chunks_json,
            cluster_label=cluster_label,
            extraction_hints=extraction_hints,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RLMFullConfig:
    """Configuration for the RLM full pipeline."""
    # Discovery (reuses ClusterDiscoveryPipeline defaults)
    enable_deep_explore: bool = False
    # Extraction RLM
    rlm_max_iterations: int = 30
    rlm_max_llm_calls: int = 60
    max_chunks_for_extraction: int = 8
    max_chunk_chars: int = 3000    # trim chunks to fit in RLM context
    # Grounding
    grounding_threshold: float = 0.8


# ══════════════════════════════════════════════════════════════════════════════
# GROUNDING UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _ground_value(value: str, chunk_texts: list[str]) -> tuple[str, float]:
    """Check if extracted value is grounded in any chunk text."""
    if not value:
        return "missing", 0.0
    v_lower = value.lower().strip()
    for ct in chunk_texts:
        if v_lower in ct.lower():
            return "verbatim", 1.0
    # Token overlap
    v_tokens = set(re.findall(r'\b\w+\b', v_lower))
    if not v_tokens:
        return "missing", 0.0
    all_tokens = set()
    for ct in chunk_texts:
        all_tokens.update(re.findall(r'\b\w+\b', ct.lower()))
    overlap = len(v_tokens & all_tokens) / len(v_tokens)
    if overlap >= 0.8:
        return "partial", 0.8
    return "hallucinated", 0.0


# ══════════════════════════════════════════════════════════════════════════════
# RLM FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class RLMFullPipeline:
    """
    Full pipeline using DSPy RLM for both discovery and extraction.

    Discovery: Uses ClusterDiscoveryPipeline (with ChainOfThought + optional RLM)
    Extraction: Uses RLMExtractor (RLM writes Python code to systematically extract)
    """

    def __init__(self, config: Optional[RLMFullConfig] = None,
                 embedder=None, lm=None):
        self.config = config or RLMFullConfig()
        self._embedder = embedder
        self._lm = lm
        self._discovery_pipeline = None
        self._extractor = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    @property
    def discovery_pipeline(self):
        if self._discovery_pipeline is None:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from core.rlm_v2_dspy import ClusterDiscoveryPipeline
            from core.field_discovery import FieldDiscoveryConfig
            self._discovery_pipeline = ClusterDiscoveryPipeline(
                config=FieldDiscoveryConfig(),
                embedder=self.embedder,
                enable_deep_explore=self.config.enable_deep_explore,
            )
        return self._discovery_pipeline

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = RLMExtractor(
                max_iterations=self.config.rlm_max_iterations,
                max_llm_calls=self.config.rlm_max_llm_calls,
            )
        return self._extractor

    def run(
        self,
        cluster_label: str,
        keywords: str,
        chunk_texts: list[str],
        n_extract_chunks: int = 8,
    ) -> dict:
        """
        Run full RLM pipeline: discovery + RLM extraction.

        Returns dict with:
            fields: list[dict] — discovered fields
            extractions: dict — field_name → {value, chunk_index, method, grounding}
            disc_meta: dict — discovery metadata
            ext_meta: dict — extraction metadata
        """
        t0 = time.time()

        # ═══ Phase 1: Discovery (reuses existing DSPy pipeline) ═══
        t_disc = time.time()
        try:
            disc_result = self.discovery_pipeline(
                cluster_label=cluster_label,
                keywords=keywords,
                chunk_texts=chunk_texts,
            )
            fields = disc_result.fields
            disc_meta = disc_result.meta
        except Exception as e:
            logger.error(f"Discovery failed: {e}")
            fields = []
            disc_meta = {"error": str(e)}
        disc_time = round(time.time() - t_disc, 2)

        if not fields:
            return {
                "fields": [],
                "extractions": {},
                "disc_meta": disc_meta,
                "ext_meta": {"error": "no_fields_discovered"},
                "discovery_time_s": disc_time,
                "extraction_time_s": 0,
                "total_time_s": round(time.time() - t0, 2),
            }

        # ═══ Phase 2: RLM Extraction ═══
        extract_chunks = chunk_texts[:n_extract_chunks]

        # Trim chunks for RLM context budget
        trimmed_chunks = [c[:self.config.max_chunk_chars] for c in extract_chunks]

        # Build extraction hints from discovery metadata
        hints = self._build_extraction_hints(disc_meta, cluster_label)

        # Prepare inputs for RLM
        fields_payload = json.dumps(fields, ensure_ascii=False)
        chunks_payload = json.dumps(trimmed_chunks, ensure_ascii=False)

        t_ext = time.time()
        ext_meta = {
            "n_fields": len(fields),
            "n_chunks": len(extract_chunks),
            "rlm_success": False,
        }

        try:
            rlm_result = self.extractor(
                fields_json=fields_payload,
                chunks_json=chunks_payload,
                cluster_label=cluster_label,
                extraction_hints=hints,
            )
            raw_extractions = self._parse_extraction_output(rlm_result.extraction_json)
            ext_meta["rlm_success"] = True
        except Exception as e:
            logger.error(f"RLM extraction failed: {e}")
            raw_extractions = {}
            ext_meta["error"] = str(e)

        ext_time = round(time.time() - t_ext, 2)

        # ═══ Phase 3: Grounding Verification ═══
        extractions = {}
        n_verified = 0
        n_halluc = 0
        n_missing = 0

        for f in fields:
            fname = f["name"]
            raw = raw_extractions.get(fname)
            if raw is None or raw.get("value") is None:
                extractions[fname] = {
                    "value": None,
                    "chunk_index": -1,
                    "method": "none",
                    "grounding_level": "missing",
                    "grounding_score": 0.0,
                }
                n_missing += 1
                continue

            value = str(raw["value"])
            chunk_idx = raw.get("chunk_index", -1)
            method = raw.get("method", "unknown")

            # Verify grounding against source chunks
            grounding_level, grounding_score = _ground_value(value, extract_chunks)

            extractions[fname] = {
                "value": value,
                "chunk_index": chunk_idx,
                "method": method,
                "grounding_level": grounding_level,
                "grounding_score": grounding_score,
            }

            if grounding_level == "verbatim":
                n_verified += 1
            elif grounding_level == "hallucinated":
                n_halluc += 1

        # Aggregate stats
        n_extracted = len(fields) - n_missing
        ext_meta.update({
            "n_extracted": n_extracted,
            "n_verified": n_verified,
            "n_hallucinated": n_halluc,
            "n_missing": n_missing,
            "verified_rate": round(n_verified / n_extracted, 3) if n_extracted else 0,
            "hallucination_rate": round(n_halluc / n_extracted, 3) if n_extracted else 0,
            "grounded_rate": round((n_extracted - n_halluc) / n_extracted, 3) if n_extracted else 0,
            "fill_rate": round(n_extracted / len(fields), 3) if fields else 0,
        })

        total_time = round(time.time() - t0, 2)

        return {
            "fields": fields,
            "extractions": extractions,
            "disc_meta": disc_meta,
            "ext_meta": ext_meta,
            "discovery_time_s": disc_time,
            "extraction_time_s": ext_time,
            "total_time_s": total_time,
        }

    # ── Helper Methods ────────────────────────────────────────────────

    def _build_extraction_hints(self, disc_meta: dict, cluster_label: str) -> str:
        """Build extraction hints from discovery metadata."""
        hints = [f"Clause type: {cluster_label}"]

        clause_types = disc_meta.get("clause_types", [])
        if clause_types:
            for ct in clause_types[:3]:
                name = ct.get("name", "") if isinstance(ct, dict) else ""
                desc = ct.get("emergent_description", "") if isinstance(ct, dict) else ""
                if name:
                    hints.append(f"Sub-pattern: {name} — {desc}")

        obligations = disc_meta.get("obligations", [])
        if obligations:
            for ob in obligations[:3]:
                if isinstance(ob, dict):
                    obligor = ob.get("obligor", "")
                    action = ob.get("action", "")
                    if obligor and action:
                        hints.append(f"Obligation: {obligor} must {action}")

        return "\n".join(hints)

    def _parse_extraction_output(self, raw) -> dict:
        """Parse RLM extraction output into dict of field→{value, chunk_index, method}."""
        if not raw:
            return {}

        if isinstance(raw, dict):
            return raw

        text = str(raw).strip()

        # Try direct JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return self._normalize_extraction_dict(parsed)
        except json.JSONDecodeError:
            pass

        # Try extracting from code fences
        match = re.search(r'```(?:json)?\s*(\{.+?\})\s*```', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, dict):
                    return self._normalize_extraction_dict(parsed)
            except json.JSONDecodeError:
                pass

        # Try finding any JSON object
        match = re.search(r'\{[^{}]*"[^"]+"\s*:\s*\{.*?\}.*?\}', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return self._normalize_extraction_dict(parsed)
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse RLM extraction output ({len(text)} chars)")
        return {}

    def _normalize_extraction_dict(self, parsed: dict) -> dict:
        """Normalize various output formats to standard {value, chunk_index, method}."""
        result = {}
        for fname, val in parsed.items():
            if val is None:
                result[fname] = None
                continue
            if isinstance(val, dict):
                result[fname] = {
                    "value": val.get("value"),
                    "chunk_index": val.get("chunk_index", -1),
                    "method": val.get("method", "unknown"),
                }
            elif isinstance(val, str):
                # Plain string value
                result[fname] = {
                    "value": val if val.lower() not in ("null", "none", "") else None,
                    "chunk_index": -1,
                    "method": "unknown",
                }
            else:
                result[fname] = {
                    "value": str(val),
                    "chunk_index": -1,
                    "method": "unknown",
                }
        return result


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Standalone test: run RLM full pipeline on one cluster."""
    import sys
    from pathlib import Path
    import duckdb
    from dotenv import load_dotenv
    load_dotenv()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.rlm_v2_dspy import _configure_dspy_lm

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Configure DSPy LM
    lm = _configure_dspy_lm()

    db_path = "data/cuad_510_demo.duckdb"
    conn = duckdb.connect(db_path, read_only=True)

    cluster_id = "ct_f778e156"  # Franchise agreement
    cluster = conn.execute(
        "SELECT label, keywords FROM clusters WHERE cluster_id = ?", [cluster_id]
    ).fetchdf().to_dict("records")[0]

    chunks = conn.execute("""
        SELECT ch.chunk_text FROM chunks ch
        JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
        WHERE ca.cluster_id = ?
        ORDER BY ch.chunk_index
    """, [cluster_id]).fetchdf()["chunk_text"].tolist()

    keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(cluster.get("keywords"), str) else (cluster.get("keywords") or [])
    keywords_str = ", ".join(keywords[:10]) if keywords else ""

    conn.close()

    # Run pipeline
    pipeline = RLMFullPipeline()
    result = pipeline.run(
        cluster_label=cluster["label"],
        keywords=keywords_str,
        chunk_texts=chunks,
        n_extract_chunks=8,
    )

    print("\n=== RLM Full Pipeline Results ===")
    print(f"  Fields discovered: {len(result['fields'])}")
    print(f"  Discovery time: {result['discovery_time_s']}s")
    print(f"  Extraction time: {result['extraction_time_s']}s")
    print(f"\n  Extraction stats:")
    for k, v in result['ext_meta'].items():
        print(f"    {k}: {v}")
    print(f"\n  Extractions:")
    for fname, ext in result['extractions'].items():
        if ext['value'] is not None:
            print(f"    {fname}: {ext['value'][:60]}... "
                  f"(ground={ext['grounding_level']}, method={ext['method']})")


if __name__ == "__main__":
    main()
