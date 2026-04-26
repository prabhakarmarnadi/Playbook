"""
Field Discovery — Stage 6 of the clustering pipeline.

For each clause cluster (micro cluster), uses Azure OpenAI to analyze
representative chunks and discover recurring fields/attributes.

Produces:
  - field_definitions: discovered fields per cluster (name, type, description)
  - extractions: extracted values per agreement per field

Ontology levels:
  Domain (macro)  →  Clause Cluster (micro)  →  Fields (this stage)
  "SaaS Agreement"  →  "Termination Clause"  →  notice_period, cure_days, ...
"""
import json
import logging
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# ── DSPy RLM helpers ────────────────────────────────────────────────────────

def _get_dspy_lm() -> "dspy.LM":
    """Build a dspy.LM from the existing env-var configuration with retry logic."""
    import dspy
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")

    # num_retries + max_tokens apply to all sub-LM calls inside RLM
    retry_kwargs = dict(num_retries=5)

    if endpoint and api_key:
        return dspy.LM(
            f"azure/{deployment}",
            api_key=api_key,
            api_base=endpoint,
            api_version=api_version,
            **retry_kwargs,
        )
    openai_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return dspy.LM(f"openai/{model}", api_key=openai_key, **retry_kwargs)


@dataclass
class FieldDiscoveryConfig:
    max_sample_chunks: int = 30
    max_fields_per_cluster: int = 15
    temperature: float = 0.2
    model: str = ""             # Azure deployment name (from env if empty)
    max_retries: int = 3
    retry_delay: float = 2.0


def _get_azure_client():
    """Create Azure OpenAI client from env vars."""
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    if not endpoint or not api_key:
        raise ValueError(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars. "
            "Example: export AZURE_OPENAI_ENDPOINT='https://myorg.openai.azure.com/'"
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def _get_deployment(config: FieldDiscoveryConfig) -> str:
    if config.model:
        return config.model
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")


FIELD_DISCOVERY_PROMPT = """You are a legal contract analyst. Analyze these clause excerpts that all belong to the same clause type: "{cluster_label}".

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

FIELD_EXTRACTION_PROMPT = """Extract field values from this contract clause.

Clause type: "{cluster_label}"
Clause text:
{chunk_text}

Fields to extract:
{fields_schema}

Respond with ONLY a JSON object mapping field names to extracted values.
Use null if the field is not present. For booleans use true/false.

Example: {{"notice_period": "30 days", "auto_renewal": true, "termination_fee": null}}"""

FIELD_EXTRACTION_PROMPT_ENHANCED = """Extract field values from this contract clause.
IMPORTANT: Only extract values that appear explicitly in the text. Do NOT hallucinate or infer values not stated.

Clause type: "{cluster_label}"
{rlm_context_section}
Clause text:
{chunk_text}

Fields to extract (with descriptions and example values from similar clauses):
{fields_schema}

Rules:
1. Extract ONLY values explicitly stated in the text above.
2. If a field's value is not clearly stated in the text, return null for that field.
3. Use the exact wording from the text when possible (prefer verbatim quotes).
4. Do not copy the field name or clause type as a value.
5. Do not fabricate URLs, dates, or names not in the text.

Respond with ONLY a JSON object mapping field names to extracted values.
Use null if the field is not present. For booleans use true/false.

Example: {{"notice_period": "30 days", "auto_renewal": true, "termination_fee": null}}"""


def _call_azure(client, deployment: str, prompt: str, config: FieldDiscoveryConfig,
                temperature: float | None = None, max_tokens: int = 2000) -> dict | None:
    """Call Azure OpenAI with retries, return parsed JSON or None."""
    temp = temperature if temperature is not None else config.temperature
    for attempt in range(config.max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"JSON parse error (attempt {attempt+1})")
        except Exception as e:
            logger.warning(f"Azure API error (attempt {attempt+1}): {e}")
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (attempt + 1))
    return None


def _build_arm_context(cluster_label: str, relationship_layer=None) -> dict[str, str]:
    """Build ARM relationship context for field discovery prompts.

    Returns dict with keys: related_clauses, field_correlations, term_packages.
    All values are formatted strings ready for prompt injection.
    """
    if relationship_layer is None:
        return {
            "related_clauses": "No relationship data available.",
            "field_correlations": "No field correlation data available.",
            "term_packages": "No term package data available.",
        }

    related = relationship_layer.get_related_clause_types(cluster_label)
    if related:
        lines = []
        for r in related[:10]:
            lines.append(f"- {r['clause_type']} (confidence: {r['confidence']:.2f}, lift: {r['lift']:.2f}, {r['direction']})")
        related_str = "\n".join(lines)
    else:
        related_str = "No related clause types discovered."

    implied = relationship_layer.get_implied_fields([cluster_label])
    if implied:
        lines = []
        for f in implied[:10]:
            lines.append(f"- {f['field_item']} (confidence: {f['confidence']:.2f})")
        field_str = "\n".join(lines)
    else:
        field_str = "No field correlations discovered."

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


def discover_fields_for_cluster(
    client, deployment: str,
    cluster_label: str, keywords: list[str],
    chunk_texts: list[str], config: FieldDiscoveryConfig,
) -> list[dict]:
    """Discover fields in a cluster's chunks. Returns list of {name, type, description, examples}."""
    sample = chunk_texts[:config.max_sample_chunks]
    chunks_text = "\n\n".join(
        f"[Excerpt {i+1}]\n{text[:1500]}" for i, text in enumerate(sample)
    )

    prompt = FIELD_DISCOVERY_PROMPT.format(
        cluster_label=cluster_label,
        keywords=", ".join(keywords[:10]),
        chunks_text=chunks_text,
        max_fields=config.max_fields_per_cluster,
    )

    parsed = _call_azure(client, deployment, prompt, config)
    if not parsed:
        return []

    # Handle both {"fields": [...]} and bare [...]
    if isinstance(parsed, dict):
        fields = parsed.get("fields", [])
    elif isinstance(parsed, list):
        fields = parsed
    else:
        return []

    valid_types = {"text", "date", "duration", "currency", "percentage",
                   "integer", "boolean", "entity_name", "enum"}
    result = []
    for f in fields[:config.max_fields_per_cluster]:
        name = f.get("name", "").strip()
        if not name:
            continue
        name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
        ftype = f.get("type", "text").strip().lower()
        if ftype not in valid_types:
            ftype = "text"
        result.append({
            "name": name,
            "type": ftype,
            "description": f.get("description", ""),
            "examples": f.get("examples", [])[:5],
        })
    return result


# ── RLM-based field discovery (Stage 6 alternative) ──────────────────────────

def discover_fields_for_cluster_rlm(
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config: FieldDiscoveryConfig,
) -> list[dict]:
    """
    RLM-based field discovery — lets the LLM programmatically explore ALL
    chunks via a sandboxed REPL instead of seeing only a 30-chunk sample.

    The LLM writes Python code to:
      1. Scan all chunks for recurring patterns (regex for dates, amounts, etc.)
      2. Strategically sample diverse chunks
      3. Call llm_query() on subsets for semantic field identification
      4. Aggregate and deduplicate results

    Falls back to standard discover_fields_for_cluster() on error.
    """
    import dspy
    from config import RLM_SUB_MODEL, RLM_MAX_ITERATIONS, RLM_MAX_LLM_CALLS

    lm = _get_dspy_lm()
    dspy.configure(lm=lm)

    # Use a cheaper sub-LM for llm_query() calls if configured
    sub_lm = lm
    if RLM_SUB_MODEL:
        sub_lm = dspy.LM(RLM_SUB_MODEL, num_retries=5)

    # Build a DSPy Signature with a docstring that guides the RLM
    class FieldDiscoverySig(dspy.Signature):
        """You are a legal contract analyst. Given clause excerpts (chunks_json)
        belonging to a clause type (cluster_label), discover ALL recurring structured
        fields. Return fields_json as a JSON list of objects with keys: name
        (snake_case), type (text|date|duration|currency|percentage|integer|boolean|
        entity_name|enum), description (one sentence), examples (2-3 values).
        Only include fields appearing in 2+ chunks. Return up to max_fields fields.

        STRATEGY — follow these steps:
        1. Parse chunks_json and print len, then print 5 diverse chunks in full.
        2. Use llm_query_batched to send 3-6 batches of chunks to the sub-LLM,
           each batch asking: 'List every extractable field (name, type, description,
           2 examples) from these contract clauses.' Cover ALL chunks across batches.
        3. Aggregate field lists from all batches. Count how many batches mention
           each field name. Keep fields mentioned in 2+ batches.
        4. Deduplicate near-synonyms (e.g. notice_period vs notice_timing).
        5. Build the final list with up to max_fields fields and SUBMIT.

        Be thorough — aim for 10-15 fields per cluster. Miss nothing."""

        chunks_json: str = dspy.InputField(desc="JSON array of clause text excerpts")
        cluster_label: str = dspy.InputField(desc="The clause type label")
        keywords: str = dspy.InputField(desc="Comma-separated clause keywords")
        max_fields: str = dspy.InputField(desc="Maximum number of fields to return")
        fields_json: str = dspy.OutputField(desc="JSON array of {name, type, description, examples} objects")

    rlm = dspy.RLM(
        FieldDiscoverySig,
        max_iterations=RLM_MAX_ITERATIONS,
        max_llm_calls=RLM_MAX_LLM_CALLS,
        verbose=False,
        sub_lm=sub_lm,
    )

    # Cap chunks to avoid overwhelming the REPL context window.
    # Keep first 20 + randomly sample 40 more for diversity.
    max_rlm_chunks = 60
    if len(chunk_texts) > max_rlm_chunks:
        import random
        rng = random.Random(42)
        head = chunk_texts[:20]
        tail_pool = chunk_texts[20:]
        sampled = rng.sample(tail_pool, min(max_rlm_chunks - 20, len(tail_pool)))
        capped_texts = head + sampled
    else:
        capped_texts = chunk_texts

    # Serialize chunks for the RLM REPL environment
    chunks_payload = json.dumps(capped_texts, ensure_ascii=False)

    try:
        result = rlm(
            chunks_json=chunks_payload,
            cluster_label=cluster_label,
            keywords=", ".join(keywords[:10]),
            max_fields=str(config.max_fields_per_cluster),
        )
        raw = result.fields_json

        # Log trajectory for observability / audit
        trajectory = getattr(result, "trajectory", None)
        if trajectory:
            n_steps = len(trajectory)
            logger.info(f"RLM field discovery for '{cluster_label}': {n_steps} REPL steps")
            for i, step in enumerate(trajectory):
                code = step.get("code", "")[:200]
                output = step.get("output", "")[:200]
                logger.debug(f"  Step {i+1}: code={code!r} output={output!r}")
    except Exception as e:
        logger.warning(f"RLM field discovery failed, falling back to standard: {e}")
        return []

    # Parse the RLM output (should be JSON array of field dicts)
    if not raw or (isinstance(raw, str) and not raw.strip()):
        logger.warning("RLM returned empty output")
        return []

    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        # RLM often returns Python literals (single quotes) instead of JSON.
        # Use ast.literal_eval as a safe fallback.
        import ast
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            pass
        else:
            if isinstance(parsed, (list, dict)):
                # Successfully parsed as Python literal
                logger.debug("Parsed RLM output via ast.literal_eval")
                # Skip to the validation below
                if isinstance(parsed, dict):
                    parsed = parsed.get("fields", [])
                # Jump past the JSON fallback block
                fields = parsed if isinstance(parsed, list) else []
                # Go directly to normalization
                valid_types = {"text", "date", "duration", "currency", "percentage",
                               "integer", "boolean", "entity_name", "enum"}
                result_fields = []
                for f in fields[:config.max_fields_per_cluster]:
                    if not isinstance(f, dict):
                        continue
                    name = (f.get("name") or f.get("field_name") or f.get("field") or "").strip()
                    if not name:
                        continue
                    name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
                    ftype = f.get("type", "text").strip().lower()
                    if ftype not in valid_types:
                        ftype = "text"
                    result_fields.append({
                        "name": name,
                        "type": ftype,
                        "description": f.get("description", ""),
                        "examples": f.get("examples") or ([f["field_value"]] if f.get("field_value") else []),
                    })
                return result_fields
    except json.JSONDecodeError:
        # Try extracting JSON from markdown fences
        match = re.search(r'```(?:json)?\s*(.+?)```', raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                # Try finding any JSON array in the raw output
                arr_match = re.search(r'\[.*\]', raw, re.DOTALL)
                if arr_match:
                    try:
                        parsed = json.loads(arr_match.group(0))
                    except json.JSONDecodeError:
                        logger.warning("RLM returned unparseable JSON")
                        return []
                else:
                    logger.warning("RLM returned unparseable JSON")
                    return []
        else:
            # Try finding any JSON array in the raw output
            arr_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if arr_match:
                try:
                    parsed = json.loads(arr_match.group(0))
                except json.JSONDecodeError:
                    logger.warning("RLM returned unparseable output")
                    return []
            else:
                logger.warning("RLM returned unparseable output")
                return []

    if isinstance(parsed, dict):
        fields = parsed.get("fields", [])
    elif isinstance(parsed, list):
        fields = parsed
    else:
        return []

    # Normalize (same validation as standard path)
    # Handle alternate key names that RLM sometimes produces
    valid_types = {"text", "date", "duration", "currency", "percentage",
                   "integer", "boolean", "entity_name", "enum"}
    result_fields = []
    for f in fields[:config.max_fields_per_cluster]:
        if not isinstance(f, dict):
            continue
        # Accept 'name', 'field_name', or 'field' key
        name = (f.get("name") or f.get("field_name") or f.get("field") or "").strip()
        if not name:
            continue
        name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
        ftype = f.get("type", "text").strip().lower()
        if ftype not in valid_types:
            ftype = "text"
        result_fields.append({
            "name": name,
            "type": ftype,
            "description": f.get("description", ""),
            "examples": f.get("examples") or ([f["field_value"]] if f.get("field_value") else []),
        })
    return result_fields


def _merge_field_lists(
    fields_a: list[dict],
    fields_b: list[dict],
    max_fields: int = 15,
) -> list[dict]:
    """Merge two field lists using exact + fuzzy name matching.

    Priority:
      1. Fields found by BOTH approaches (highest confidence).
      2. Fields unique to either approach, ranked by description quality.

    For duplicates, prefer the entry with the longer description and merge
    example lists.
    """
    map_a = {f["name"]: f for f in fields_a}
    map_b = {f["name"]: f for f in fields_b}

    # Exact matches
    common_names = set(map_a.keys()) & set(map_b.keys())

    # Fuzzy matches: substring containment (e.g. "notice_period" ∈ "termination_notice_period")
    remaining_a = {n for n in map_a if n not in common_names}
    remaining_b = {n for n in map_b if n not in common_names}
    fuzzy_pairs: list[tuple[str, str]] = []
    matched_b: set[str] = set()
    for na in sorted(remaining_a):
        for nb in sorted(remaining_b):
            if nb in matched_b:
                continue
            if na in nb or nb in na:
                fuzzy_pairs.append((na, nb))
                matched_b.add(nb)
                break

    # Build merged list
    merged: list[dict] = []

    def _pick_best(fa: dict, fb: dict) -> dict:
        """Pick the richer definition and merge examples."""
        best = fa if len(fa.get("description", "")) >= len(fb.get("description", "")) else fb
        examples = list(dict.fromkeys(
            (fa.get("examples") or []) + (fb.get("examples") or [])
        ))[:5]
        return {**best, "examples": examples, "source": "both"}

    # 1. Exact matches first (both approaches agree → highest signal)
    for name in sorted(common_names):
        merged.append(_pick_best(map_a[name], map_b[name]))

    # 2. Fuzzy matches (semantically same field, different names)
    fuzzy_a_used = set()
    for na, nb in fuzzy_pairs:
        fa, fb = map_a[na], map_b[nb]
        entry = _pick_best(fa, fb)
        # Use the shorter name (more canonical)
        entry["name"] = na if len(na) <= len(nb) else nb
        entry["source"] = "both_fuzzy"
        merged.append(entry)
        fuzzy_a_used.add(na)
        # nb already in matched_b

    # 3. Unique to A (Standard) — reliable, fast
    for na in sorted(remaining_a - fuzzy_a_used):
        entry = {**map_a[na], "source": "standard"}
        merged.append(entry)

    # 4. Unique to B (RLM) — complementary deep-dive fields
    for nb in sorted(remaining_b - matched_b):
        entry = {**map_b[nb], "source": "rlm"}
        merged.append(entry)

    return merged[:max_fields]


def discover_fields_hybrid(
    client,
    deployment: str,
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config: FieldDiscoveryConfig,
) -> tuple[list[dict], dict]:
    """Run Standard + RLM in sequence, merge results into a richer field set.

    Returns:
        (merged_fields, meta) where meta contains timing and provenance info.
    """
    import time as _time

    meta: dict = {"standard_ok": False, "rlm_ok": False}

    # ── Standard (fast, reliable) ──
    t0 = _time.time()
    try:
        fields_std = discover_fields_for_cluster(
            client, deployment, cluster_label, keywords, chunk_texts, config,
        )
        meta["standard_ok"] = True
    except Exception as e:
        logger.warning(f"Hybrid: standard failed: {e}")
        fields_std = []
    meta["standard_time"] = round(_time.time() - t0, 2)
    meta["standard_count"] = len(fields_std)

    # ── RLM (complementary, deeper) ──
    t0 = _time.time()
    try:
        fields_rlm = discover_fields_for_cluster_rlm(
            cluster_label, keywords, chunk_texts, config,
        )
        meta["rlm_ok"] = True
    except Exception as e:
        logger.warning(f"Hybrid: RLM failed: {e}")
        fields_rlm = []
    meta["rlm_time"] = round(_time.time() - t0, 2)
    meta["rlm_count"] = len(fields_rlm)

    # ── Merge ──
    if fields_std and fields_rlm:
        merged = _merge_field_lists(fields_std, fields_rlm, config.max_fields_per_cluster)
        meta["strategy"] = "merged"
    elif fields_std:
        merged = [{**f, "source": "standard"} for f in fields_std]
        meta["strategy"] = "standard_only"
    elif fields_rlm:
        merged = [{**f, "source": "rlm"} for f in fields_rlm]
        meta["strategy"] = "rlm_only"
    else:
        merged = []
        meta["strategy"] = "none"

    source_counts = {}
    for f in merged:
        src = f.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    meta["merged_count"] = len(merged)
    meta["source_breakdown"] = source_counts

    return merged, meta


def discover_fields_for_cluster_rlm_lb(
    cluster_label: str,
    keywords: list[str],
    chunk_texts: list[str],
    config: FieldDiscoveryConfig,
    briefing_tau: float = 1.5,
    briefing_max_tokens: int = 2000,
    adaptive_tau: bool = True,
) -> tuple[list[dict], dict]:
    """
    RLM field discovery with Latent Briefing memory sharing.

    Same RLM approach as discover_fields_for_cluster_rlm(), but each
    llm_query() worker call receives a compressed trajectory briefing
    instead of operating in isolation. This means:

    - Worker call 2 knows what worker call 1 already found
    - Worker call 3 gets a task-specific slice of all prior reasoning
    - Duplicate field discovery is reduced (workers share memory)
    - Later calls focus on gaps rather than re-discovering known fields

    Uses Text Latent Briefing (embedding-based scoring as proxy for
    attention) since we use Azure OpenAI API (no KV cache access).

    Args:
        cluster_label: The clause type label
        keywords: Clause keywords
        chunk_texts: All chunk texts in this cluster
        config: Field discovery config
        briefing_tau: Compaction aggressiveness (1.0=light, 2.0=aggressive)
        briefing_max_tokens: Max tokens in the briefing injected per worker call
        adaptive_tau: If True, adjust tau based on iteration depth

    Returns:
        (fields, briefing_meta) — discovered fields and latent briefing statistics
    """
    import dspy
    from config import RLM_SUB_MODEL, RLM_MAX_ITERATIONS, RLM_MAX_LLM_CALLS
    from core.rlm_with_memory import RLMWithLatentBriefing

    lm = _get_dspy_lm()
    dspy.configure(lm=lm)

    sub_lm = lm
    if RLM_SUB_MODEL:
        sub_lm = dspy.LM(RLM_SUB_MODEL, num_retries=5)

    class FieldDiscoverySig(dspy.Signature):
        """You are a legal contract analyst. Given clause excerpts (chunks_json)
        belonging to a clause type (cluster_label), discover ALL recurring structured
        fields. Return fields_json as a JSON list of objects with keys: name
        (snake_case), type (text|date|duration|currency|percentage|integer|boolean|
        entity_name|enum), description (one sentence), examples (2-3 values).
        Only include fields appearing in 2+ chunks. Return up to max_fields fields.

        STRATEGY — follow these steps:
        1. Parse chunks_json and print len, then print 5 diverse chunks in full.
        2. Use llm_query_batched to send 3-6 batches of chunks to the sub-LLM,
           each batch asking: 'List every extractable field (name, type, description,
           2 examples) from these contract clauses.' Cover ALL chunks across batches.
        3. Aggregate field lists from all batches. Count how many batches mention
           each field name. Keep fields mentioned in 2+ batches.
        4. Deduplicate near-synonyms (e.g. notice_period vs notice_timing).
        5. Build the final list with up to max_fields fields and SUBMIT.

        NOTE: Your llm_query calls receive context from prior analysis automatically
        via Latent Briefing. Each worker sees relevant memory from prior steps,
        so later batches are aware of what was already discovered. Use this to
        focus on finding NEW fields, not re-discovering known ones."""

        chunks_json: str = dspy.InputField(desc="JSON array of clause text excerpts")
        cluster_label: str = dspy.InputField(desc="The clause type label")
        keywords: str = dspy.InputField(desc="Comma-separated clause keywords")
        max_fields: str = dspy.InputField(desc="Maximum number of fields to return")
        fields_json: str = dspy.OutputField(desc="JSON array of {name, type, description, examples} objects")

    # Create the memory-augmented RLM
    rlm = RLMWithLatentBriefing(
        signature=FieldDiscoverySig,
        max_iterations=RLM_MAX_ITERATIONS,
        max_llm_calls=RLM_MAX_LLM_CALLS,
        verbose=False,
        sub_lm=sub_lm,
        briefing_tau=briefing_tau,
        briefing_max_tokens=briefing_max_tokens,
        adaptive_tau=adaptive_tau,
    )

    # Cap chunks (same as standard RLM)
    max_rlm_chunks = 60
    if len(chunk_texts) > max_rlm_chunks:
        import random
        rng = random.Random(42)
        head = chunk_texts[:20]
        tail_pool = chunk_texts[20:]
        sampled = rng.sample(tail_pool, min(max_rlm_chunks - 20, len(tail_pool)))
        capped_texts = head + sampled
    else:
        capped_texts = chunk_texts

    chunks_payload = json.dumps(capped_texts, ensure_ascii=False)

    briefing_meta = {"briefing_enabled": True, "tau": briefing_tau, "adaptive": adaptive_tau}

    try:
        result = rlm(
            chunks_json=chunks_payload,
            cluster_label=cluster_label,
            keywords=", ".join(keywords[:10]),
            max_fields=str(config.max_fields_per_cluster),
        )
        raw = result.fields_json

        # Capture briefing stats
        if hasattr(result, "briefing_stats"):
            briefing_meta.update(result.briefing_stats)

    except Exception as e:
        logger.warning(f"RLM+LB field discovery failed: {e}")
        briefing_meta["error"] = str(e)
        return [], briefing_meta

    # Parse output (same fallback chain as standard RLM)
    if not raw or (isinstance(raw, str) and not raw.strip()):
        logger.warning("RLM+LB returned empty output")
        return [], briefing_meta

    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        import ast
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return [], briefing_meta
            else:
                return [], briefing_meta

    if isinstance(parsed, dict):
        fields = parsed.get("fields", [])
    elif isinstance(parsed, list):
        fields = parsed
    else:
        return [], briefing_meta

    # Normalize
    valid_types = {"text", "date", "duration", "currency", "percentage",
                   "integer", "boolean", "entity_name", "enum"}
    result_fields = []
    for f in fields[:config.max_fields_per_cluster]:
        if not isinstance(f, dict):
            continue
        name = (f.get("name") or f.get("field_name") or f.get("field") or "").strip()
        if not name:
            continue
        name = re.sub(r'[^a-z0-9_]', '_', name.lower())[:40]
        ftype = f.get("type", "text").strip().lower()
        if ftype not in valid_types:
            ftype = "text"
        result_fields.append({
            "name": name,
            "type": ftype,
            "description": f.get("description", ""),
            "examples": f.get("examples") or ([f["field_value"]] if f.get("field_value") else []),
        })

    briefing_meta["fields_discovered"] = len(result_fields)
    return result_fields, briefing_meta


def extract_fields_from_chunk(
    client, deployment: str,
    cluster_label: str, chunk_text: str,
    fields: list[dict], config: FieldDiscoveryConfig,
    rlm_context: dict | None = None,
) -> dict[str, tuple[str | None, float]]:
    """Extract field values from a single chunk. Returns {field_name: (value, confidence)}."""
    # Build enhanced schema with examples when available
    schema_lines = []
    for f in fields:
        line = f"- {f['name']} ({f['type']}): {f['description']}"
        examples = f.get('examples', [])
        if examples:
            ex_str = ', '.join(f'"{e}"' for e in examples[:2] if e)
            if ex_str:
                line += f"  [examples from similar clauses: {ex_str}]"
        schema_lines.append(line)
    fields_schema = "\n".join(schema_lines)

    # Build RLM context section if available
    rlm_context_section = ""
    if rlm_context:
        parts = []
        clause_types = rlm_context.get('clause_types', [])
        if clause_types:
            ct_str = "; ".join(
                f"{ct.get('name', '')}: {ct.get('emergent_description', '')}" 
                for ct in clause_types[:3]
            )
            parts.append(f"Clause sub-types found in similar clauses: {ct_str}")
        obligations = rlm_context.get('obligations', [])
        if obligations:
            ob_str = "; ".join(
                f"{ob.get('obligor', '')} must {ob.get('action', '')}" +
                (f" if {ob['trigger_condition']}" if ob.get('trigger_condition') else "")
                for ob in obligations[:3]
            )
            parts.append(f"Key obligations: {ob_str}")
        intents = rlm_context.get('intents', [])
        if intents:
            int_str = "; ".join(i.get('intent_description', '') for i in intents[:3])
            parts.append(f"Clause intents: {int_str}")
        if parts:
            rlm_context_section = "\n".join(parts) + "\n"

    if rlm_context or any(f.get('examples') for f in fields):
        prompt = FIELD_EXTRACTION_PROMPT_ENHANCED.format(
            cluster_label=cluster_label,
            rlm_context_section=rlm_context_section,
            chunk_text=chunk_text[:4000],
            fields_schema=fields_schema,
        )
    else:
        prompt = FIELD_EXTRACTION_PROMPT.format(
            cluster_label=cluster_label,
            chunk_text=chunk_text[:4000],
            fields_schema=fields_schema,
        )
    parsed = _call_azure(client, deployment, prompt, config, temperature=0.0, max_tokens=1000)
    if not parsed or not isinstance(parsed, dict):
        return {}
    valid_names = {f["name"] for f in fields}
    results = {}
    chunk_lower = chunk_text.lower()
    for k, v in parsed.items():
        if k not in valid_names:
            continue
        if v is None or v == "" or v == "null" or v == "N/A" or v == "n/a":
            continue  # Skip null/empty extractions
        v_str = str(v)
        # Confidence scoring with source-text grounding
        conf = 0.5  # base confidence
        # Grounding: is the value actually present in the source text?
        v_lower = v_str.lower().strip()
        if v_lower in chunk_lower:
            conf += 0.3  # strong grounding — value appears verbatim
        elif len(v_lower) > 5 and any(w in chunk_lower for w in v_lower.split()[:3]):
            conf += 0.15  # partial grounding — some words match
        # Value quality signals
        if len(v_str) > 3:
            conf += 0.05
        if any(c.isdigit() for c in v_str):
            conf += 0.05  # numeric values are typically more reliable
        if len(v_str) > 50:
            conf -= 0.1  # very long values are likely hallucinated passages
        if v_str.startswith("{") or v_str.startswith("["):
            conf -= 0.15  # JSON fragments are extraction errors
        conf = max(0.1, min(1.0, conf))  # clamp
        results[k] = (v_str, round(conf, 3))
    return results


def run_field_discovery(
    store,
    config: FieldDiscoveryConfig | None = None,
    progress_fn=None,
    extract_values: bool = True,
    min_cluster_chunks: int = 5,
    max_extraction_workers: int = 5,
    min_confidence: float = 0.5,
    use_rlm: bool | str = False,
) -> dict:
    """
    Run field discovery across all clusters in the store.

    For each cluster with enough chunks:
      1. Discover fields via Azure OpenAI
      2. Optionally extract values per chunk (parallelized with ThreadPoolExecutor)
      3. Write field_definitions and extractions to the store

    Args:
        max_extraction_workers: Number of parallel Azure OpenAI calls for extraction.
        min_confidence: Minimum confidence threshold to store an extraction.
        use_rlm: False → standard only, True → RLM with standard fallback,
                 "hybrid" → run both Standard + RLM and merge results.
    """
    config = config or FieldDiscoveryConfig()
    progress = progress_fn or (lambda stage, msg: logger.info(f"[{stage}] {msg}"))

    client = _get_azure_client()
    deployment = _get_deployment(config)
    progress("field_discovery", f"Azure OpenAI deployment: {deployment}")

    clusters = store.conn.execute("""
        SELECT c.cluster_id, c.domain_id, c.label, c.keywords, c.chunk_count
        FROM clusters c WHERE c.chunk_count >= ?
        ORDER BY c.chunk_count DESC
    """, [min_cluster_chunks]).fetchdf().to_dict("records")

    progress("field_discovery", f"Discovering fields for {len(clusters)} clusters")

    total_fields = 0
    total_extractions = 0
    skipped_low_confidence = 0

    for ci, cluster in enumerate(clusters):
        cluster_id = cluster["cluster_id"]
        cluster_label = cluster["label"] or "Unknown"
        try:
            keywords = json.loads(cluster.get("keywords", "[]")) if isinstance(cluster.get("keywords"), str) else (cluster.get("keywords") or [])
        except json.JSONDecodeError:
            keywords = []

        chunk_rows = store.conn.execute("""
            SELECT ch.chunk_id, ch.chunk_text, ch.agreement_id
            FROM chunks ch
            JOIN cluster_assignments ca ON ch.chunk_id = ca.chunk_id
            WHERE ca.cluster_id = ?
            ORDER BY ch.chunk_index
        """, [cluster_id]).fetchdf().to_dict("records")

        if len(chunk_rows) < min_cluster_chunks:
            continue

        progress("field_discovery", f"  [{ci+1}/{len(clusters)}] {cluster_label} ({len(chunk_rows)} chunks)")

        # Step 1: Discover fields
        chunk_texts = [r["chunk_text"] for r in chunk_rows]
        hybrid_meta = None
        if use_rlm == "hybrid":
            fields, hybrid_meta = discover_fields_hybrid(
                client, deployment, cluster_label, keywords, chunk_texts, config,
            )
            progress("field_discovery",
                     f"    Hybrid: {hybrid_meta['strategy']} — "
                     f"std={hybrid_meta['standard_count']} ({hybrid_meta['standard_time']}s) "
                     f"rlm={hybrid_meta['rlm_count']} ({hybrid_meta['rlm_time']}s) "
                     f"merged={hybrid_meta['merged_count']} {hybrid_meta['source_breakdown']}")
        elif use_rlm:
            fields = discover_fields_for_cluster_rlm(
                cluster_label, keywords, chunk_texts, config,
            )
            if not fields:
                # Fallback to standard on RLM failure
                logger.info(f"    RLM returned no fields, falling back to standard")
                fields = discover_fields_for_cluster(
                    client, deployment, cluster_label, keywords, chunk_texts, config,
                )
        else:
            fields = discover_fields_for_cluster(
                client, deployment, cluster_label, keywords, chunk_texts, config,
            )
        if not fields:
            progress("field_discovery", f"    No fields discovered")
            continue

        # Persist field definitions
        field_id_map = {}
        for f in fields:
            field_id = f"fld_{uuid.uuid4().hex[:10]}"
            field_id_map[f["name"]] = field_id
            store.upsert_field(
                field_id=field_id, cluster_id=cluster_id,
                name=f["name"], field_type=f["type"],
                description=f["description"], source="auto_discovered",
                examples=f.get("examples", []),
            )

        total_fields += len(fields)
        field_names = [f["name"] for f in fields]
        progress("field_discovery", f"    {len(fields)} fields: {field_names}")

        # Step 2: Extract values (parallelized)
        if extract_values:
            cluster_extractions = 0
            cluster_skipped = 0

            def _extract_one(chunk_row):
                """Extract fields from a single chunk (thread worker)."""
                return chunk_row, extract_fields_from_chunk(
                    client, deployment, cluster_label,
                    chunk_row["chunk_text"], fields, config,
                )

            with ThreadPoolExecutor(max_workers=max_extraction_workers) as pool:
                futures = {pool.submit(_extract_one, cr): cr for cr in chunk_rows}
                for future in as_completed(futures):
                    try:
                        chunk_row, extractions = future.result()
                    except Exception as e:
                        logger.warning(f"Extraction failed: {e}")
                        continue

                    for field_name, (value, confidence) in extractions.items():
                        if confidence < min_confidence:
                            cluster_skipped += 1
                            continue
                        field_id = field_id_map.get(field_name)
                        if not field_id:
                            continue
                        extraction_id = f"ext_{uuid.uuid4().hex[:10]}"
                        store.insert_extraction(
                            extraction_id=extraction_id,
                            agreement_id=chunk_row["agreement_id"],
                            field_id=field_id,
                            value=value,
                            confidence=confidence,
                            source_chunk_id=chunk_row["chunk_id"],
                        )
                        cluster_extractions += 1

            total_extractions += cluster_extractions
            skipped_low_confidence += cluster_skipped
            progress("field_discovery", f"    {cluster_extractions} extractions ({cluster_skipped} below confidence threshold)")

    progress("field_discovery", f"Done: {total_fields} fields, {total_extractions} extractions "
             f"({skipped_low_confidence} skipped low-confidence) across {len(clusters)} clusters")
    return {"total_fields": total_fields, "total_extractions": total_extractions,
            "skipped_low_confidence": skipped_low_confidence}
