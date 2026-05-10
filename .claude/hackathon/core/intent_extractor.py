"""
Dynamic Intent System — discovers and catalogs clause intents automatically.

Instead of forcing every clause into 7 hardcoded boxes, the system:

  1. DISCOVERS intents: asks the LLM "what legal intents are in this clause?"
  2. DECOMPOSES each intent: extracts only the attributes that are relevant
  3. REGISTERS new intent types: builds a growing catalog of intent types
  4. MATCHES against existing types: new intents are matched to the registry
     by embedding similarity — "obligation_to_indemnify" in doc A matches
     the same intent in doc B even if worded differently

A single clause can have 0 intents (Table of Contents) or 5+
(a dense miscellaneous section). Each intent has:
  - label:      what it is ("grant_of_indemnity", "liability_cap")
  - party_from: who bears it
  - party_to:   who benefits
  - attributes: flexible dict of intent-specific properties

The intent registry is the ontology of "what legal effects exist" across
all processed contracts. Two documents are comparable by comparing their
intent fingerprints — which intents are present, which are missing, and
where attribute values deviate.

Usage:
    from core.intent_extractor import run_intent_extraction
    result = run_intent_extraction(store, progress_fn=print)

    # Or extract for a single clause:
    from core.intent_extractor import extract_clause_intents
    intents = extract_clause_intents(client, deployment, clause_text, clause_type, config)
    # Returns: [{"label": "grant_of_indemnity", "party_from": "Vendor", ...}, ...]
"""
import hashlib
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class IntentExtractionConfig:
    max_clause_chars: int = 3000
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 2.0
    model: str = ""
    max_workers: int = 5
    min_clause_chars: int = 80
    match_threshold: float = 0.85   # cosine sim to match a new intent to an existing type


# ── Prompts ──────────────────────────────────────────────────────────────────

INTENT_DISCOVERY_PROMPT = """You are a legal contract analyst. Analyze this clause and identify every distinct legal intent it contains.

A "legal intent" is a single obligation, right, prohibition, condition, or declaration that has legal effect.
One clause may contain 0 intents (e.g., a table of contents) or many (e.g., a miscellaneous section combining
governing law, termination, and arbitration).

Clause type: "{clause_type}"
Clause text:
{clause_text}

For each intent found, provide:
- label: a short snake_case name (e.g., "obligation_to_indemnify", "liability_cap", "governing_law_choice", "auto_renewal_term")
- summary: one sentence describing what this intent does
- party_from: who bears the obligation/restriction (or "none" if declarative)
- party_to: who benefits (or "none" if declarative)
- attributes: a dict of intent-specific properties. Include ONLY properties that are actually stated in the text.
  Examples: {{"cap_amount": "$5M"}}, {{"notice_period": "30 days"}}, {{"jurisdiction": "New York"}},
  {{"survival_period": "24 months"}}, {{"is_mutual": true}}, {{"excludes": "willful misconduct"}}

If the clause has NO legal intent (e.g., table of contents, formatting, definitions list), return an empty array.

Respond with ONLY a JSON object:
{{"intents": [
  {{"label": "...", "summary": "...", "party_from": "...", "party_to": "...", "attributes": {{...}}}},
  ...
]}}"""


def _get_azure_client():
    """Return a chat-completions-compatible client for the configured LLM backend.

    Honors AZURE_OPENAI_*, OPENAI_API_KEY, GEMINI (Vertex), Ollama, LiteLLM. The
    returned object exposes the OpenAI SDK's `.chat.completions.create(...)` shape
    so existing call sites need no changes."""
    from core.llm_client import make_openai_compatible_client
    return make_openai_compatible_client()


def _get_deployment(config: IntentExtractionConfig) -> str:
    if config.model:
        return config.model
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")


def _call_azure(client, deployment: str, prompt: str,
                config: IntentExtractionConfig) -> dict | None:
    """Call Azure OpenAI with retries, return parsed JSON or None."""
    import time
    for attempt in range(config.max_retries):
        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_completion_tokens=1500,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning(f"Intent JSON parse error (attempt {attempt+1})")
        except Exception as e:
            logger.warning(f"Intent Azure API error (attempt {attempt+1}): {e}")
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (attempt + 1))
    return None


def _intent_hash(intent: dict) -> str:
    """Deterministic hash for dedup — based on label + party direction + key attributes."""
    hash_input = json.dumps({
        "label": intent.get("label", ""),
        "party_from": intent.get("party_from", ""),
        "party_to": intent.get("party_to", ""),
        "attributes": intent.get("attributes", {}),
    }, sort_keys=True)
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def _normalize_intent(raw: dict) -> dict | None:
    """Normalize a single intent dict from LLM output. Returns None if invalid."""
    label = raw.get("label", "").strip()
    if not label:
        return None
    # Ensure snake_case
    label = label.lower().replace(" ", "_").replace("-", "_")

    summary = str(raw.get("summary", "")).strip()
    party_from = str(raw.get("party_from", "none")).strip()
    party_to = str(raw.get("party_to", "none")).strip()

    attrs = raw.get("attributes", {})
    if not isinstance(attrs, dict):
        attrs = {}
    # Clean attribute values
    clean_attrs = {}
    for k, v in attrs.items():
        k = str(k).strip().lower().replace(" ", "_")
        if k and v is not None and str(v).strip().lower() not in ("", "none", "null", "n/a"):
            clean_attrs[k] = v
    
    return {
        "label": label,
        "summary": summary,
        "party_from": party_from,
        "party_to": party_to,
        "attributes": clean_attrs,
        "intent_hash": _intent_hash({
            "label": label, "party_from": party_from,
            "party_to": party_to, "attributes": clean_attrs,
        }),
    }


def extract_clause_intents(
    client, deployment: str,
    clause_text: str, clause_type: str,
    config: IntentExtractionConfig,
) -> list[dict]:
    """
    Discover and decompose all intents in a clause.
    
    Returns list of intent dicts, each with:
        label, summary, party_from, party_to, attributes, intent_hash
    
    Returns empty list for clauses with no legal intent.
    """
    text = clause_text[:config.max_clause_chars]
    prompt = INTENT_DISCOVERY_PROMPT.format(
        clause_type=clause_type,
        clause_text=text,
    )
    parsed = _call_azure(client, deployment, prompt, config)
    if not parsed or not isinstance(parsed, dict):
        return []

    raw_intents = parsed.get("intents", [])
    if not isinstance(raw_intents, list):
        return []

    results = []
    for raw in raw_intents:
        if not isinstance(raw, dict):
            continue
        normalized = _normalize_intent(raw)
        if normalized:
            results.append(normalized)
    return results


# ── Intent Registry ──────────────────────────────────────────────────────────

class IntentRegistry:
    """
    A growing catalog of discovered intent types.
    
    When a new intent is discovered (e.g., "obligation_to_indemnify"), it's
    matched against existing types by label similarity and optionally by
    embedding similarity. If a match is found (e.g., "indemnification_obligation"
    already exists), the new intent is assigned to the existing type. Otherwise,
    a new type is registered.
    
    This prevents the registry from exploding with synonymous types while
    still allowing genuine new intents to be cataloged.
    """
    
    def __init__(self, store=None):
        self.store = store
        self._label_cache: dict[str, str] = {}  # label → intent_type_id
        if store:
            self._load_existing()
    
    def _load_existing(self):
        """Load existing intent types into the label cache."""
        types = self.store.get_intent_types()
        for t in types:
            self._label_cache[t["label"]] = t["intent_type_id"]
            # Also cache known aliases
            for alias in json.loads(t.get("aliases", "[]") or "[]"):
                self._label_cache[alias] = t["intent_type_id"]
    
    def match_or_register(self, intent: dict) -> str:
        """
        Match an intent to an existing type or register a new one.
        Returns the intent_type_id.
        """
        label = intent["label"]
        
        # Exact label match (fast path)
        if label in self._label_cache:
            type_id = self._label_cache[label]
            if self.store:
                self.store.increment_intent_type_count(type_id)
            return type_id
        
        # Fuzzy label match: check if any existing label is a substring or vice versa
        for existing_label, type_id in self._label_cache.items():
            if _labels_match(label, existing_label):
                # Register as alias
                self._label_cache[label] = type_id
                if self.store:
                    self.store.add_intent_type_alias(type_id, label)
                    self.store.increment_intent_type_count(type_id)
                return type_id
        
        # No match — register new type
        type_id = f"it_{uuid.uuid4().hex[:10]}"
        self._label_cache[label] = type_id
        if self.store:
            self.store.upsert_intent_type(
                intent_type_id=type_id,
                label=label,
                description=intent.get("summary", ""),
                sample_attributes=list(intent.get("attributes", {}).keys()),
            )
        return type_id


def _labels_match(a: str, b: str) -> bool:
    """
    Check if two snake_case intent labels refer to the same concept.
    
    Handles: word reordering, common synonyms, substring matching.
    E.g., "obligation_to_indemnify" matches "indemnification_obligation"
    """
    if a == b:
        return True
    
    words_a = set(a.split("_"))
    words_b = set(b.split("_"))
    
    # Remove common filler words
    fillers = {"of", "to", "the", "and", "or", "a", "an", "for", "in", "on", "by"}
    core_a = words_a - fillers
    core_b = words_b - fillers
    
    if not core_a or not core_b:
        return False
    
    # If core words overlap significantly (Jaccard > 0.6), it's a match
    overlap = len(core_a & core_b)
    union = len(core_a | core_b)
    if union > 0 and overlap / union >= 0.6:
        return True
    
    # Check common synonyms
    _SYNONYMS = {
        "obligation": {"duty", "requirement", "commitment"},
        "termination": {"cancellation", "end", "expiry", "expiration"},
        "indemnify": {"indemnification", "indemnity", "hold_harmless"},
        "confidentiality": {"nondisclosure", "non_disclosure", "secrecy"},
        "limitation": {"cap", "limit", "ceiling", "restriction"},
        "liability": {"damages", "responsibility"},
        "renewal": {"extension", "auto_renewal", "automatic_renewal"},
        "notice": {"notification", "written_notice"},
        "governing": {"applicable", "choice"},
        "law": {"jurisdiction", "venue"},
        "assignment": {"transfer", "delegation"},
        "warranty": {"representation", "guarantee"},
        "payment": {"compensation", "fee", "pricing"},
    }
    
    # Expand both label word sets with synonyms
    expanded_a = set(core_a)
    expanded_b = set(core_b)
    for word in core_a:
        for key, syns in _SYNONYMS.items():
            if word == key or word in syns:
                expanded_a.add(key)
                expanded_a.update(syns)
    for word in core_b:
        for key, syns in _SYNONYMS.items():
            if word == key or word in syns:
                expanded_b.add(key)
                expanded_b.update(syns)
    
    overlap = len(expanded_a & expanded_b)
    union = len(expanded_a | expanded_b)
    return union > 0 and overlap / union >= 0.4


# ── Main entry point ─────────────────────────────────────────────────────────

def run_intent_extraction(
    store,
    config: IntentExtractionConfig | None = None,
    progress_fn=None,
) -> dict:
    """
    Discover and store intents for all clustered clauses.

    For each clause:
      1. LLM discovers 0..N intents dynamically
      2. Each intent is matched against the intent registry
      3. New intent types are registered; existing ones get incremented
      4. All intents stored in clause_intents with flexible JSON attributes

    Args:
        store: ClusteringStore instance.
        config: Extraction configuration.
        progress_fn: Callback(stage, message).

    Returns:
        {"total_clauses": int, "total_intents": int, "total_failed": int,
         "intent_types_discovered": int}
    """
    config = config or IntentExtractionConfig()
    progress = progress_fn or (lambda stage, msg: logger.info(f"[{stage}] {msg}"))

    client = _get_azure_client()
    deployment = _get_deployment(config)
    registry = IntentRegistry(store)

    # Get clauses not yet intent-extracted
    clauses = store.conn.execute("""
        SELECT cl.clause_id, cl.agreement_id, cl.clause_type_id,
               cl.full_text, c.label as clause_type_label
        FROM clauses cl
        JOIN clusters c ON cl.clause_type_id = c.cluster_id
        WHERE cl.clause_type_id IS NOT NULL
          AND cl.clause_id NOT IN (SELECT DISTINCT clause_id FROM clause_intents)
          AND LENGTH(cl.full_text) >= ?
        ORDER BY c.label, cl.clause_id
    """, [config.min_clause_chars]).fetchdf().to_dict("records")

    if not clauses:
        progress("intent", "No new clauses to extract intents for")
        return {"total_clauses": 0, "total_intents": 0, "total_failed": 0,
                "intent_types_discovered": 0}

    progress("intent", f"Discovering intents for {len(clauses)} clauses "
             f"(deployment={deployment}, workers={config.max_workers})")

    types_before = len(registry._label_cache)
    total_intents = 0
    total_failed = 0
    total_clauses_processed = 0

    def _extract_one(clause_row):
        return clause_row, extract_clause_intents(
            client, deployment,
            clause_row["full_text"],
            clause_row["clause_type_label"],
            config,
        )

    batch_size = 50
    for batch_start in range(0, len(clauses), batch_size):
        batch = clauses[batch_start:batch_start + batch_size]
        batch_intents = 0
        batch_failed = 0

        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = {pool.submit(_extract_one, cr): cr for cr in batch}
            for future in as_completed(futures):
                try:
                    clause_row, intents = future.result()
                except Exception as e:
                    logger.warning(f"Intent extraction failed: {e}")
                    batch_failed += 1
                    continue

                if intents is None:
                    batch_failed += 1
                    continue

                total_clauses_processed += 1

                for intent in intents:
                    intent_type_id = registry.match_or_register(intent)
                    store.insert_clause_intent(
                        intent_id=f"ci_{uuid.uuid4().hex[:10]}",
                        clause_id=clause_row["clause_id"],
                        agreement_id=clause_row["agreement_id"],
                        clause_type_id=clause_row["clause_type_id"],
                        intent_type_id=intent_type_id,
                        intent_label=intent["label"],
                        summary=intent.get("summary", ""),
                        party_from=intent.get("party_from", "none"),
                        party_to=intent.get("party_to", "none"),
                        attributes=intent.get("attributes", {}),
                        intent_hash=intent.get("intent_hash", ""),
                    )
                    batch_intents += 1

        total_intents += batch_intents
        total_failed += batch_failed
        types_now = len(registry._label_cache)
        progress("intent", f"  Batch {batch_start//batch_size + 1}: "
                 f"{batch_intents} intents from {len(batch)} clauses, "
                 f"{batch_failed} failed "
                 f"(registry: {types_now} types, total intents: {total_intents})")

    types_discovered = len(registry._label_cache) - types_before
    progress("intent", f"Done: {total_intents} intents from {total_clauses_processed} clauses, "
             f"{types_discovered} new intent types discovered, "
             f"{total_failed} failed")

    return {
        "total_clauses": total_clauses_processed,
        "total_intents": total_intents,
        "total_failed": total_failed,
        "intent_types_discovered": types_discovered,
    }
