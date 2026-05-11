"""FastAPI routes that emit the data shapes consumed by the bundled
React UX at `ux/unzipped/Playbook Intelligence.html`.

All endpoints under `/api/ui/*` return JSON whose top-level shape matches the
mock `window.*` globals in `ux/unzipped/data.js`. The bundled `data_loader.js`
calls these endpoints on page load and overwrites the mock globals with live
data before React mounts.

Endpoints:
    GET  /api/ui/corpus_meta?playbook_id=...
    GET  /api/ui/clusters?account_id=demo
    GET  /api/ui/rules?playbook_id=...
    GET  /api/ui/contract/{agreement_id}?account_id=demo
    GET  /api/ui/findings?run_id=...
    GET  /api/ui/portfolio?playbook_id=...&limit=24
    GET  /api/ui/severity_palette
    GET  /api/ui/playbooks                       # list available playbooks
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ui", tags=["ui"])


# ─────────────────────────────────────────────────────────────────────────────
# Connection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _db_path() -> str:
    """Resolve which DuckDB the UI should read from.

    Honors `DB_PATH` env (set at deploy time) and falls back to demo.duckdb.
    """
    return os.getenv("DB_PATH", "data/demo.duckdb")


def _pb_store():
    from core.playbooks.store import PlaybookStore
    return PlaybookStore(_db_path())


def _cs_store():
    from core.store import ClusteringStore
    return ClusteringStore(_db_path())


# ─────────────────────────────────────────────────────────────────────────────
# /api/ui/severity_palette — static lookup matching SEV in data.js
# ─────────────────────────────────────────────────────────────────────────────

_SEV_PALETTE = {
    "blocker":            {"label": "Blocker",            "rank": 0, "glyph": "x"},
    "approval_required":  {"label": "Approval required",  "rank": 1, "glyph": "alert"},
    "warn":               {"label": "Warn",               "rank": 2, "glyph": "tri"},
    "info":               {"label": "Info",               "rank": 3, "glyph": "i"},
    "pass":               {"label": "Pass",               "rank": 4, "glyph": "ok"},
    "na":                 {"label": "N/A",                "rank": 5, "glyph": "·"},
}


@router.get("/severity_palette")
def severity_palette() -> dict:
    return _SEV_PALETTE


# ─────────────────────────────────────────────────────────────────────────────
# /api/ui/playbooks — list available playbooks (for picker)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/playbooks")
def list_playbooks() -> dict:
    s = _pb_store()
    try:
        pbs = s.list_playbooks()
        out = []
        for p in pbs:
            n_rules = s.conn.execute(
                "SELECT count(*) FROM playbook_rules WHERE playbook_id=?",
                [p["playbook_id"]],
            ).fetchone()[0]
            out.append({
                "playbook_id": p["playbook_id"],
                "name": p["name"],
                "owner_org": p.get("owner_org"),
                "status": p.get("status"),
                "rule_count": int(n_rules),
                "created_at": str(p.get("created_at") or ""),
            })
        return {"playbooks": out}
    finally:
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
# /api/ui/corpus_meta — top header strip
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/corpus_meta")
def corpus_meta(playbook_id: Optional[str] = None) -> dict:
    """Returns the CORPUS_META object."""
    cs = _cs_store()
    s = _pb_store()
    try:
        # Pick a playbook: explicit > most-rules > first
        pbs = s.list_playbooks()
        pb = None
        if playbook_id:
            pb = next((p for p in pbs if p["playbook_id"] == playbook_id), None)
        if pb is None and pbs:
            # Pick the one with the most rules
            counts = []
            for p in pbs:
                n = s.conn.execute(
                    "SELECT count(*) FROM playbook_rules WHERE playbook_id=?",
                    [p["playbook_id"]],
                ).fetchone()[0]
                counts.append((n, p))
            counts.sort(key=lambda x: -x[0])
            pb = counts[0][1]
        if pb is None:
            return {
                "playbook_id": None, "playbook_label": "No playbook yet",
                "domain": "", "agreement_count": 0,
                "mined_at": None, "drafts_pending": 0,
                "rules_active": 0, "rules_retired": 0,
            }

        # Domain label (first one we can find in clusters/domains)
        try:
            row = cs.conn.execute("SELECT label FROM domains LIMIT 1").fetchone()
            domain = row[0] if row else ""
        except Exception:
            domain = ""

        agreement_count = cs.conn.execute("SELECT count(*) FROM agreements").fetchone()[0]

        # Rule status counts
        rows = s.conn.execute(
            "SELECT status, count(*) FROM playbook_rules WHERE playbook_id=? GROUP BY status",
            [pb["playbook_id"]],
        ).fetchall()
        status_counts = dict(rows)

        return {
            "playbook_id": pb["playbook_id"],
            "playbook_label": pb["name"],
            "domain": domain,
            "agreement_count": int(agreement_count),
            "mined_at": str(pb.get("created_at") or ""),
            "drafts_pending": int(status_counts.get("draft", 0)),
            "rules_active": int(status_counts.get("active", 0)),
            "rules_retired": int(status_counts.get("retired", 0)),
        }
    finally:
        s.close()
        if hasattr(cs, "close"):
            cs.close()


# ─────────────────────────────────────────────────────────────────────────────
# /api/ui/clusters — left rail / Page-1 treemap source
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/clusters")
def clusters() -> dict:
    cs = _cs_store()
    try:
        rows = cs.conn.execute(
            "SELECT cluster_id AS id, label, chunk_count, "
            "       COALESCE(quality_score, 0.0) AS quality "
            "FROM clusters ORDER BY chunk_count DESC"
        ).fetchall()
        # Build the inferred-label cache once per request so we don't query
        # field_definitions inside a loop.
        label_map = _inferred_cluster_labels(cs)
        out = []
        for r in rows:
            raw_label = r[1] or ""
            inferred = label_map.get(r[0])
            display = inferred or raw_label
            out.append({
                "id": r[0],
                "label": display,
                "raw_label": raw_label,
                "inferred": bool(inferred and inferred != raw_label),
                "chunk_count": int(r[2] or 0),
                "quality": float(r[3] or 0.0),
            })
        return {"clusters": out}
    finally:
        if hasattr(cs, "close"):
            cs.close()


# ─────────────────────────────────────────────────────────────────────────────
# Cluster-label inference
#
# The clustering pipeline emits placeholder labels like "Clause_Type_7". For
# display purposes we infer a meaningful name from the cluster's discovered
# fields. The mapping is keyword-driven — fast, no LLM call needed — and
# falls back to title-casing the first 1-2 field names when no theme matches.
# ─────────────────────────────────────────────────────────────────────────────

# Ordered most-specific-first so we don't fall back to "Term" when both
# "termination" and "term" keywords are present.
_LABEL_THEMES = [
    ("Confidentiality",   ("confidential", "nda", "non_disclosure", "trade_secret")),
    ("Intellectual Property", ("intellectual_property", "ip_ownership", "assignment_of_inventions", "patent", "copyright", "license")),
    ("Indemnification",   ("indemnif", "hold_harmless", "defense_obligation")),
    ("Limitation of Liability", ("liability_cap", "limitation_of_liability", "consequential_damages")),
    ("Termination",       ("termination", "terminate_for_cause", "terminate_for_convenience")),
    ("Auto-Renewal",      ("auto_renewal", "renewal_term", "non_renewal")),
    ("Governing Law & Jurisdiction", ("governing_law", "jurisdiction", "venue", "choice_of_law")),
    ("Arbitration",       ("arbitration", "arbitrator", "dispute_resolution", "jams", "aaa")),
    ("Insurance",         ("insurance", "coverage_amount", "policy_limit")),
    ("Compensation",      ("consulting_fee", "compensation", "monthly_fee", "hourly_rate", "retainer", "vesting", "bonus")),
    ("Payment Terms",     ("payment_terms", "payment_deadline", "invoice", "net_days", "late_payment")),
    ("Non-Compete & Non-Solicitation", ("non_compete", "non_solicit", "restricted_period")),
    ("Compliance & Audit", ("compliance", "audit", "anti_corruption", "anti_bribery", "fcpa")),
    ("Change of Control", ("change_of_control", "merger", "acquisition")),
    ("Force Majeure",     ("force_majeure", "act_of_god")),
    ("Notice & Communications", ("notice_period", "notice_address", "notification")),
    ("Representations & Warranties", ("representation", "warranty", "warranties")),
    ("Party Identification", ("consultant_name", "company_name", "client_name", "vendor_name", "party_name")),
    ("Term & Effective Date", ("effective_date", "expiration_date", "initial_term", "term_duration")),
    ("Definitions",       ("defined_term", "definition", "regulatory_authority")),
]

# Pretty noun for a field name — strips trailing _days/_amount/_pct/_id etc.
_FIELD_NAME_STRIP = ("_days", "_amount", "_pct", "_percent", "_id", "_text",
                      "_status", "_type", "_value", "_date", "_name", "_period",
                      "_threshold", "_count", "_duration", "_months", "_years")


def _title_field(name: str) -> str:
    s = (name or "").strip()
    for suf in _FIELD_NAME_STRIP:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return s.replace("_", " ").strip().title() or "Misc"


def _infer_label_from_fields(field_names: list[str], raw_label: str) -> str:
    """Pick a human-readable label from a cluster's field names.

    Strategy:
      1. Score each theme by how many of its keywords appear as substrings
         in the lowercased, underscore-preserved field names.
      2. Pick the highest-scoring theme; if tied, the first wins (order is
         most-specific first).
      3. If no theme has any hits, fall back to title-casing the first 1-2
         field names ("Compensation, Vesting").
      4. If no field names at all, keep raw_label.
    """
    if not field_names:
        return raw_label
    bag = " ".join(f.lower() for f in field_names if f)
    best = (0, None)
    for theme_name, kws in _LABEL_THEMES:
        score = sum(1 for kw in kws if kw in bag)
        if score > best[0]:
            best = (score, theme_name)
    if best[1]:
        return best[1]
    # Fallback: join first 1–2 normalized field names
    nouns = []
    seen = set()
    for f in field_names[:4]:
        n = _title_field(f)
        if n and n not in seen:
            seen.add(n)
            nouns.append(n)
        if len(nouns) == 2:
            break
    return " & ".join(nouns) if nouns else raw_label


# Module-level cache so Gemini is called at most once per cluster_id per process.
_LLM_LABEL_CACHE: dict[str, str] = {}


def _llm_label_from_samples(cluster_id: str, samples: list[str]) -> Optional[str]:
    """Best-effort: ask Gemini for a 1–3 word legal clause label given a few
    example clauses. Returns None when Gemini is not configured or any error
    occurs. Caches successful results module-level."""
    if cluster_id in _LLM_LABEL_CACHE:
        return _LLM_LABEL_CACHE[cluster_id]
    if not samples:
        return None
    backend = os.getenv("LLM_BACKEND", "openai").lower()
    if backend == "gemini" and not (os.getenv("GOOGLE_CLOUD_PROJECT") and os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
        return None
    try:
        from core.llm_client import LLMClient
        client = LLMClient()
    except Exception:
        return None

    excerpts = "\n\n".join(
        f"({i + 1}) {s.strip()[:400]}" for i, s in enumerate(samples[:4])
    )
    prompt = (
        "Label this legal clause type in 1 to 3 words (Title Case). Examples of "
        "good labels: Indemnification, Limitation of Liability, Governing Law, "
        "Termination, Non-Compete, Notice & Communications, Confidentiality. "
        "Return ONLY the label, no other text.\n\n"
        f"Sample clauses from one cluster:\n{excerpts}"
    )
    try:
        text = client.complete(prompt, system="You are a precise legal clause taxonomist.",
                                  temperature=0.0, max_tokens=20)
    except Exception as e:
        logger.warning(f"LLM label call failed for {cluster_id}: {e}")
        return None
    label = (text or "").strip().strip("\"'`")
    # Reject empty / overly long / obviously wrong responses
    if not label or len(label) > 60 or "\n" in label:
        return None
    _LLM_LABEL_CACHE[cluster_id] = label
    return label


def _inferred_cluster_labels(cs) -> dict[str, str]:
    """Returns {cluster_id: inferred_label} for every cluster.

    Order of precedence per cluster:
      1. Real label (anything not matching the Clause_Type_N placeholder) → pass through.
      2. Theme-keyword match against the cluster's discovered field names.
      3. Title-cased fallback joining the first 1-2 normalized field names.
      4. Gemini call seeded with 3-4 sample clauses from the cluster
         (only when no fields exist; cached module-level).
      5. Raw placeholder if all of the above failed.
    """
    out: dict[str, str] = {}
    try:
        rows = cs.conn.execute(
            "SELECT c.cluster_id, c.label, "
            "       string_agg(f.name, '||') AS field_names "
            "FROM clusters c "
            "LEFT JOIN field_definitions f ON f.cluster_id = c.cluster_id "
            "GROUP BY c.cluster_id, c.label"
        ).fetchall()
    except Exception:
        return out
    placeholder = re.compile(r"^(Clause_Type_\d+|Cluster_\d+|cluster_\d+|topic_\d+)$")

    for cid, raw, fnames_blob in rows:
        if not cid:
            continue
        is_placeholder = bool(raw and placeholder.match(str(raw)))
        if not is_placeholder:
            out[cid] = raw
            continue
        fnames = [s for s in (fnames_blob or "").split("||") if s]
        if fnames:
            out[cid] = _infer_label_from_fields(fnames, raw)
            continue
        # No fields — try LLM (cheap, cached).
        samples: list[str] = []
        try:
            sample_rows = cs.conn.execute(
                "SELECT full_text FROM clauses WHERE clause_type_id = ? "
                "AND full_text IS NOT NULL LIMIT 4",
                [cid],
            ).fetchall()
            samples = [r[0] for r in sample_rows if r and r[0]]
        except Exception:
            samples = []
        llm_label = _llm_label_from_samples(cid, samples) if samples else None
        out[cid] = llm_label or raw
    return out


# ─────────────────────────────────────────────────────────────────────────────
# /api/ui/rules — Playbook Studio queue (mocks RULES array)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/rules")
def rules(playbook_id: Optional[str] = None,
           include_examples: bool = Query(True, description="Compute pass/fail examples"),
           max_examples: int = Query(3, ge=0, le=10)) -> dict:
    s = _pb_store()
    cs = _cs_store()
    try:
        # Default to the most-populated playbook if none specified
        pbs = s.list_playbooks()
        if playbook_id is None:
            if not pbs:
                return {"rules": []}
            counts = [(s.conn.execute(
                "SELECT count(*) FROM playbook_rules WHERE playbook_id=?",
                [p["playbook_id"]],
            ).fetchone()[0], p) for p in pbs]
            counts.sort(key=lambda x: -x[0])
            playbook_id = counts[0][1]["playbook_id"]

        rule_rows = s.list_rules(playbook_id)
        # cluster_id → inferred human label.
        cluster_label_map = _inferred_cluster_labels(cs)
        # raw label → inferred label (used for predicate-string substitution).
        raw_label_map: dict[str, str] = {}
        for r in cs.conn.execute("SELECT cluster_id, label FROM clusters").fetchall():
            cid, raw = r[0], r[1]
            if cid and raw:
                inf = cluster_label_map.get(cid)
                if inf:
                    raw_label_map[raw] = inf
        # field_id → cluster_id, and field_name → cluster_id (for field-scope rules)
        fname_to_cluster: dict[str, str] = {}
        for fid, fname, cid in cs.conn.execute(
            "SELECT field_id, name, cluster_id FROM field_definitions"
        ).fetchall():
            if fname and cid:
                fname_to_cluster[fname] = cid

        domain_row = cs.conn.execute(
            "SELECT label FROM domains ORDER BY agreement_count DESC LIMIT 1"
        ).fetchone()
        domain_label = (domain_row[0] if domain_row else "All Documents") or "All Documents"

        out = []
        for r in rule_rows:
            rid = r["rule_id"]
            bindings = s.bindings_for(rid)
            applies_to = r["applies_to"]
            prov = r.get("source_provenance") or {}

            # Field name (if applicable) for distribution/categorical rules
            field_name = None
            if applies_to == "field" and isinstance(r.get("predicate"), dict):
                args = r["predicate"].get("args", [])
                if args and isinstance(args[0], str):
                    field_name = args[0]

            # ── Resolve target cluster (id + inferred label).
            target_cluster_id = None
            target_cluster_label = None
            if applies_to == "cluster":
                # Bound cluster
                for b in bindings:
                    if b["entity_kind"] == "cluster":
                        target_cluster_id = b["entity_id"]
                        break
            elif applies_to == "field" and field_name:
                target_cluster_id = fname_to_cluster.get(field_name)
            elif applies_to in ("domain", "document"):
                # Parse predicate: any_of[clause.classified_as("ClusterName")]
                target_raw_label = _extract_target_cluster_label_from_predicate(r.get("predicate"))
                if target_raw_label:
                    target_cluster_label = raw_label_map.get(target_raw_label, target_raw_label)
                    # Find cluster_id by raw label
                    row = cs.conn.execute(
                        "SELECT cluster_id FROM clusters WHERE label = ? LIMIT 1",
                        [target_raw_label],
                    ).fetchone()
                    if row:
                        target_cluster_id = row[0]
            elif applies_to in ("composite", "cross_field"):
                # Try first cluster binding
                for b in bindings:
                    if b["entity_kind"] == "cluster":
                        target_cluster_id = b["entity_id"]
                        break

            if target_cluster_id and not target_cluster_label:
                target_cluster_label = cluster_label_map.get(target_cluster_id)

            # ── Rewrite title using raw → inferred label map + domain label
            display_title = _rewrite_title_with_label(
                r["title"], raw_label_map, domain_label,
            )

            # Group key for the UI hierarchy.
            if applies_to in ("domain", "document"):
                group_kind = "domain"
                group_id = "__domain__"
                group_label = domain_label
            else:
                group_kind = "cluster"
                group_id = target_cluster_id or "__unbound__"
                group_label = target_cluster_label or "Unbound"

            # Derivation: a short structured trail showing WHERE this rule
            # came from. Frontend renders this as a "X clauses · Y fields · Z%
            # coverage" chip line under the rule title.
            derivation = _build_derivation(prov, applies_to, field_name, target_cluster_label, cs)

            entry = {
                "id": rid,
                "title": display_title,
                "title_raw": r["title"],
                "description": r.get("description") or "",
                "applies_to": applies_to,
                "field": field_name,
                "cluster_label": target_cluster_label,
                "cluster_id": target_cluster_id,
                "group_kind": group_kind,    # "domain" | "cluster"
                "group_id": group_id,
                "group_label": group_label,
                "predicate": r.get("predicate"),
                "severity": r["severity"],
                "answer_type": r.get("answer_type"),
                "escalation_owner": r.get("escalation_owner"),
                "status": r.get("status", "draft"),
                "confidence": _confidence_for_rule(r, bindings),
                "source": _shape_source(prov),
                "derivation": derivation,
                "preferred_language": r.get("preferred_language"),
                "walkaway_language": r.get("walkaway_language"),
                "rationale": r.get("rationale"),
                "lift": prov.get("lift") or prov.get("cluster_lift") or prov.get("lift_ratio"),
                "support": prov.get("ratio") or prov.get("frequency") or prov.get("support"),
            }

            if include_examples:
                entry["examples"] = _compute_examples(
                    cs, rule=r, field_name=field_name,
                    cluster_id=target_cluster_id, max_per_side=max_examples,
                )
            else:
                entry["examples"] = {"pass": [], "fail": []}

            out.append(entry)
        return {"rules": out, "playbook_id": playbook_id}
    finally:
        s.close()
        if hasattr(cs, "close"):
            cs.close()


def _shape_source(prov: dict) -> dict:
    """Normalize source_provenance into the UI's expected shape."""
    out = dict(prov)
    # The UX expects miner = one of {coverage, distribution, categorical,
    # outlier, contrastive} or migrated_from = "...".
    if "miner" not in out and "migrated_from" in out:
        # Keep migrated_from key; UI handles it.
        pass
    return out


def _confidence_for_rule(rule: dict, bindings: list[dict]) -> float:
    """Derive a 0..1 confidence for the rule from the strongest binding."""
    if not bindings:
        return 0.5
    return float(max(b.get("confidence", 0.0) or 0.0 for b in bindings))


_PLACEHOLDER_RE = re.compile(r"\b(Clause_Type_\d+|Cluster_\d+)\b", re.IGNORECASE)


def _rewrite_title_with_label(title: str,
                                raw_label_map: dict[str, str],
                                domain_label: str) -> str:
    """Replace each Clause_Type_N placeholder in a rule title with its
    inferred human label, looked up by exact raw match. Also rewrites the
    noisy 'ALL DOCUMENTS' prefix to the actual domain label."""
    out = title or ""
    def _sub(m: re.Match) -> str:
        raw = m.group(0)
        return raw_label_map.get(raw, raw)
    out = _PLACEHOLDER_RE.sub(_sub, out)
    out = out.replace("ALL DOCUMENTS", domain_label)
    return out


def _extract_target_cluster_label_from_predicate(predicate) -> Optional[str]:
    """Recursively walk a predicate JSON tree and return the FIRST raw cluster
    label found inside a `clause.classified_as` arg, if any. Used to
    determine which cluster a `domain`-scoped rule actually targets."""
    if not isinstance(predicate, dict):
        return None
    if predicate.get("op") == "clause.classified_as":
        args = predicate.get("args") or []
        if args and isinstance(args[0], str):
            return args[0]
    for child in (predicate.get("args") or []):
        found = _extract_target_cluster_label_from_predicate(child)
        if found:
            return found
    return None


def _build_derivation(prov: dict, applies_to: str,
                       field_name: Optional[str],
                       cluster_label: Optional[str],
                       cs) -> dict:
    """Return a structured derivation summary the UI can render directly.

    Shape:
      {
        "miner_kind": "coverage" | "distribution" | "categorical" | "outlier"
                       | "contrastive" | "migrated" | "unknown",
        "label": "...short human descriptor",   # one-liner
        "stats": [{"key":"n","value":"5"}, ...],
        "trail": ["cluster: Confidentiality", "field: confidentiality_duration"]
      }
    """
    miner = (prov or {}).get("miner") or (
        "migrated" if (prov or {}).get("migrated_from") else "unknown"
    )
    stats: list[dict] = []
    trail: list[str] = []
    label = ""

    if cluster_label:
        trail.append(f"cluster: {cluster_label}")
    if field_name:
        trail.append(f"field: {field_name}")

    n = prov.get("n")

    if miner == "coverage":
        ratio = prov.get("ratio")
        if isinstance(ratio, (int, float)):
            stats.append({"key": "coverage", "value": f"{ratio:.0%}"})
        if n is not None:
            stats.append({"key": "n", "value": str(n)})
        label = (
            f"{int(ratio * 100) if isinstance(ratio,(int,float)) else '?'}% of "
            f"agreements contain this clause" if ratio else "Coverage rule"
        )

    elif miner == "distribution":
        lo, hi, med = prov.get("lo"), prov.get("hi"), prov.get("median")
        if med is not None:
            stats.append({"key": "median", "value": _num(med)})
        if lo is not None and hi is not None:
            stats.append({"key": "range", "value": f"{_num(lo)}–{_num(hi)}"})
        if n is not None:
            stats.append({"key": "n", "value": str(n)})
        label = "Derived from corpus value distribution (p10..p90)."

    elif miner == "categorical":
        mode = prov.get("mode")
        freq = prov.get("frequency")
        if mode is not None:
            stats.append({"key": "mode", "value": str(mode)[:32]})
        if isinstance(freq, (int, float)):
            stats.append({"key": "freq", "value": f"{freq:.0%}"})
        if n is not None:
            stats.append({"key": "n", "value": str(n)})
        label = "Derived from dominant categorical value."

    elif miner == "outlier":
        out_pct = prov.get("outlier_pct")
        if isinstance(out_pct, (int, float)):
            stats.append({"key": "outlier", "value": f"{out_pct:.0%}"})
        if n is not None:
            stats.append({"key": "n", "value": str(n)})
        label = "Derived from cluster-centroid outlier scoring."

    elif miner == "contrastive":
        cl = prov.get("cluster_lift")
        gl = prov.get("global_lift")
        lr = prov.get("lift_ratio")
        if isinstance(lr, (int, float)):
            stats.append({"key": "lift×", "value": f"{lr:.1f}"})
        if isinstance(cl, (int, float)):
            stats.append({"key": "cluster_lift", "value": f"{cl:.1f}"})
        if isinstance(gl, (int, float)):
            stats.append({"key": "global_lift", "value": f"{gl:.1f}"})
        label = "Contrastive — pattern unique to this cluster vs corpus baseline."

    elif miner == "migrated":
        stats.append({"key": "source", "value": "legacy benchmark"})
        label = "Migrated from prior similarity-only benchmark."

    else:
        if n is not None:
            stats.append({"key": "n", "value": str(n)})
        label = "Derived rule."

    return {
        "miner_kind": miner,
        "label": label,
        "stats": stats,
        "trail": trail,
    }


def _num(v) -> str:
    """Compact numeric formatter for derivation stats."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f.is_integer():
        return str(int(f))
    return f"{f:.2f}".rstrip("0").rstrip(".")


def _compute_examples(cs, *, rule: dict, field_name: Optional[str],
                       cluster_id: Optional[str], max_per_side: int = 3) -> dict:
    """For each rule, return up to N pass and N fail examples drawn from the
    corpus. Best-effort — degrades gracefully when the rule shape is unusual."""
    pass_ex: list[dict] = []
    fail_ex: list[dict] = []
    applies_to = rule.get("applies_to")
    predicate = rule.get("predicate") or {}
    op = predicate.get("op") if isinstance(predicate, dict) else None
    args = predicate.get("args") if isinstance(predicate, dict) else None

    try:
        if applies_to == "field" and field_name:
            # Find extractions of this field; check predicate per row.
            rows = cs.conn.execute(
                "SELECT e.agreement_id, e.value, c.full_text "
                "FROM extractions e "
                "JOIN field_definitions f ON e.field_id = f.field_id "
                "LEFT JOIN clauses c ON c.agreement_id = e.agreement_id AND c.clause_type_id = f.cluster_id "
                "WHERE f.name = ? LIMIT 200",
                [field_name],
            ).fetchall()
            for aid, val, full_text in rows:
                rec = {
                    "agreement": aid,
                    "snippet": (full_text or "")[:200],
                    "value": str(val) if val is not None else None,
                }
                if _predicate_matches_field(op, args, val):
                    if len(pass_ex) < max_per_side:
                        pass_ex.append(rec)
                else:
                    if len(fail_ex) < max_per_side:
                        fail_ex.append(rec)
                if len(pass_ex) >= max_per_side and len(fail_ex) >= max_per_side:
                    break

        elif applies_to in ("domain", "cluster"):
            # Coverage rules: resolve a cluster_id from the binding OR from
            # the predicate's `clause.classified_as` arg.
            target_cluster_id = cluster_id
            if not target_cluster_id and op in ("any_of", "all_of") and isinstance(args, list):
                # Drill into nested ops to find the cluster label
                for child in args:
                    if isinstance(child, dict) and child.get("op") == "clause.classified_as":
                        c_args = child.get("args") or []
                        if c_args:
                            label = c_args[0]
                            row = cs.conn.execute(
                                "SELECT cluster_id FROM clusters WHERE label = ?",
                                [label],
                            ).fetchone()
                            if row:
                                target_cluster_id = row[0]
                            break
            elif not target_cluster_id and op == "clause.classified_as" and isinstance(args, list) and args:
                row = cs.conn.execute(
                    "SELECT cluster_id FROM clusters WHERE label = ?",
                    [args[0]],
                ).fetchone()
                if row:
                    target_cluster_id = row[0]

            if target_cluster_id:
                rows = cs.conn.execute(
                    "SELECT agreement_id, full_text FROM clauses "
                    "WHERE clause_type_id = ? LIMIT ?",
                    [target_cluster_id, max_per_side * 2],
                ).fetchall()
                for aid, full_text in rows:
                    if len(pass_ex) < max_per_side:
                        pass_ex.append({
                            "agreement": aid,
                            "snippet": (full_text or "")[:200],
                        })
                    else:
                        break
    except Exception as e:
        logger.warning(f"example computation failed for rule {rule.get('rule_id')}: {e}")
    return {"pass": pass_ex, "fail": fail_ex}


def _predicate_matches_field(op: Optional[str], args: Optional[list],
                                value) -> bool:
    """Evaluate the rule's predicate against a single extracted value."""
    if value is None or op is None or not args:
        return False
    try:
        if op == "field.eq" and len(args) >= 2:
            return str(value).strip().lower() == str(args[1]).strip().lower()
        if op == "field.between" and len(args) >= 3:
            try:
                v = float(str(value).replace("$", "").replace(",", "").split()[0])
                return float(args[1]) <= v <= float(args[2])
            except (ValueError, IndexError):
                return False
        if op == "field.exists":
            return value is not None and str(value).strip() != ""
        if op == "field.gte" and len(args) >= 2:
            try:
                return float(value) >= float(args[1])
            except (TypeError, ValueError):
                return False
        if op == "field.lte" and len(args) >= 2:
            try:
                return float(value) <= float(args[1])
            except (TypeError, ValueError):
                return False
    except Exception:
        return False
    return False


# ─────────────────────────────────────────────────────────────────────────────
# /api/ui/contract/{agreement_id} — ACTIVE_CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/contract/{agreement_id}")
def contract(agreement_id: str) -> dict:
    cs = _cs_store()
    try:
        row = cs.conn.execute(
            "SELECT agreement_id, filename, domain_id "
            "FROM agreements WHERE agreement_id = ?",
            [agreement_id],
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="agreement not found")
        aid, filename, domain_id = row

        # governing_law: best-effort lookup from extractions
        gl = None
        try:
            r = cs.conn.execute(
                "SELECT e.value FROM extractions e "
                "JOIN field_definitions f ON e.field_id = f.field_id "
                "WHERE e.agreement_id = ? AND lower(f.name) IN ('governing_law', 'jurisdiction_state') "
                "ORDER BY e.confidence DESC LIMIT 1",
                [aid],
            ).fetchone()
            if r:
                gl = r[0]
        except Exception:
            pass

        # Counts
        n_clauses = cs.conn.execute(
            "SELECT count(*) FROM clauses WHERE agreement_id = ?", [aid]
        ).fetchone()[0]
        n_chunks = cs.conn.execute(
            "SELECT count(*) FROM chunks WHERE agreement_id = ?", [aid]
        ).fetchone()[0]

        return {
            "id": aid,
            "name": filename or aid,
            "counterparty": None,
            "uploaded_at": None,
            "pages": None,
            "word_count": int(n_chunks * 256) if n_chunks else None,  # rough estimate
            "governing_law": gl,
            "aligned_at": None,
            "duration_ms": None,
            "n_clauses": int(n_clauses),
            "n_chunks": int(n_chunks),
        }
    finally:
        if hasattr(cs, "close"):
            cs.close()


# ─────────────────────────────────────────────────────────────────────────────
# /api/ui/findings — FINDINGS array
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/findings")
def findings(agreement_id: Optional[str] = None,
              run_id: Optional[str] = None,
              playbook_id: Optional[str] = None) -> dict:
    """Returns findings. Either provide a run_id (specific alignment run) OR an
    agreement_id (gets the most recent run for that agreement)."""
    s = _pb_store()
    cs = _cs_store()
    try:
        # Resolve run_id from agreement_id if not given
        if run_id is None and agreement_id:
            row = s.conn.execute(
                "SELECT run_id FROM rule_evaluations WHERE agreement_id=? "
                "ORDER BY evaluated_at DESC LIMIT 1",
                [agreement_id],
            ).fetchone()
            if row:
                run_id = row[0]

        if not run_id:
            return {"findings": [], "run_id": None, "agreement_id": agreement_id}

        rows = s.findings_for_run(run_id)
        # Enrich each finding with rule title, clause label, evidence
        rule_meta = {}
        rule_ids = list({r["rule_id"] for r in rows})
        if rule_ids:
            placeholders = ",".join(["?"] * len(rule_ids))
            for rule_id, title, applies_to, severity, escalation_owner, preferred in s.conn.execute(
                f"SELECT rule_id, title, applies_to, severity, escalation_owner, preferred_language "
                f"FROM playbook_rules WHERE rule_id IN ({placeholders})",
                rule_ids,
            ).fetchall():
                rule_meta[rule_id] = {
                    "title": title, "applies_to": applies_to,
                    "severity": severity, "escalation_owner": escalation_owner,
                    "preferred_language": preferred,
                }

        out = []
        for f in rows:
            evidence = f.get("evidence") or []
            ev_text = ""
            location = None
            if evidence:
                ev0 = evidence[0]
                ev_text = ev0.get("text") or ""
                location = {
                    "page": ev0.get("page"),
                    "section": ev0.get("section"),
                    "line": ev0.get("line"),
                }
            meta = rule_meta.get(f["rule_id"], {})
            outcome = f["outcome"]
            status = "pass" if outcome == "pass" else "open" if outcome == "fail" else "needs_human"

            out.append({
                "id": f["eval_id"],
                "rule_id": f["rule_id"],
                "severity": f["severity"],
                "status": status,
                "title": meta.get("title", f"Rule {f['rule_id'][:8]}"),
                "rule_title": meta.get("title", ""),
                "clause_label": "",
                "location": location,
                "evidence": ev_text,
                "suggestion": f.get("rationale"),
                "preferred_language": meta.get("preferred_language"),
                "escalation_owner": meta.get("escalation_owner"),
                "deviation": float(f.get("deviation") or 0.0),
            })
        return {"findings": out, "run_id": run_id, "agreement_id": agreement_id}
    finally:
        s.close()
        if hasattr(cs, "close"):
            cs.close()


# ─────────────────────────────────────────────────────────────────────────────
# /api/ui/portfolio — N contracts × R rules heatmap
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/portfolio")
def portfolio(playbook_id: Optional[str] = None,
                limit: int = Query(24, ge=1, le=200)) -> dict:
    """Returns the PORTFOLIO array."""
    s = _pb_store()
    cs = _cs_store()
    try:
        # Get top N agreements with the most recent evaluations
        rows = cs.conn.execute(
            f"SELECT agreement_id, filename FROM agreements LIMIT {int(limit)}"
        ).fetchall()
        if not rows:
            return {"portfolio": []}

        # Default to the most-populated playbook (matches corpus_meta logic).
        if playbook_id is None:
            pbs = s.list_playbooks()
            if pbs:
                counts = [(s.conn.execute(
                    "SELECT count(*) FROM playbook_rules WHERE playbook_id=?",
                    [p["playbook_id"]],
                ).fetchone()[0], p) for p in pbs]
                counts.sort(key=lambda x: -x[0])
                playbook_id = counts[0][1]["playbook_id"]
        active_rules: list[str] = []
        if playbook_id:
            active_rules = [
                r["rule_id"] for r in s.list_rules(playbook_id)
                if r["status"] in ("active", "draft")
            ][:8]

        out = []
        for aid, filename in rows:
            cells: list[dict] = []
            for rid in active_rules:
                ev = s.conn.execute(
                    "SELECT outcome, severity FROM rule_evaluations "
                    "WHERE agreement_id=? AND rule_id=? ORDER BY evaluated_at DESC LIMIT 1",
                    [aid, rid],
                ).fetchone()
                if ev:
                    outcome, sev = ev
                    cells.append({
                        "rule_id": rid,
                        "severity": "pass" if outcome == "pass" else sev,
                    })
                else:
                    cells.append({"rule_id": rid, "severity": "na"})
            out.append({
                "id": aid,
                "name": (filename or aid)[:24],
                "cells": cells,
            })
        return {"portfolio": out, "playbook_id": playbook_id,
                 "rule_ids": active_rules}
    finally:
        s.close()
        if hasattr(cs, "close"):
            cs.close()


# ─────────────────────────────────────────────────────────────────────────────
# /api/ui/healthz
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "db_path": _db_path()}
