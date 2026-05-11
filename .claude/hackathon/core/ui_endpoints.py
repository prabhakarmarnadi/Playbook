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
        return {"clusters": [
            {"id": r[0], "label": r[1], "chunk_count": int(r[2] or 0),
             "quality": float(r[3] or 0.0)}
            for r in rows
        ]}
    finally:
        if hasattr(cs, "close"):
            cs.close()


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
        cluster_label_map = {
            r[0]: r[1]
            for r in cs.conn.execute("SELECT cluster_id, label FROM clusters").fetchall()
        }

        out = []
        for r in rule_rows:
            rid = r["rule_id"]
            bindings = s.bindings_for(rid)
            # Pull a cluster label if any binding is cluster-kind
            cluster_id = None
            cluster_label = None
            for b in bindings:
                if b["entity_kind"] == "cluster":
                    cluster_id = b["entity_id"]
                    cluster_label = cluster_label_map.get(cluster_id) or b.get("label_text")
                    break
            if cluster_label is None and bindings:
                cluster_label = bindings[0].get("label_text")

            prov = r.get("source_provenance") or {}
            # Field name (if applicable) for distribution/categorical rules
            field_name = None
            if r.get("applies_to") == "field" and isinstance(r.get("predicate"), dict):
                args = r["predicate"].get("args", [])
                if args:
                    field_name = args[0] if isinstance(args[0], str) else None

            entry = {
                "id": rid,
                "title": r["title"],
                "description": r.get("description") or "",
                "applies_to": r["applies_to"],
                "field": field_name,
                "cluster_label": cluster_label,
                "cluster_id": cluster_id,
                "predicate": r.get("predicate"),
                "severity": r["severity"],
                "answer_type": r.get("answer_type"),
                "escalation_owner": r.get("escalation_owner"),
                "status": r.get("status", "draft"),
                "confidence": _confidence_for_rule(r, bindings),
                "source": _shape_source(prov),
                "preferred_language": r.get("preferred_language"),
                "walkaway_language": r.get("walkaway_language"),
                "rationale": r.get("rationale"),
                "lift": prov.get("lift") or prov.get("cluster_lift") or prov.get("lift_ratio"),
                "support": prov.get("ratio") or prov.get("frequency") or prov.get("support"),
            }

            if include_examples:
                entry["examples"] = _compute_examples(
                    cs, rule=r, field_name=field_name,
                    cluster_id=cluster_id, max_per_side=max_examples,
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
