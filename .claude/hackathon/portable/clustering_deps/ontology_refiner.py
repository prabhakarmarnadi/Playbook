"""
Ontology Refinement Engine — Tier 1 & Tier 2 structural improvements.

Phase 1 (Tier 1 — Structural Stabilization):
  1. Outlier Assignment     — embed & KNN-assign the 43.9% unclustered clauses
  2. Field Deduplication    — merge semantically redundant fields within/across clusters
  3. Cluster Merging        — consolidate clusters with >80% field overlap
  4. Contextual Relabeling  — LLM-driven domain-aware cluster labels

Phase 2 (Tier 2 — Feature Activation):
  5. Feedback Loop          — user corrections retrain field definitions
  6. Composite Fields       — arithmetic/logic relationships between fields

Usage:
    from core.ontology_refiner import OntologyRefiner
    refiner = OntologyRefiner(store, embedder)

    # Phase 1
    refiner.assign_outliers(min_similarity=0.35)
    refiner.deduplicate_fields(similarity_threshold=0.85)
    refiner.merge_overlapping_clusters(overlap_threshold=0.80)
    refiner.synthesize_labels()

    # Phase 2
    refiner.register_correction(clause_id, field_id, old_val, new_val)
    refiner.define_composite("last_opt_out", "expiration_date - notice_period")
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Structural Stabilization
# ═══════════════════════════════════════════════════════════════════════════

class OntologyRefiner:
    """
    Transforms the raw clustering output into a refined, functional ontology.
    """

    def __init__(self, store, embedder=None):
        self.store = store
        self.embedder = embedder
        self._centroids: dict[str, np.ndarray] = {}
        self._centroid_labels: dict[str, str] = {}
        self._load_centroids()

    def _load_centroids(self):
        """Load cluster centroids from DB."""
        rows = self.store.conn.execute("""
            SELECT cc.cluster_id, cc.centroid, c.label
            FROM cluster_centroids cc
            JOIN clusters c ON cc.cluster_id = c.cluster_id
        """).fetchall()
        for cid, blob, label in rows:
            self._centroids[cid] = np.frombuffer(blob, dtype=np.float32).copy()
            self._centroid_labels[cid] = label

    # ── 1. Outlier Assignment ────────────────────────────────────────────

    def assign_outliers(self, min_similarity: float = 0.35,
                        batch_size: int = 128) -> dict:
        """
        Embed all unclustered clauses and assign each to the nearest
        cluster centroid via cosine similarity.

        Clauses below `min_similarity` are tagged as "Unclassifiable"
        rather than force-assigned to a poor match.

        Returns:
            {"assigned": int, "unclassifiable": int, "total": int,
             "assignments": {cluster_label: count, ...}}
        """
        if not self.embedder:
            raise ValueError("Embedder required for outlier assignment")
        if not self._centroids:
            raise ValueError("No cluster centroids loaded")

        # 1. Get unclustered clauses
        unclustered = self.store.conn.execute("""
            SELECT clause_id, agreement_id, full_text
            FROM clauses
            WHERE clause_type_id IS NULL AND LENGTH(full_text) >= 50
            ORDER BY clause_id
        """).fetchall()

        if not unclustered:
            logger.info("No unclustered clauses to assign")
            return {"assigned": 0, "unclassifiable": 0, "total": 0,
                    "assignments": {}}

        logger.info(f"Assigning {len(unclustered)} unclustered clauses "
                    f"(threshold={min_similarity})")

        # 2. Build centroid matrix
        centroid_ids = sorted(self._centroids.keys())
        centroid_matrix = np.stack([self._centroids[cid] for cid in centroid_ids])
        # Normalize centroids
        norms = np.linalg.norm(centroid_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        centroid_matrix = centroid_matrix / norms

        # 3. Embed and assign in batches
        assigned = 0
        unclassifiable = 0
        assignment_counts = Counter()

        # Create "Unclassifiable" cluster if not exists
        uncl_cluster_id = None

        for i in range(0, len(unclustered), batch_size):
            batch = unclustered[i:i + batch_size]
            texts = [row[2][:3000] for row in batch]  # cap text length

            # Embed batch
            embeddings = self.embedder.embed(
                texts, prefix="search_document",
                normalize=True, batch_size=64
            )

            # Cosine similarity to all centroids
            sims = embeddings @ centroid_matrix.T  # (batch, n_centroids)

            for j, (clause_id, agreement_id, _) in enumerate(batch):
                best_idx = int(np.argmax(sims[j]))
                best_sim = float(sims[j, best_idx])

                if best_sim >= min_similarity:
                    cluster_id = centroid_ids[best_idx]
                    label = self._centroid_labels[cluster_id]
                    self.store.conn.execute(
                        "UPDATE clauses SET clause_type_id = ? WHERE clause_id = ?",
                        [cluster_id, clause_id]
                    )
                    assigned += 1
                    assignment_counts[label] += 1
                else:
                    # Tag as unclassifiable
                    if uncl_cluster_id is None:
                        uncl_cluster_id = self._get_or_create_unclassifiable()
                    self.store.conn.execute(
                        "UPDATE clauses SET clause_type_id = ? WHERE clause_id = ?",
                        [uncl_cluster_id, clause_id]
                    )
                    unclassifiable += 1
                    assignment_counts["[Unclassifiable]"] += 1

            logger.info(f"  Batch {i // batch_size + 1}: "
                        f"{assigned} assigned, {unclassifiable} unclassifiable")

        result = {
            "assigned": assigned,
            "unclassifiable": unclassifiable,
            "total": len(unclustered),
            "assignments": dict(assignment_counts.most_common()),
        }
        logger.info(f"Outlier assignment complete: {result['assigned']} assigned, "
                    f"{result['unclassifiable']} unclassifiable out of {result['total']}")
        return result

    def _get_or_create_unclassifiable(self) -> str:
        """Get or create the [Unclassifiable] catch-all cluster."""
        row = self.store.conn.execute(
            "SELECT cluster_id FROM clusters WHERE label = '[Unclassifiable]'"
        ).fetchone()
        if row:
            return row[0]
        cluster_id = f"ct_{uuid.uuid4().hex[:8]}"
        self.store.conn.execute("""
            INSERT INTO clusters (cluster_id, label, level, parent_id, domain_id)
            VALUES (?, '[Unclassifiable]', 'clause_type', NULL, NULL)
        """, [cluster_id])
        return cluster_id

    # ── 2. Field Deduplication ───────────────────────────────────────────

    def deduplicate_fields(self, similarity_threshold: float = 0.85) -> dict:
        """
        Merge semantically redundant field definitions within each cluster.

        Uses embedding similarity on field name + description to detect
        near-duplicates like 'effective_date' vs 'agreement_effective_date'.

        Returns:
            {"clusters_processed": int, "fields_merged": int,
             "fields_before": int, "fields_after": int}
        """
        if not self.embedder:
            raise ValueError("Embedder required for field deduplication")

        all_fields = self.store.conn.execute("""
            SELECT fd.field_id, fd.cluster_id, fd.name, fd.description
            FROM field_definitions fd
            ORDER BY fd.cluster_id, fd.name
        """).fetchall()

        # Group by cluster
        cluster_fields = defaultdict(list)
        for fid, cid, name, desc in all_fields:
            cluster_fields[cid].append({
                "field_id": fid, "name": name,
                "description": desc or "",
            })

        fields_before = len(all_fields)
        fields_merged = 0
        clusters_processed = 0

        for cluster_id, fields in cluster_fields.items():
            if len(fields) < 2:
                continue

            # Embed field names + descriptions
            texts = [f"{f['name']}: {f['description']}" for f in fields]
            embeddings = self.embedder.embed(
                texts, prefix="search_document",
                normalize=True, batch_size=64
            )

            # Find merge candidates
            sim_matrix = embeddings @ embeddings.T
            merged_ids = set()

            for i in range(len(fields)):
                if fields[i]["field_id"] in merged_ids:
                    continue
                for j in range(i + 1, len(fields)):
                    if fields[j]["field_id"] in merged_ids:
                        continue
                    if sim_matrix[i, j] >= similarity_threshold:
                        # Merge j into i (keep i, delete j)
                        # Update extractions pointing to j
                        self.store.conn.execute(
                            "UPDATE extractions SET field_id = ? WHERE field_id = ?",
                            [fields[i]["field_id"], fields[j]["field_id"]]
                        )
                        # Delete the redundant field
                        self.store.conn.execute(
                            "DELETE FROM field_definitions WHERE field_id = ?",
                            [fields[j]["field_id"]]
                        )
                        merged_ids.add(fields[j]["field_id"])
                        fields_merged += 1
                        logger.debug(
                            f"  Merged '{fields[j]['name']}' into "
                            f"'{fields[i]['name']}' (sim={sim_matrix[i,j]:.3f})")

            if merged_ids:
                clusters_processed += 1

        fields_after = fields_before - fields_merged
        result = {
            "clusters_processed": clusters_processed,
            "fields_merged": fields_merged,
            "fields_before": fields_before,
            "fields_after": fields_after,
        }
        logger.info(f"Field dedup: {fields_merged} merged, "
                    f"{fields_before} → {fields_after}")
        return result

    # ── 3. Cluster Merging ───────────────────────────────────────────────

    def merge_overlapping_clusters(self, overlap_threshold: float = 0.80) -> dict:
        """
        Merge clusters whose field definitions overlap by ≥threshold.

        Overlap is measured as Jaccard similarity on normalized field names.
        When merging, the larger cluster absorbs the smaller one.

        Returns:
            {"pairs_evaluated": int, "clusters_merged": int,
             "clusters_before": int, "clusters_after": int,
             "merges": [{"absorbed": str, "into": str, "overlap": float}, ...]}
        """
        # Get field names per cluster
        field_data = self.store.conn.execute("""
            SELECT fd.cluster_id, c.label, LOWER(fd.name) as fname
            FROM field_definitions fd
            JOIN clusters c ON fd.cluster_id = c.cluster_id
        """).fetchall()

        cluster_fields = defaultdict(set)
        cluster_labels = {}
        for cid, label, fname in field_data:
            cluster_fields[cid].add(fname)
            cluster_labels[cid] = label

        # Get cluster sizes for tie-breaking
        cluster_sizes = {}
        for cid in cluster_fields:
            cnt = self.store.conn.execute(
                "SELECT COUNT(*) FROM clauses WHERE clause_type_id = ?", [cid]
            ).fetchone()[0]
            cluster_sizes[cid] = cnt

        clusters_before = len(cluster_fields)
        merges = []
        absorbed = set()  # cluster_ids that got absorbed

        # Evaluate all pairs
        pairs_evaluated = 0
        cluster_ids = sorted(cluster_fields.keys())

        for i, cid_a in enumerate(cluster_ids):
            if cid_a in absorbed:
                continue
            for cid_b in cluster_ids[i + 1:]:
                if cid_b in absorbed:
                    continue
                pairs_evaluated += 1

                fields_a = cluster_fields[cid_a]
                fields_b = cluster_fields[cid_b]
                if not fields_a or not fields_b:
                    continue

                jaccard = len(fields_a & fields_b) / len(fields_a | fields_b)
                if jaccard >= overlap_threshold:
                    # Absorb smaller into larger
                    if cluster_sizes.get(cid_a, 0) >= cluster_sizes.get(cid_b, 0):
                        keep, drop = cid_a, cid_b
                    else:
                        keep, drop = cid_b, cid_a

                    self._merge_cluster(drop, keep)
                    absorbed.add(drop)
                    merges.append({
                        "absorbed": cluster_labels.get(drop, drop),
                        "into": cluster_labels.get(keep, keep),
                        "overlap": round(jaccard, 3),
                    })
                    logger.info(
                        f"  Merged '{cluster_labels.get(drop)}' into "
                        f"'{cluster_labels.get(keep)}' (Jaccard={jaccard:.3f})")

        clusters_after = clusters_before - len(absorbed)
        result = {
            "pairs_evaluated": pairs_evaluated,
            "clusters_merged": len(merges),
            "clusters_before": clusters_before,
            "clusters_after": clusters_after,
            "merges": merges,
        }
        logger.info(f"Cluster merge: {len(merges)} merges, "
                    f"{clusters_before} → {clusters_after} clusters")
        return result

    def _merge_cluster(self, source_id: str, target_id: str):
        """Absorb source cluster into target: move clauses, fields, delete source."""
        # Move clauses
        self.store.conn.execute(
            "UPDATE clauses SET clause_type_id = ? WHERE clause_type_id = ?",
            [target_id, source_id]
        )
        # Move field definitions (skip if target already has same field name)
        existing = set(r[0] for r in self.store.conn.execute(
            "SELECT LOWER(name) FROM field_definitions WHERE cluster_id = ?",
            [target_id]).fetchall())

        source_fields = self.store.conn.execute(
            "SELECT field_id, name FROM field_definitions WHERE cluster_id = ?",
            [source_id]).fetchall()

        for fid, fname in source_fields:
            if fname.lower() in existing:
                # Redirect extractions, then delete redundant field
                target_fid = self.store.conn.execute(
                    "SELECT field_id FROM field_definitions "
                    "WHERE cluster_id = ? AND LOWER(name) = ?",
                    [target_id, fname.lower()]).fetchone()
                if target_fid:
                    self.store.conn.execute(
                        "UPDATE extractions SET field_id = ? WHERE field_id = ?",
                        [target_fid[0], fid])
                self.store.conn.execute(
                    "DELETE FROM field_definitions WHERE field_id = ?", [fid])
            else:
                # Move field to target cluster
                self.store.conn.execute(
                    "UPDATE field_definitions SET cluster_id = ? WHERE field_id = ?",
                    [target_id, fid])
                existing.add(fname.lower())

        # Move intent data
        try:
            self.store.conn.execute(
                "UPDATE clause_intents SET clause_type_id = ? WHERE clause_type_id = ?",
                [target_id, source_id])
        except Exception:
            pass

        # Delete source cluster
        try:
            self.store.conn.execute(
                "DELETE FROM cluster_centroids WHERE cluster_id = ?", [source_id])
        except Exception:
            pass
        self.store.conn.execute(
            "DELETE FROM clusters WHERE cluster_id = ?", [source_id])

    # ── 4. Contextual Label Synthesis ────────────────────────────────────

    def synthesize_labels(self, max_clusters: int | None = None) -> dict:
        """
        Replace keyword-based cluster labels with LLM-synthesized names
        derived from field definitions, sample clause texts, and domain context.

        Returns:
            {"relabeled": int, "labels": {old: new, ...}}
        """
        clusters = self.store.conn.execute("""
            SELECT c.cluster_id, c.label,
                   (SELECT COUNT(*) FROM clauses cl WHERE cl.clause_type_id = c.cluster_id) as n
            FROM clusters c
            WHERE c.label != '[Unclassifiable]'
            ORDER BY n DESC
        """).fetchall()

        if max_clusters:
            clusters = clusters[:max_clusters]

        client = self._get_azure_client()
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")

        labels = {}
        relabeled = 0

        for cluster_id, old_label, n_clauses in clusters:
            # Get field names for context
            fields = self.store.conn.execute(
                "SELECT name FROM field_definitions WHERE cluster_id = ? ORDER BY name",
                [cluster_id]).fetchall()
            field_names = [f[0] for f in fields]

            # Get sample clause texts
            samples = self.store.conn.execute("""
                SELECT LEFT(full_text, 200) FROM clauses
                WHERE clause_type_id = ? LIMIT 3
            """, [cluster_id]).fetchall()
            sample_texts = [s[0] for s in samples]

            prompt = f"""You are a legal document taxonomy expert. Given the following information about a cluster of contract clauses, generate a precise, professional document-type label.

Cluster contains {n_clauses} clauses.
Current label: "{old_label}"
Fields extracted: {', '.join(field_names[:12])}

Sample excerpts:
{chr(10).join(f'- {t}...' for t in sample_texts)}

Requirements:
- The label should be 2-5 words, title case
- It should describe the DOCUMENT TYPE or CLAUSE CATEGORY, not just keywords
- Be specific: "Master Service Agreement" not "Agreement Terms"
- If it's a clause type (like indemnification, termination), name the clause type

Respond with ONLY the label, nothing else."""

            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_completion_tokens=30,
                )
                new_label = response.choices[0].message.content.strip().strip('"\'')
                if new_label and len(new_label) < 60 and new_label != old_label:
                    self.store.conn.execute(
                        "UPDATE clusters SET label = ? WHERE cluster_id = ?",
                        [new_label, cluster_id])
                    labels[old_label] = new_label
                    relabeled += 1
                    logger.info(f"  '{old_label}' → '{new_label}'")
            except Exception as e:
                logger.warning(f"Label synthesis failed for {cluster_id}: {e}")

        result = {"relabeled": relabeled, "labels": labels}
        logger.info(f"Label synthesis: {relabeled} clusters relabeled")
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Phase 2: Feature Activation
    # ═══════════════════════════════════════════════════════════════════════

    # ── 5. Feedback Loop ─────────────────────────────────────────────────

    def register_correction(self, clause_id: str, field_id: str,
                            old_value: str, new_value: str,
                            user_id: str = "system") -> str:
        """
        Record a user correction to an extraction. This feeds back into
        field definitions by adjusting examples and description.

        Returns the correction_id.
        """
        correction_id = f"cor_{uuid.uuid4().hex[:10]}"

        # Store correction
        self.store.conn.execute("""
            INSERT INTO corrections (correction_id, clause_id, field_id,
                                     old_value, new_value, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [correction_id, clause_id, field_id, old_value, new_value, user_id])

        # Update extraction
        self.store.conn.execute("""
            UPDATE extractions SET value = ?, was_corrected = TRUE,
                                   original_value = COALESCE(original_value, value)
            WHERE clause_id = ? AND field_id = ?
        """, [new_value, clause_id, field_id])

        # Update field definition examples with corrected value
        existing_examples = self.store.conn.execute(
            "SELECT examples FROM field_definitions WHERE field_id = ?",
            [field_id]).fetchone()
        if existing_examples and existing_examples[0]:
            try:
                examples = json.loads(existing_examples[0])
            except (json.JSONDecodeError, TypeError):
                examples = []
        else:
            examples = []

        # Add corrected value as positive example, limit to 10
        if new_value not in examples:
            examples.append(new_value)
            examples = examples[-10:]
            self.store.conn.execute(
                "UPDATE field_definitions SET examples = ? WHERE field_id = ?",
                [json.dumps(examples), field_id])

        logger.info(f"Correction registered: {field_id} '{old_value}' → '{new_value}'")
        return correction_id

    def apply_corrections_to_definitions(self) -> dict:
        """
        Batch process all corrections to refine field definitions.
        Groups corrections by field and updates description/examples.

        Returns:
            {"fields_updated": int, "total_corrections": int}
        """
        corrections = self.store.conn.execute("""
            SELECT field_id, COUNT(*) as n,
                   ARRAY_AGG(new_value) as values
            FROM corrections
            GROUP BY field_id
            HAVING COUNT(*) >= 2
        """).fetchall()

        fields_updated = 0
        total_corrections = 0

        for field_id, n, values in corrections:
            total_corrections += n
            # Update examples with most common corrected values
            value_counts = Counter(values)
            top_examples = [v for v, _ in value_counts.most_common(10)]

            self.store.conn.execute(
                "UPDATE field_definitions SET examples = ? WHERE field_id = ?",
                [json.dumps(top_examples), field_id])
            fields_updated += 1

        return {"fields_updated": fields_updated,
                "total_corrections": total_corrections}

    # ── 6. Composite Field Logic ─────────────────────────────────────────

    def define_composite(self, cluster_id: str, name: str,
                         expression: str, description: str = "") -> str:
        """
        Define a composite field that derives its value from other fields.

        Examples:
            define_composite("ct_abc", "last_opt_out_date",
                             "expiration_date - notice_period",
                             "Last date to opt out of auto-renewal")
            define_composite("ct_abc", "total_liability",
                             "liability_cap * number_of_units",
                             "Maximum aggregate liability")

        The expression uses field names and supports:
            - Arithmetic: +, -, *, /
            - Date math: field - duration
            - Conditionals: IF(field, then_val, else_val)
        """
        composite_id = f"comp_{uuid.uuid4().hex[:10]}"

        # Validate that referenced fields exist
        # Extract field references from expression
        import re
        field_refs = re.findall(r'\b([a-z_]+(?:_[a-z_]+)+)\b', expression.lower())
        existing_fields = set(r[0] for r in self.store.conn.execute(
            "SELECT LOWER(name) FROM field_definitions WHERE cluster_id = ?",
            [cluster_id]).fetchall())

        missing = [f for f in field_refs if f not in existing_fields
                   and f not in ("if", "then", "else")]
        if missing:
            logger.warning(f"Composite '{name}' references missing fields: {missing}")

        self.store.conn.execute("""
            INSERT INTO composite_definitions
            (composite_id, cluster_id, name, expression, description)
            VALUES (?, ?, ?, ?, ?)
        """, [composite_id, cluster_id, name, expression, description])

        logger.info(f"Composite field defined: {name} = {expression}")
        return composite_id

    def evaluate_composite(self, composite_id: str,
                           agreement_id: str) -> str | None:
        """
        Evaluate a composite field for a specific agreement.

        Returns the computed value or None if inputs are missing.
        """
        comp = self.store.conn.execute(
            "SELECT cluster_id, name, expression FROM composite_definitions "
            "WHERE composite_id = ?", [composite_id]).fetchone()
        if not comp:
            return None

        cluster_id, name, expression = comp

        # Get field values for this agreement + cluster
        values = self.store.conn.execute("""
            SELECT fd.name, e.value
            FROM extractions e
            JOIN field_definitions fd ON e.field_id = fd.field_id
            WHERE e.agreement_id = ? AND fd.cluster_id = ?
        """, [agreement_id, cluster_id]).fetchall()

        field_values = {name.lower(): val for name, val in values}

        # Simple expression evaluation (safe subset)
        try:
            result = self._safe_eval(expression, field_values)
            return str(result)
        except Exception as e:
            logger.warning(f"Composite eval failed: {name} = {expression}: {e}")
            return None

    def _safe_eval(self, expression: str, field_values: dict) -> str:
        """
        Safely evaluate a composite expression with field substitution.
        Supports date arithmetic and basic math.
        """
        import re
        from datetime import datetime, timedelta

        result = expression.lower()

        # Substitute field values
        for field_name, value in field_values.items():
            result = result.replace(field_name, repr(value))

        # Try date arithmetic: "date - N days/months"
        date_pattern = r"'(\d{4}-\d{2}-\d{2})'\s*-\s*'?(\d+)\s*(days?|months?|years?)'?"
        match = re.search(date_pattern, result)
        if match:
            date_str, amount, unit = match.groups()
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            amount = int(amount)
            if "day" in unit:
                dt -= timedelta(days=amount)
            elif "month" in unit:
                month = dt.month - amount
                year = dt.year + (month - 1) // 12
                month = (month - 1) % 12 + 1
                dt = dt.replace(year=year, month=month)
            elif "year" in unit:
                dt = dt.replace(year=dt.year - amount)
            return dt.strftime("%Y-%m-%d")

        # Simple numeric math
        try:
            # Only allow digits, operators, spaces, parens
            if re.match(r'^[\d\s\+\-\*\/\.\(\)]+$', result):
                return str(eval(result))  # safe: validated input
        except Exception:
            pass

        return result

    # ── Utilities ────────────────────────────────────────────────────────

    def _get_azure_client(self):
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )

    def get_stats(self) -> dict:
        """Return current ontology statistics."""
        total_clauses = self.store.conn.execute(
            "SELECT COUNT(*) FROM clauses").fetchone()[0]
        clustered = self.store.conn.execute(
            "SELECT COUNT(*) FROM clauses WHERE clause_type_id IS NOT NULL").fetchone()[0]
        n_clusters = self.store.conn.execute(
            "SELECT COUNT(*) FROM clusters").fetchone()[0]
        n_fields = self.store.conn.execute(
            "SELECT COUNT(*) FROM field_definitions").fetchone()[0]
        n_extractions = self.store.conn.execute(
            "SELECT COUNT(*) FROM extractions").fetchone()[0]
        avg_confidence = self.store.conn.execute(
            "SELECT AVG(confidence) FROM extractions").fetchone()[0] or 0

        return {
            "total_clauses": total_clauses,
            "clustered": clustered,
            "unclustered": total_clauses - clustered,
            "coverage_pct": round(100 * clustered / total_clauses, 1) if total_clauses else 0,
            "n_clusters": n_clusters,
            "n_fields": n_fields,
            "n_extractions": n_extractions,
            "avg_confidence": round(avg_confidence, 3),
        }
