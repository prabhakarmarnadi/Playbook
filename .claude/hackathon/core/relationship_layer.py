"""Unified query interface over ARM relationship outputs.

Consumers (field_discovery, extractor, context_graph) call this layer
to get clause dependencies, field correlations, and term packages.
"""
import json
import logging
from typing import Any

from core.store import ClusteringStore

logger = logging.getLogger(__name__)


class RelationshipLayer:
    def __init__(self, store: ClusteringStore):
        self.store = store
        self._rules_cache: list[dict] | None = None
        self._packages_cache: list[dict] | None = None

    def _load_rules(self) -> list[dict]:
        if self._rules_cache is not None:
            return self._rules_cache
        try:
            rows = self.store.conn.execute(
                "SELECT rule_id, antecedent, consequent, rule_type, "
                "support, confidence, lift, cluster_id FROM arm_rules"
            ).fetchall()
            self._rules_cache = [
                {
                    "rule_id": r[0],
                    "antecedent": json.loads(r[1]),
                    "consequent": json.loads(r[2]),
                    "rule_type": r[3],
                    "support": r[4],
                    "confidence": r[5],
                    "lift": r[6],
                    "cluster_id": r[7],
                }
                for r in rows
            ]
        except Exception:
            self._rules_cache = []
        return self._rules_cache

    def _load_packages(self) -> list[dict]:
        if self._packages_cache is not None:
            return self._packages_cache
        try:
            rows = self.store.conn.execute(
                "SELECT package_id, clause_types, support, size, cluster_id "
                "FROM arm_term_packages"
            ).fetchall()
            self._packages_cache = [
                {
                    "package_id": r[0],
                    "clause_types": json.loads(r[1]),
                    "support": r[2],
                    "size": r[3],
                    "cluster_id": r[4],
                }
                for r in rows
            ]
        except Exception:
            self._packages_cache = []
        return self._packages_cache

    def get_related_clause_types(self, clause_type: str) -> list[dict]:
        rules = self._load_rules()
        related = []
        seen = set()
        for rule in rules:
            if rule["rule_type"] != "clause_dependency":
                continue
            if clause_type in rule["antecedent"]:
                for ct in rule["consequent"]:
                    if ct not in seen:
                        related.append({
                            "clause_type": ct,
                            "confidence": rule["confidence"],
                            "lift": rule["lift"],
                            "direction": "implies",
                        })
                        seen.add(ct)
            elif clause_type in rule["consequent"]:
                for ct in rule["antecedent"]:
                    if ct not in seen:
                        related.append({
                            "clause_type": ct,
                            "confidence": rule["confidence"],
                            "lift": rule["lift"],
                            "direction": "implied_by",
                        })
                        seen.add(ct)
        related.sort(key=lambda x: x["confidence"], reverse=True)
        return related

    def get_implied_fields(self, clause_types: list[str]) -> list[dict]:
        rules = self._load_rules()
        implied = []
        for rule in rules:
            if rule["rule_type"] not in ("field_correlation", "cross_clause_field"):
                continue
            ant_set = set(rule["antecedent"])
            for item in ant_set:
                clause = item.split(".")[0] if "." in item else ""
                if clause in clause_types:
                    for cons_item in rule["consequent"]:
                        implied.append({
                            "field_item": cons_item,
                            "confidence": rule["confidence"],
                            "lift": rule["lift"],
                            "source_rule": rule["rule_id"],
                        })
                    break
        implied.sort(key=lambda x: x["confidence"], reverse=True)
        return implied

    def get_field_correlations(self, field_name: str, cluster_id: str | None = None) -> list[dict]:
        rules = self._load_rules()
        correlations = []
        for rule in rules:
            if rule["rule_type"] not in ("field_correlation", "cross_clause_field"):
                continue
            if cluster_id and rule.get("cluster_id") and rule["cluster_id"] != cluster_id:
                continue
            all_items = rule["antecedent"] + rule["consequent"]
            if any(field_name in item for item in all_items):
                correlations.append({
                    "antecedent": rule["antecedent"],
                    "consequent": rule["consequent"],
                    "confidence": rule["confidence"],
                    "lift": rule["lift"],
                })
        return correlations

    def get_term_packages(self) -> list[dict]:
        return self._load_packages()

    def get_obligation_chains(self, clause_type: str) -> list[dict]:
        rules = self._load_rules()
        chains = []
        for rule in rules:
            if rule["rule_type"] != "obligation_chain":
                continue
            if clause_type in rule["antecedent"] or clause_type in rule["consequent"]:
                chains.append(rule)
        return chains

    def mine_from_assignments(self, cluster_assignments: dict[str, list[str]],
                               cluster_labels: dict[str, int] | None = None,
                               pipeline_run_id: str = "") -> None:
        from core.arm.arm_miner import (
            build_clause_presence_matrix,
            discover_term_packages,
            mine_global,
            mine_per_cluster,
        )
        matrix = build_clause_presence_matrix(cluster_assignments)
        if matrix.empty:
            return
        global_rules = mine_global(matrix)
        for rule in global_rules:
            self._persist_rule(rule, pipeline_run_id)
        if cluster_labels:
            cluster_rules = mine_per_cluster(matrix, cluster_labels)
            for cid, rules in cluster_rules.items():
                for rule in rules:
                    rule["cluster_id"] = str(cid)
                    self._persist_rule(rule, pipeline_run_id)
        packages = discover_term_packages(matrix)
        for pkg in packages:
            self._persist_package(pkg, pipeline_run_id)
        self._rules_cache = None
        self._packages_cache = None
        logger.info(f"Persisted {len(global_rules)} global rules, {len(packages)} term packages")

    def mine_field_correlations(self, extractions: dict[str, dict],
                                 pipeline_run_id: str = "") -> None:
        from core.arm.field_arm_miner import build_field_itemsets, mine_field_rules
        matrix = build_field_itemsets(extractions)
        if matrix.empty:
            return
        rules = mine_field_rules(matrix)
        for rule in rules:
            self._persist_rule(rule, pipeline_run_id)
        self._rules_cache = None
        logger.info(f"Persisted {len(rules)} field rules")

    def _persist_rule(self, rule: dict, pipeline_run_id: str) -> None:
        self.store.conn.execute(
            "INSERT OR REPLACE INTO arm_rules "
            "(rule_id, antecedent, consequent, rule_type, support, confidence, lift, cluster_id, pipeline_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                rule["rule_id"],
                json.dumps(rule["antecedent"]),
                json.dumps(rule["consequent"]),
                rule["rule_type"],
                rule.get("support"),
                rule.get("confidence"),
                rule.get("lift"),
                rule.get("cluster_id"),
                pipeline_run_id,
            ],
        )

    def _persist_package(self, pkg: dict, pipeline_run_id: str) -> None:
        self.store.conn.execute(
            "INSERT OR REPLACE INTO arm_term_packages "
            "(package_id, clause_types, support, size, cluster_id, pipeline_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                pkg["package_id"],
                json.dumps(pkg["clause_types"]),
                pkg.get("support"),
                pkg.get("size"),
                pkg.get("cluster_id"),
                pipeline_run_id,
            ],
        )
