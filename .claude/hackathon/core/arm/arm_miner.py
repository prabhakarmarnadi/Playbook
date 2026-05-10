"""Clause-level Association Rule Mining using FP-Growth.

Discovers clause co-occurrence patterns, term packages, and dependency chains.
"""
import json
import logging
import os
import uuid

import numpy as np
import pandas as pd

from core.arm.configs import CLAUSE_ARM_DEFAULTS

logger = logging.getLogger(__name__)

try:
    from mlxtend.frequent_patterns import association_rules, fpgrowth
    _MLXTEND_AVAILABLE = True
except ImportError:
    _MLXTEND_AVAILABLE = False
    logger.warning("mlxtend not available — ARM mining disabled")


# Skip ARM when the corpus has fewer than this many transactions (documents).
# FP-Growth on tiny but wide matrices (few docs × many clause types) can
# explode combinatorially and produces no statistically meaningful rules anyway.
ARM_MIN_TRANSACTIONS = int(os.getenv("ARM_MIN_TRANSACTIONS", "20"))


def mlxtend_available() -> bool:
    return _MLXTEND_AVAILABLE


def _too_small(matrix: pd.DataFrame, label: str) -> bool:
    """Bail out before fpgrowth if the matrix is too small to produce
    statistically meaningful rules. Logs once per stage."""
    if matrix.empty:
        return True
    n_docs = matrix.shape[0]
    if n_docs < ARM_MIN_TRANSACTIONS:
        logger.info(
            f"ARM/{label}: skipping ({n_docs} transactions < ARM_MIN_TRANSACTIONS={ARM_MIN_TRANSACTIONS}). "
            "Tiny corpora cannot produce statistically meaningful association rules."
        )
        return True
    return False


def build_clause_presence_matrix(
    cluster_assignments: dict[str, list[str]],
) -> pd.DataFrame:
    all_clause_types = sorted({ct for cts in cluster_assignments.values() for ct in cts})
    if not all_clause_types:
        return pd.DataFrame()
    rows = []
    for doc_id, clause_types in cluster_assignments.items():
        row = {ct: 1 if ct in clause_types else 0 for ct in all_clause_types}
        row["doc_id"] = doc_id
        rows.append(row)
    df = pd.DataFrame(rows).set_index("doc_id")
    return df


def mine_global(
    matrix: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    if not _MLXTEND_AVAILABLE or _too_small(matrix, "mine_global"):
        return []
    cfg = config or CLAUSE_ARM_DEFAULTS["global"]
    min_support = cfg.get("min_support", 0.20)
    min_confidence = cfg.get("min_confidence", 0.7)
    min_lift = cfg.get("min_lift", 1.8)
    df_bool = matrix.astype(bool)
    try:
        itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
        if itemsets.empty:
            return []
        rules_df = association_rules(itemsets, metric="lift", min_threshold=min_lift)
    except Exception as e:
        logger.warning(f"FP-Growth failed: {e}")
        return []
    rules_df = rules_df[rules_df["confidence"] >= min_confidence]
    rules = []
    for _, row in rules_df.iterrows():
        rules.append({
            "rule_id": f"R_{uuid.uuid4().hex[:8]}",
            "antecedent": sorted(list(row["antecedents"])),
            "consequent": sorted(list(row["consequents"])),
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "rule_type": "clause_dependency",
        })
    logger.info(f"Mined {len(rules)} global clause rules")
    return rules


def mine_per_cluster(
    matrix: pd.DataFrame,
    cluster_labels: dict[str, int],
    config: dict | None = None,
) -> dict[int, list[dict]]:
    if not _MLXTEND_AVAILABLE or matrix.empty:
        return {}
    cfg = config or CLAUSE_ARM_DEFAULTS
    size_tiers = cfg.get("per_cluster_by_size", CLAUSE_ARM_DEFAULTS["per_cluster_by_size"])
    cluster_rules = {}
    unique_clusters = set(cluster_labels.values())
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        doc_ids = [d for d, c in cluster_labels.items() if c == cluster_id]
        valid_ids = [d for d in doc_ids if d in matrix.index]
        cluster_matrix = matrix.loc[valid_ids]
        n_docs = len(cluster_matrix)
        if n_docs < 10:
            continue
        if n_docs >= size_tiers["large"]["min_docs"]:
            min_support = size_tiers["large"]["min_support"]
        elif n_docs >= size_tiers["medium"]["min_docs"]:
            min_support = size_tiers["medium"]["min_support"]
        else:
            min_support = size_tiers["small"]["min_support"]
        col_sums = cluster_matrix.sum()
        valid_cols = col_sums[(col_sums > 0) & (col_sums < n_docs)].index
        if len(valid_cols) < 2:
            continue
        filtered = cluster_matrix[valid_cols].astype(bool)
        try:
            itemsets = fpgrowth(filtered, min_support=min_support, use_colnames=True)
            if itemsets.empty:
                continue
            rules_df = association_rules(itemsets, metric="lift", min_threshold=1.5)
        except Exception:
            continue
        rules = []
        for _, row in rules_df.iterrows():
            rules.append({
                "rule_id": f"RC{cluster_id}_{uuid.uuid4().hex[:8]}",
                "antecedent": sorted(list(row["antecedents"])),
                "consequent": sorted(list(row["consequents"])),
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
                "rule_type": "clause_dependency",
                "cluster_id": str(cluster_id),
            })
        if rules:
            cluster_rules[cluster_id] = rules
    return cluster_rules


def discover_term_packages(
    matrix: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    if not _MLXTEND_AVAILABLE or _too_small(matrix, "discover_term_packages"):
        return []
    cfg = config or CLAUSE_ARM_DEFAULTS.get("term_packages", {})
    min_support = cfg.get("min_support", 0.15)
    min_size = cfg.get("min_itemset_size", 3)
    df_bool = matrix.astype(bool)
    try:
        itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
    except Exception as e:
        logger.warning(f"FP-Growth for term packages failed: {e}")
        return []
    packages = []
    for _, row in itemsets.iterrows():
        items = sorted(list(row["itemsets"]))
        if len(items) >= min_size:
            packages.append({
                "package_id": f"TP_{uuid.uuid4().hex[:8]}",
                "clause_types": items,
                "support": float(row["support"]),
                "size": len(items),
            })
    packages.sort(key=lambda x: x["support"], reverse=True)
    logger.info(f"Discovered {len(packages)} term packages")
    return packages
