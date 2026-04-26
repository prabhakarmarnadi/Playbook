"""Field-value Association Rule Mining.

Discovers field-value correlations within and across clause types.
"""
import json
import logging
import uuid

import numpy as np
import pandas as pd

from core.arm.configs import FIELD_ARM_DEFAULTS

logger = logging.getLogger(__name__)

try:
    from mlxtend.frequent_patterns import association_rules, fpgrowth
    _MLXTEND_AVAILABLE = True
except ImportError:
    _MLXTEND_AVAILABLE = False


def normalize_field_value(value, field_name: str, config: dict | None = None) -> str | None:
    if value is None:
        return None
    cfg = config or FIELD_ARM_DEFAULTS.get("normalization", {})
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        bins = cfg.get("numeric_bins", [0, 7, 30, 60, 90, 180, 365, 999999])
        labels = cfg.get("numeric_labels", ["0-7d", "8-30d", "31-60d", "61-90d", "91-180d", "181-365d", "365d+"])
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                return labels[i] if i < len(labels) else str(value)
        return labels[-1] if labels else str(value)
    if isinstance(value, str):
        return value.lower().strip()[:50]
    if isinstance(value, list):
        return "+".join(str(v) for v in value[:3])
    return str(value)[:50]


def build_field_itemsets(
    extractions: dict[str, dict],
    config: dict | None = None,
) -> pd.DataFrame:
    cfg = config or FIELD_ARM_DEFAULTS
    rows = []
    for doc_id, clause_data in extractions.items():
        row = {"doc_id": doc_id}
        for clause_type, fields in clause_data.items():
            if not isinstance(fields, dict):
                continue
            for field_name, value in fields.items():
                normalized = normalize_field_value(value, field_name, cfg.get("normalization"))
                if normalized is not None:
                    col = f"{clause_type}.{field_name}={normalized}"
                    row[col] = 1
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("doc_id").fillna(0).astype(int)
    col_sums = df.sum()
    valid_cols = col_sums[col_sums >= 5].index
    return df[valid_cols] if len(valid_cols) > 0 else pd.DataFrame()


def mine_field_rules(
    matrix: pd.DataFrame,
    config: dict | None = None,
) -> list[dict]:
    if not _MLXTEND_AVAILABLE or matrix.empty:
        return []
    cfg = config or FIELD_ARM_DEFAULTS.get("global", {})
    min_support = cfg.get("min_support", 0.05)
    min_confidence = cfg.get("min_confidence", 0.5)
    min_lift = cfg.get("min_lift", 1.5)
    max_rules = cfg.get("max_rules", 1000)
    df_bool = matrix.astype(bool)
    try:
        itemsets = fpgrowth(df_bool, min_support=min_support, use_colnames=True)
        if itemsets.empty:
            return []
        rules_df = association_rules(itemsets, metric="lift", min_threshold=min_lift)
    except Exception as e:
        logger.warning(f"Field ARM failed: {e}")
        return []
    rules_df = rules_df[rules_df["confidence"] >= min_confidence]
    rules_df = rules_df.head(max_rules)
    rules = []
    for _, row in rules_df.iterrows():
        antecedent = sorted(list(row["antecedents"]))
        consequent = sorted(list(row["consequents"]))
        ant_clauses = {item.split(".")[0] for item in antecedent if "." in item}
        con_clauses = {item.split(".")[0] for item in consequent if "." in item}
        is_cross_clause = bool(ant_clauses and con_clauses and ant_clauses != con_clauses)
        rules.append({
            "rule_id": f"FR_{uuid.uuid4().hex[:8]}",
            "antecedent": antecedent,
            "consequent": consequent,
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "rule_type": "cross_clause_field" if is_cross_clause else "field_correlation",
        })
    logger.info(f"Mined {len(rules)} field rules")
    return rules
