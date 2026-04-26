"""ARM threshold configuration defaults."""
from config import (
    ARM_MIN_CONFIDENCE,
    ARM_MIN_LIFT,
    ARM_MIN_SUPPORT,
    FIELD_ARM_MIN_CONFIDENCE,
    FIELD_ARM_MIN_LIFT,
    FIELD_ARM_MIN_SUPPORT,
)

CLAUSE_ARM_DEFAULTS = {
    "global": {
        "min_support": ARM_MIN_SUPPORT,
        "min_confidence": ARM_MIN_CONFIDENCE,
        "min_lift": ARM_MIN_LIFT,
    },
    "per_cluster_by_size": {
        "large": {"min_docs": 100, "min_support": 0.15},
        "medium": {"min_docs": 50, "min_support": 0.20},
        "small": {"min_docs": 20, "min_support": 0.25},
    },
    "term_packages": {
        "min_itemset_size": 3,
        "min_support": 0.15,
    },
}

FIELD_ARM_DEFAULTS = {
    "global": {
        "min_support": FIELD_ARM_MIN_SUPPORT,
        "min_confidence": FIELD_ARM_MIN_CONFIDENCE,
        "min_lift": FIELD_ARM_MIN_LIFT,
        "max_rules": 1000,
    },
    "per_cluster": {
        "min_support": FIELD_ARM_MIN_SUPPORT,
        "min_confidence": FIELD_ARM_MIN_CONFIDENCE,
        "min_lift": FIELD_ARM_MIN_LIFT,
        "max_rules": 500,
    },
    "normalization": {
        "numeric_bins": [0, 7, 30, 60, 90, 180, 365, 999999],
        "numeric_labels": ["0-7d", "8-30d", "31-60d", "61-90d", "91-180d", "181-365d", "365d+"],
    },
}
